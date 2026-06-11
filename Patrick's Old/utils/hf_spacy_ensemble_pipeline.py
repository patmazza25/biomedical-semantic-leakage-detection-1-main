# utils/hf_spacy_ensemble_pipeline.py
# Pure spaCy/scispaCy ensemble — NO HuggingFace Pipeline dependency.
# Provides SpacyEnsembleNERPipeline and EnsembleConfig for concept extraction.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

log = logging.getLogger(__name__)

# ================================================================================
# Config
# ================================================================================
@dataclass
class EnsembleConfig:
    # REQUIRED: spaCy pipelines to ensemble (install via pip)
    spacy_model_names: List[str]

    # UMLS linker (scispaCy)
    enable_linker: bool = True
    top_k_linker: int = 5
    allowed_semantic_types: Optional[List[str]] = None   # e.g., ["T047", "T121"]; None => don't filter by type
    allowed_kb_sources: Optional[List[str]] = None       # e.g., ["MSH","SNOMEDCT_US","RXNORM"]; None => accept all
    min_link_score: float = 0.0                          # drop link candidates below this score

    # Optional re-ranker (sentence-transformers CrossEncoder)
    enable_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Merge + filtering strategies
    merge_strategy: str = "surface"   # "surface" | "span" | "fuzzy"
    min_span_len: int = 1
    allowed_labels: Optional[List[str]] = None          # if set, only keep these NER labels
    stop_terms: Optional[List[str]] = None              # lowercase surface forms to reject
    drop_if_no_cui_when_linking: bool = False           # if True (+linker), entities without CUI are dropped
    fuzzy_jaccard_threshold: float = 0.72               # for merge_strategy="fuzzy"

    # Confidence weighting + thresholds
    w_model_score: float = 0.50
    w_link_score: float = 0.35
    w_rerank_score: float = 0.15
    min_confidence: float = 0.0                          # drop entities with final confidence < threshold

    # Execution
    max_workers_models: int = 2                          # parallelism across models
    max_workers_texts: int = 4                           # parallelism across batch of texts
    random_seed: Optional[int] = None                    # set for determinism

    # Debug / diagnostics
    enable_debug: bool = False
    collect_stats: bool = True

    def to_json(self) -> str:
        import json
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(s: str) -> "EnsembleConfig":
        import json
        return EnsembleConfig(**json.loads(s))

# ================================================================================
# Global caches & lazy loading
# ================================================================================
_SPACY = None
# name -> (nlp, linker, linker_pipe_name)
_SPACY_MODELS: Dict[str, Tuple[Any, Optional[Any], Optional[str]]] = {}
_SPACY_MODEL_LOCK = threading.Lock()

def _lazy_spacy():
    global _SPACY
    if _SPACY is None:
        import spacy  # type: ignore
        _SPACY = spacy
    return _SPACY

# Try to import scispaCy bits and register a local fallback factory if needed
try:
    import scispacy.linking  # noqa: F401
except Exception:
    pass

try:
    from scispacy.abbreviation import AbbreviationDetector  # type: ignore
except Exception:
    AbbreviationDetector = None  # type: ignore

try:
    from scispacy.linking import EntityLinker  # type: ignore
    from spacy.language import Language

    @Language.factory("umls_linker")
    def create_umls_linker(nlp, name):
        # Local fallback factory if "scispacy_linker" isn't available
        return EntityLinker(resolve_abbreviations=True, name="umls")
except Exception:
    EntityLinker = None  # type: ignore

def _ensure_abbrev(nlp) -> None:
    """Add an abbreviation detector if possible; ignore failures."""
    try:
        if "abbreviation_detector" not in nlp.pipe_names:
            nlp.add_pipe("abbreviation_detector")
    except Exception:
        if AbbreviationDetector is not None:
            try:
                nlp.add_pipe(AbbreviationDetector(nlp))
            except Exception:
                pass

def _attach_umls_linker(nlp) -> Optional[str]:
    """
    Ensure a UMLS linker component is present in the pipeline.
    Returns the pipe name if attached/present, else None.
    Preference: official "scispacy_linker" → fallback "umls_linker".
    """
    # Already present?
    if "scispacy_linker" in nlp.pipe_names:
        return "scispacy_linker"
    if "umls_linker" in nlp.pipe_names:
        return "umls_linker"

    # Try official factory first
    try:
        nlp.add_pipe(
            "scispacy_linker",
            config={"resolve_abbreviations": True, "linker_name": "umls"},
            last=True,
        )
        return "scispacy_linker"
    except Exception:
        pass

    # Fallback to local factory (if available)
    if EntityLinker is not None:
        try:
            nlp.add_pipe("umls_linker", last=True)
            return "umls_linker"
        except Exception:
            pass

    return None

def _load_spacy_model(name: str) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    """
    Load a spaCy pipeline by name once and cache it.
    Returns (nlp, linker, linker_pipe_name).
    - For SciSpaCy NER-only models (bc5cdr, bionlp13cg, craft, jnlpba), no linker is attached.
    - For en_core_sci_scibert, we attach a UMLS linker if available.
    """
    if not name:
        return None, None, None

    with _SPACY_MODEL_LOCK:
        if name in _SPACY_MODELS:
            return _SPACY_MODELS[name]

        spacy = _lazy_spacy()
        try:
            nlp = spacy.load(name)
            _ensure_abbrev(nlp)
            linker, linker_pipe = None, None
            # Only attach linker to SciBERT (our linking backbone)
            if "sci" in name.lower():
                try:
                    linker_pipe = _attach_umls_linker(nlp)
                    if linker_pipe:
                        linker = nlp.get_pipe(linker_pipe)
                        log.info("Attached UMLS linker (%s) to model '%s'", linker_pipe, name)
                except Exception as e:
                    log.warning("Could not attach UMLS linker to '%s': %s", name, e)

            _SPACY_MODELS[name] = (nlp, linker, linker_pipe)
            return _SPACY_MODELS[name]

        except Exception as e:
            log.warning("Could not load spaCy model '%s': %s", name, e)
            _SPACY_MODELS[name] = (None, None, None)
            return _SPACY_MODELS[name]

def _pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except Exception:
        return 0.5

class _CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model = None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            device = _pick_device()
            self.model = CrossEncoder(model_name, device=device)
            log.info("Loaded CrossEncoder reranker '%s' on %s", model_name, device)
        except Exception as e:
            log.warning("Reranker unavailable ('%s'): %s", model_name, e)
            self.model = None

    def score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Returns scores in [0,1] via sigmoid of model output.
        Keeps length stable even if model is missing.
        """
        if not self.model or not pairs:
            return [0.0] * len(pairs)
        try:
            scores = self.model.predict(pairs, convert_to_numpy=True)  # shape [N]
            return [float(_sigmoid(float(s))) for s in scores]
        except Exception as e:
            log.warning("Reranker scoring failed: %s", e)
            return [0.0] * len(pairs)

# ================================================================================
# Helpers
# ================================================================================
def _token_jaccard(a: str, b: str) -> float:
    aset = set(a.lower().split())
    bset = set(b.lower().split())
    if not aset and not bset:
        return 1.0
    if not aset or not bset:
        return 0.0
    inter = len(aset & bset)
    union = len(aset | bset)
    return inter / union if union else 0.0

def _normalize_surface(s: str) -> str:
    return (s or "").strip().lower()

def _mk_entity(
    text: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    label: Optional[str] = None,
    cui: Optional[str] = None,
    canonical: Optional[str] = None,
    semantic_types: Optional[List[str]] = None,
    kb_sources: Optional[List[str]] = None,
    model_score: float = 1.0,
    link_score: Optional[float] = None,
    rerank_score: Optional[float] = None,
    valid: bool = False,
    conf_weights: Tuple[float, float, float] = (0.5, 0.35, 0.15),
) -> Dict[str, Any]:
    w_m, w_l, w_r = conf_weights
    m = float(model_score or 0.0)
    l = float(link_score or 0.0)
    r = float(rerank_score or 0.0)
    confidence = max(0.0, min(1.0, w_m * m + w_l * l + w_r * r))
    return {
        "text": text,
        "start": start,
        "end": end,
        "label": label,
        "cui": cui,
        "canonical": (canonical or text),
        "semantic_types": list(semantic_types or []),
        "kb_sources": list(kb_sources or []),
        "valid": bool(valid),
        "scores": {"model": m, "link": link_score, "rerank": rerank_score, "confidence": confidence},
    }

def _passes_link_filters(linker, cui: Optional[str],
                         allowed_sem_types: Optional[List[str]],
                         allowed_sources: Optional[List[str]]) -> bool:
    if not linker or not cui:
        return True
    kb_ent = linker.kb.cui_to_entity.get(cui)
    if kb_ent is None:
        return False
    if allowed_sem_types and not (set(kb_ent.types or []) & set(allowed_sem_types)):
        return False
    if allowed_sources and not (set(kb_ent.sources or []) & set(allowed_sources)):
        return False
    return True

# ================================================================================
# Linker enrichment (always prefer SciBERT's linker)
# ================================================================================
def _link_enrich(
    rec: Dict[str, Any],
    models: List[Tuple[str, Any, Optional[Any], Optional[str]]],  # (model_name, nlp, linker, linker_pipe)
    topk: int,
    min_link_score: float,
    context_text: str,
) -> None:
    """
    Attach UMLS metadata (CUI, canonical, semantic_types, kb_sources) to `rec` by
    running the scispaCy linker on a constructed Doc that contains exactly the
    entity span. Prefer the linker hosted by en_core_sci_scibert.
    """
    try:
        # 1) Choose a linker: prefer SciBERT
        chosen_linker = None
        host_nlp = None
        for mname, nlp, linker, _pipe in models:
            if linker is not None and mname and "sci" in mname.lower():
                chosen_linker, host_nlp = linker, nlp
                break
        if chosen_linker is None:
            for _mname, nlp, linker, _pipe in models:
                if linker is not None:
                    chosen_linker, host_nlp = linker, nlp
                    break
        if not chosen_linker or not host_nlp:
            return  # no linker available

        # Helper to run linker on a given span string
        def _link_on_span(span_text: str) -> List[Tuple[str, float]]:
            if not span_text.strip():
                return []
            # Build a doc with the host tokenizer so char offsets align
            doc = host_nlp.make_doc(span_text)
            span = doc.char_span(0, len(span_text), label=rec.get("label") or "ENTITY", alignment_mode="expand")
            if span is None:
                return []
            doc.set_ents([span])
            # Call the linker component directly (bypass pipeline selection)
            chosen_linker(doc)
            ent = doc.ents[0] if doc.ents else None
            return list(getattr(ent._, "kb_ents", []) or []) if ent is not None else []

        # Helper to run linker on the original context using provided offsets
        def _link_on_offsets(full_text: str, start: int, end: int) -> List[Tuple[str, float]]:
            if not full_text or end <= start:
                return []
            doc = host_nlp.make_doc(full_text)
            span = doc.char_span(int(start), int(end), label=rec.get("label") or "ENTITY", alignment_mode="expand")
            if span is None:
                return []
            doc.set_ents([span])
            chosen_linker(doc)
            ent = doc.ents[0] if doc.ents else None
            return list(getattr(ent._, "kb_ents", []) or []) if ent is not None else []

        # 2) Try offsets first (best alignment with original text)
        kb_ents: List[Tuple[str, float]] = []
        text = context_text or ""
        start, end = rec.get("start"), rec.get("end")
        if isinstance(start, int) and isinstance(end, int) and text:
            kb_ents = _link_on_offsets(text, start, end)

        # 3) Fallback: mention-only
        if not kb_ents:
            kb_ents = _link_on_span(rec.get("text") or "")

        if not kb_ents:
            return

        # 4) Score filter + take top-1
        candidates = [(cui, float(score)) for cui, score in kb_ents[: int(topk)] if float(score) >= float(min_link_score)]
        if not candidates:
            return

        cui, score = candidates[0]
        kb_ent = chosen_linker.kb.cui_to_entity.get(cui)

        rec["cui"] = cui
        # Keep existing canonical if present, otherwise kb canonical name
        rec["canonical"] = (kb_ent.canonical_name if (kb_ent and getattr(kb_ent, "canonical_name", None)) else rec.get("canonical") or rec.get("text") or "")
        rec["semantic_types"] = list(getattr(kb_ent, "types", []) or []) if kb_ent is not None else []
        rec["kb_sources"] = list(getattr(kb_ent, "sources", []) or []) if kb_ent is not None else []
        rec["scores"]["link"] = float(score)
        rec["valid"] = True

    except Exception as e:
        # Quiet in normal mode; flip to debug if you want noise
        log.debug("Linker enrich failed: %s", e)


# ======================================================================================
# The Ensemble (HF-free)
# ======================================================================================

class SpacyEnsembleNERPipeline:
    """
    spaCy/scispaCy ensemble with optional UMLS linking and optional CrossEncoder re-ranking.

    Call with:
      - str  -> List[entity]
      - list -> List[List[entity]]

    Diagnostics (if collect_stats=True):
      .stats = {
        "models": {"<name>": {"ok": int, "fail": int}},
        "timing": {"load_ms": float, "run_ms": float, "rerank_ms": float},
        "linker_yield": {"total_ents": int, "linked": int},
        "reranker": {"enabled": bool, "scored": int}
      }
    """

    version: str = "3.2.0-spacy-ensemble+reranker"

    def __init__(self, config: EnsembleConfig):
        if not isinstance(config, EnsembleConfig):
            raise TypeError("SpacyEnsembleNERPipeline requires an EnsembleConfig.")
        self.config = config
        self._num_workers = 1  # harmless attribute; external checks may reference it

        self.stats: Dict[str, Any] = {
            "models": {},
            "timing": {"load_ms": 0.0, "run_ms": 0.0, "rerank_ms": 0.0},
            "linker_yield": {"total_ents": 0, "linked": 0},
            "reranker": {"enabled": bool(config.enable_reranker), "scored": 0},
        }

        # Load models (cached) ------------------------------------------------
        t0 = time.time()
        # (name, nlp, linker, linker_pipe)
        self._models: List[Tuple[str, Any, Optional[Any], Optional[str]]] = []
        for name in self.config.spacy_model_names:
            nlp, linker, linker_pipe = _load_spacy_model(name)
            if nlp is None:
                log.warning("Skipping unavailable spaCy model: %s", name)
                continue
            use_linker = linker if self.config.enable_linker else None
            use_pipe = linker_pipe if self.config.enable_linker else None
            self._models.append((name, nlp, use_linker, use_pipe))
            self.stats["models"][name] = {"ok": 0, "fail": 0}
        if not self._models:
            raise RuntimeError("No spaCy models could be loaded for the ensemble.")
        self.stats["timing"]["load_ms"] = (time.time() - t0) * 1000.0

        # Optional reranker ---------------------------------------------------
        self._reranker = _CrossEncoderReranker(self.config.reranker_model) if self.config.enable_reranker else None

        if self.config.random_seed is not None:
            try:
                import random
                random.seed(self.config.random_seed)
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def __call__(self, inputs: Union[str, List[str]]) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        if isinstance(inputs, str):
            return self._process_text(inputs)
        if isinstance(inputs, list):
            return self._process_batch(inputs)
        raise TypeError("SpacyEnsembleNERPipeline expects a string or a list[str].")

    def extract(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        return self._process_batch(texts)

    # ---------------------------------------------------------------------
    # Core
    # ---------------------------------------------------------------------

    def _process_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        texts = list(texts or [])
        if not texts:
            return []
        t0 = time.time()
        out: List[List[Dict[str, Any]]] = [[] for _ in texts]

        max_workers = max(1, int(self.config.max_workers_texts))
        if max_workers == 1 or len(texts) == 1:
            for i, t in enumerate(texts):
                out[i] = self._process_text(t)
        else:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="spacy-ens-text") as ex:
                futures = {ex.submit(self._process_text, t): i for i, t in enumerate(texts)}
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        out[i] = fut.result()
                    except Exception as e:
                        log.warning("Text #%d extraction failed: %s", i, e)
                        out[i] = []

        self.stats["timing"]["run_ms"] = (time.time() - t0) * 1000.0
        return out

    def _process_text(self, text: str) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []

        # Collect per-model entities (parallel if >1 model)
        results: List[List[Dict[str, Any]]] = [[] for _ in self._models]

        def _run_one(idx: int, name: str, nlp, linker, linker_pipe):
            try:
                ents = self._extract_from_model(nlp, linker, linker_pipe, text)
                results[idx] = ents
                self.stats["models"][name]["ok"] += 1
            except Exception as e:
                log.warning("Model '%s' failed: %s", name, e)
                self.stats["models"][name]["fail"] += 1
                results[idx] = []

        if len(self._models) == 1 or self.config.max_workers_models <= 1:
            for i, (name, nlp, linker, linker_pipe) in enumerate(self._models):
                _run_one(i, name, nlp, linker, linker_pipe)
        else:
            workers = max(1, int(self.config.max_workers_models))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="spacy-ens-model") as ex:
                futs = [ex.submit(_run_one, i, m[0], m[1], m[2], m[3]) for i, m in enumerate(self._models)]
                for _ in as_completed(futs):
                    pass

        # Optional reranker — score each candidate against context
        if self._reranker is not None:
            t_r0 = time.time()
            self._apply_reranker(results, text)
            self.stats["timing"]["rerank_ms"] += (time.time() - t_r0) * 1000.0

        # Merge across models
        merged = self._merge(results, strategy=self.config.merge_strategy)

        # Final gating
        final: List[Dict[str, Any]] = []
        stop = set(self.config.stop_terms or [])
        for e in merged:
            if self.config.drop_if_no_cui_when_linking and self.config.enable_linker and not e.get("cui"):
                continue
            if float(e["scores"]["confidence"]) < float(self.config.min_confidence):
                continue
            if stop and _normalize_surface(e["text"]) in stop:
                continue
            final.append(e)

        return final

    # ---------------------------------------------------------------------
    # Model extraction
    # ---------------------------------------------------------------------

    def _extract_from_model(self, nlp, linker, linker_pipe: Optional[str], text: str) -> List[Dict[str, Any]]:
        doc = nlp(text)
        allowed = set(self.config.allowed_labels or [])
        ents: List[Dict[str, Any]] = []

        # Named entities
        for sp in getattr(doc, "ents", []):
            if self.config.min_span_len and (sp.end_char - sp.start_char) < int(self.config.min_span_len):
                continue
            if allowed and sp.label_ not in allowed:
                continue

            rec = _mk_entity(
                text=sp.text,
                start=sp.start_char,
                end=sp.end_char,
                label=sp.label_,
                model_score=1.0,  # spaCy doesn't expose calibrated probability; treat as 1.0 baseline
                conf_weights=(self.config.w_model_score, self.config.w_link_score, self.config.w_rerank_score),
            )

            if self.config.enable_linker:
                _link_enrich(rec, self._models, self.config.top_k_linker, self.config.min_link_score, text)
                if linker is not None and not _passes_link_filters(linker, rec.get("cui"),
                                                                   self.config.allowed_semantic_types,
                                                                   self.config.allowed_kb_sources):
                    # Remove link if it fails filters
                    rec["cui"] = None
                    rec["semantic_types"] = []
                    rec["kb_sources"] = []
                    rec["scores"]["link"] = None
                    rec["valid"] = False

            # update confidence after linking
            m = float(rec["scores"]["model"] or 0.0)
            l = float(rec["scores"]["link"] or 0.0) if rec["scores"]["link"] is not None else 0.0
            rec["scores"]["confidence"] = max(0.0, min(1.0, self.config.w_model_score * m + self.config.w_link_score * l))
            ents.append(rec)

            # stats: linker yield
            if self.config.enable_linker:
                self.stats["linker_yield"]["total_ents"] += 1
                if rec.get("cui"):
                    self.stats["linker_yield"]["linked"] += 1

        # Optional salvage (noun chunks) if no NER hits and linker on
        if not ents and self.config.enable_linker:
            try:
                for ch in getattr(doc, "noun_chunks", []):
                    if self.config.min_span_len and (ch.end_char - ch.start_char) < int(self.config.min_span_len):
                        continue
                    rec = _mk_entity(
                        text=ch.text,
                        start=ch.start_char,
                        end=ch.end_char,
                        label="NP",
                        model_score=0.6,  # lower baseline for chunks
                        conf_weights=(self.config.w_model_score, self.config.w_link_score, self.config.w_rerank_score),
                    )
                    _link_enrich(rec, self._models, self.config.top_k_linker, self.config.min_link_score, text)
                    if linker is not None and not _passes_link_filters(linker, rec.get("cui"),
                                                                       self.config.allowed_semantic_types,
                                                                       self.config.allowed_kb_sources):
                        rec["cui"] = None
                        rec["semantic_types"] = []
                        rec["kb_sources"] = []
                        rec["scores"]["link"] = None
                        rec["valid"] = False

                    # recompute confidence
                    m = float(rec["scores"]["model"] or 0.0)
                    l = float(rec["scores"]["link"] or 0.0) if rec["scores"]["link"] is not None else 0.0
                    rec["scores"]["confidence"] = max(0.0, min(1.0, self.config.w_model_score * m + self.config.w_link_score * l))
                    ents.append(rec)
            except Exception:
                pass

        return ents

    # ---------------------------------------------------------------------
    # Reranker application
    # ---------------------------------------------------------------------

    def _apply_reranker(self, per_model: List[List[Dict[str, Any]]], context: str) -> None:
        flat = [e for lst in per_model for e in lst]
        if not flat or self._reranker is None:
            return

        pairs: List[Tuple[str, str]] = []
        for e in flat:
            cand = e.get("canonical") or e.get("text") or ""
            stypes = e.get("semantic_types") or []
            meta = f" [{' '.join(stypes)}]" if stypes else ""
            pairs.append((context, f"{cand}{meta}"))

        scores = self._reranker.score(pairs)
        self.stats["reranker"]["scored"] += len(scores)

        # write back rerank + recompute confidence with weights
        for e, s in zip(flat, scores):
            e["scores"]["rerank"] = float(s)
            m = float(e["scores"].get("model") or 0.0)
            l = float(e["scores"].get("link") or 0.0) if e["scores"].get("link") is not None else 0.0
            r = float(s)
            e["scores"]["confidence"] = max(
                0.0, min(1.0, self.config.w_model_score * m + self.config.w_link_score * l + self.config.w_rerank_score * r)
            )

    # ---------------------------------------------------------------------
    # Merge
    # ---------------------------------------------------------------------

    def _merge(self, per_model: List[List[Dict[str, Any]]], strategy: str = "surface") -> List[Dict[str, Any]]:
        flat: List[Dict[str, Any]] = [e for lst in per_model for e in lst]
        if not flat:
            return []

        if strategy not in {"surface", "span", "fuzzy"}:
            strategy = "surface"

        buckets: List[List[Dict[str, Any]]] = []

        if strategy == "surface":
            by_key: Dict[str, List[Dict[str, Any]]] = {}
            for e in flat:
                key = _normalize_surface(e["text"])
                by_key.setdefault(key, []).append(e)
            buckets = list(by_key.values())

        elif strategy == "span":
            by_span: Dict[Tuple[Optional[int], Optional[int]], List[Dict[str, Any]]] = {}
            for e in flat:
                key = (e.get("start"), e.get("end"))
                by_span.setdefault(key, []).append(e)
            buckets = list(by_span.values())

        else:  # fuzzy token-jaccard
            remaining = flat[:]
            used = [False] * len(remaining)
            for i, e in enumerate(remaining):
                if used[i]:
                    continue
                cluster = [e]
                used[i] = True
                for j in range(i + 1, len(remaining)):
                    if used[j]:
                        continue
                    sim = _token_jaccard(e["text"], remaining[j]["text"])
                    if sim >= float(self.config.fuzzy_jaccard_threshold):
                        cluster.append(remaining[j])
                        used[j] = True
                buckets.append(cluster)

        merged: List[Dict[str, Any]] = []
        for group in buckets:
            merged.append(self._reduce_group(group))
        return merged

    def _reduce_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not group:
            return _mk_entity("", valid=False)

        # Prefer item with CUI; if tie, max link score; else longest surface form
        with_cui = [e for e in group if e.get("cui")]
        if with_cui:
            best = max(with_cui, key=lambda x: (float(x["scores"].get("link") or 0.0), len(x["text"])))
        else:
            best = max(group, key=lambda x: len(x["text"]))

        # Aggregate types/sources + scores
        all_types = set()
        all_srcs = set()
        model_scores = []
        link_scores = []
        rerank_scores = []

        for e in group:
            for t in (e.get("semantic_types") or []):
                all_types.add(t)
            for s in (e.get("kb_sources") or []):
                all_srcs.add(s)
            model_scores.append(float(e["scores"].get("model") or 0.0))
            if e["scores"].get("link") is not None:
                link_scores.append(float(e["scores"]["link"]))
            if e["scores"].get("rerank") is not None:
                rerank_scores.append(float(e["scores"]["rerank"]))

        agg_model = sum(model_scores) / max(1, len(model_scores))
        agg_link = max(link_scores) if link_scores else None
        agg_rerank = sum(rerank_scores) / max(1, len(rerank_scores)) if rerank_scores else None

        # recompute final confidence with weights
        m = float(agg_model or 0.0)
        l = float(agg_link or 0.0) if agg_link is not None else 0.0
        r = float(agg_rerank or 0.0) if agg_rerank is not None else 0.0
        conf = max(0.0, min(1.0, self.config.w_model_score * m + self.config.w_link_score * l + self.config.w_rerank_score * r))

        return {
            "text": best["text"],
            "start": best.get("start"),
            "end": best.get("end"),
            "label": best.get("label"),
            "cui": best.get("cui"),
            "canonical": best.get("canonical") or best["text"],
            "semantic_types": sorted(all_types),
            "kb_sources": sorted(all_srcs),
            "valid": bool(best.get("cui")) if self.config.enable_linker else True,
            "scores": {"model": m, "link": agg_link, "rerank": agg_rerank, "confidence": conf},
        }


# ======================================================================================
# Self-test (run this file directly)
# ======================================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = EnsembleConfig(
        spacy_model_names=[
            "en_ner_bc5cdr_md",
            "en_ner_bionlp13cg_md",
            "en_ner_craft_md",
            "en_ner_jnlpba_md",
            "en_core_sci_scibert",  # hosts the UMLS linker
        ],
        enable_linker=True,
        top_k_linker=5,
        min_link_score=0.0,
        merge_strategy="surface",
        enable_reranker=True,  # try reranker if sentence-transformers is installed
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_confidence=0.0,
        max_workers_models=2,
        max_workers_texts=2,
        enable_debug=False,
    )
    pipe = SpacyEnsembleNERPipeline(cfg)
    texts = [
        "Aspirin reduces platelet aggregation and is used for myocardial infarction prevention.",
        "EGFR mutations are associated with response to erlotinib in non-small cell lung cancer."
    ]
    outs = pipe(texts)
    for i, ents in enumerate(outs):
        print(f"\n=== TEXT {i} ===")
        for e in ents:
            print(e)
    print("\nStats:", pipe.stats)
