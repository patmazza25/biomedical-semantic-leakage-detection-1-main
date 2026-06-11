#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for:
Ontology-Guided Semantic Leakage Detection in Biomedical Chain-of-Thought Reasoning

What you get
------------
• CLI for single/batch questions (JSONL/JSON/TXT)
• Optional FastAPI server with GET/POST endpoints
• Parallel execution, JSONL logging, per-question artifacts
• Robust loaders for .jsonl/.json/.txt prompt files
• Pluggable CoT generation + ontology concepts + entailment
• “Actual work” safeguards: smoke tests + strict end-of-run gates
• Clean handling of evolving utils/.* function signatures
• Artifacts respect --out_dir and create folders automatically

Quick CLI examples
------------------
# Single question (prints JSON to stdout)
python main.py --question "Does aspirin reduce MI risk?" --json_only --prefer anthropic

# Batch from JSONL, 8 workers, append to runs/batch.jsonl (strict gates on)
REQUIRE_ENTAILMENT=1 REQUIRE_UMLS=1 \
python main.py --batch \
  --questions_file combined/p8001-10000.jsonl \
  --workers 9 \
  --json_only \
  --log_jsonl runs/batch.jsonl \
  --prefer anthropic \
  --scispacy_when never \
  --strict \
  --min_entailment_coverage 0.80 \
  --min_concept_valid_rate 0.70 \
  --metrics_json runs/metrics.json

# Start the API server (requires: pip install fastapi uvicorn pydantic)
python main.py --serve --host 0.0.0.0 --port 5005
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import inspect
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Safe imports (we don’t trust exact symbols because your utils evolve)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_import_cot_generator():
    try:
        from utils import cot_generator as cg  # project layout
        return cg
    except Exception:
        try:
            import cot_generator as cg  # flat layout
            return cg
        except Exception as e:
            logging.warning("[cot] Could not import cot_generator: %s", e)
            return None

def _safe_import_concept_extractor():
    try:
        from utils.concept_extractor import extract_concepts  # type: ignore
        return extract_concepts
    except Exception as e:
        logging.warning("[concepts] extractor unavailable: %s", e)
        return None

def _safe_import_hybrid_builder():
    try:
        from utils.hybrid_checker import build_entailment_records  # type: ignore
        return build_entailment_records
    except Exception as e:
        logging.info("[entailment] hybrid_checker unavailable: %s", e)
        return None

def _safe_import_entailment_fns():
    ce = ceb = afl = None
    try:
        from utils.entailment_checker import check_entailment as ce  # type: ignore
    except Exception as e:
        logging.info("[entailment] check_entailment unavailable: %s", e)
    try:
        from utils.entailment_checker import check_entailment_bidirectional as ceb  # type: ignore
    except Exception as e:
        logging.info("[entailment] check_entailment_bidirectional unavailable: %s", e)
    try:
        from utils.entailment_checker import attach_final_labels as afl  # type: ignore
    except Exception as e:
        logging.info("[entailment] attach_final_labels unavailable: %s", e)
    return ce, ceb, afl

def _safe_import_umls_linker_configured():
    try:
        from utils.umls_api_linker import is_configured as umls_is_configured  # type: ignore
    except Exception:
        umls_is_configured = lambda: False  # noqa: E731
    return umls_is_configured

def _safe_import_umls_link_texts_batch():
    try:
        from utils.umls_api_linker import link_texts_batch  # type: ignore
        return link_texts_batch
    except Exception as e:
        logging.info("[umls] link_texts_batch unavailable: %s", e)
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt file loading (.jsonl / .json / .txt)
# ─────────────────────────────────────────────────────────────────────────────

_Q_RE = re.compile(r"^\s*question\s*:\s*(.+)$", re.IGNORECASE)
_I_RE = re.compile(r"^\s*instructions?\s*:\s*(.+)$", re.IGNORECASE)
QUESTION_KEYS = ("question", "query", "q", "prompt", "text", "title")

def _pick_question_from_obj(obj: Dict[str, Any]) -> str:
    for k in QUESTION_KEYS:
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            return obj[k].strip()
    for v in obj.values():
        if isinstance(v, str):
            m = re.search(r"Question\s*:\s*(.+)", v, flags=re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1).strip()
    return ""

def _load_jsonl(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                if line.lower().startswith("question:"):
                    out.append(line.split(":", 1)[1].strip())
                elif line.lower().startswith("instructions:"):
                    pass
                else:
                    out.append(line)
                continue
            if isinstance(obj, dict):
                q = _pick_question_from_obj(obj)
                if q:
                    out.append(q)
            elif isinstance(obj, str) and obj.strip():
                s = obj.strip()
                if s.lower().startswith("question:"):
                    out.append(s.split(":", 1)[1].strip())
                elif s.lower().startswith("instructions:"):
                    pass
                else:
                    out.append(s)
    seen, deduped = set(), []
    for q in out:
        if q and q not in seen:
            seen.add(q); deduped.append(q)
    return deduped

def _load_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [str(x).strip() for x in obj if str(x).strip()]
    if isinstance(obj, dict):
        if "questions" in obj and isinstance(obj["questions"], list):
            return [str(x).strip() for x in obj["questions"] if str(x).strip()]
        vals = []
        for v in obj.values():
            if isinstance(v, dict):
                q = _pick_question_from_obj(v)
                if q:
                    vals.append(q)
        if vals:
            return vals
    return []

def _load_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    lines = [ln.rstrip() for ln in raw.splitlines()]
    questions: List[str] = []
    cur_q: List[str] = []

    def flush():
        nonlocal cur_q
        if cur_q:
            q = " ".join(cur_q).strip()
            if q:
                questions.append(q)
        cur_q = []

    for ln in lines:
        if not ln.strip():
            continue
        if _I_RE.match(ln):
            continue
        m_q = _Q_RE.match(ln)
        if m_q:
            flush()
            cur_q = [m_q.group(1).strip()]
        else:
            if cur_q:
                cur_q.append(ln.strip())
            else:
                questions.append(ln.strip())
    flush()
    seen, out = set(), []
    for q in questions:
        if q and q not in seen:
            seen.add(q); out.append(q)
    return out

def load_questions_file(path: str) -> List[str]:
    path = os.path.expanduser(os.path.expandvars(path))
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".jsonl":
            return _load_jsonl(path)
        if ext == ".json":
            out = _load_json(path)
            return out if out else _load_jsonl(path)
        return _load_txt(path)
    except FileNotFoundError:
        raise
    except Exception as e:
        for loader in (_load_jsonl, _load_json, _load_txt):
            try:
                return loader(path)
            except Exception:
                continue
        raise RuntimeError(f"Failed to load questions from {path}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# CoT generation
# ─────────────────────────────────────────────────────────────────────────────

def _call_cot_generator(question: str, prefer: str = "anthropic") -> Dict[str, Any]:
    """
    Normalize result to:
      {"provider": str, "model": str, "steps": [..], "final": str, "raw": Any, "errors": []}
    """
    cg = _safe_import_cot_generator()
    out: Dict[str, Any] = {"provider": "local", "model": "fallback", "steps": [], "final": "", "raw": None, "errors": []}
    if cg is None:
        out["errors"].append("cot_generator module not found")
        return out

    candidates = []
    for name in ("generate", "generate_cot", "run", "demo", "produce"):
        if hasattr(cg, name):
            candidates.append(getattr(cg, name))
    if not candidates and hasattr(cg, "CoTGenerator"):
        try:
            inst = cg.CoTGenerator(prefer=prefer)
            if hasattr(inst, "generate"):
                candidates.append(inst.generate)
        except Exception as e:
            out["errors"].append(f"init CoTGenerator failed: {e}")
    if not candidates:
        out["errors"].append("No callable entry point found in cot_generator")
        return out

    last_exc = None
    res = None
    for fn in candidates:
        try:
            res = fn(question=question, prefer=prefer); break
        except TypeError:
            try:
                res = fn(question, prefer=prefer); break
            except Exception as e:
                last_exc = e
        except Exception as e:
            last_exc = e

    if res is None:
        if last_exc is not None:
            out["errors"].append(f"generator call failed: {last_exc}")
        return out

    try:
        if isinstance(res, dict):
            steps = []
            if isinstance(res.get("steps"), list):
                steps = [str(s).strip() for s in res["steps"] if str(s).strip()]
            elif isinstance(res.get("text"), str):
                steps = [ln.strip() for ln in res["text"].splitlines() if ln.strip()]
            out["steps"]  = steps or out["steps"]
            out["final"]  = str(res.get("final") or res.get("answer") or out["final"]).strip()
            out["provider"] = str(res.get("provider") or out["provider"])
            out["model"]  = str(res.get("model") or out["model"])
            out["raw"]    = res.get("raw")
        elif isinstance(res, (list, tuple)):
            out["steps"] = [str(s).strip() for s in res if str(s).strip()]
        elif isinstance(res, str):
            out["steps"] = [ln.strip() for ln in res.splitlines() if ln.strip()]
    except Exception as e:
        out["errors"].append(f"normalize result failed: {e}")

    if not out["steps"]:
        out["steps"] = [
            "Identify biomedical entities and relations in the question.",
            "Recall evidence or mechanisms.",
            "Synthesize a cautious conclusion with caveats."
        ]
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Concept extraction
# ─────────────────────────────────────────────────────────────────────────────

def _call_concept_extractor(
    steps: List[str],
    scispacy_when: str,
    top_k_umls: int
) -> List[List[Dict[str, Any]]]:
    extractor = _safe_import_concept_extractor()
    if extractor is None:
        return [[] for _ in steps]

    try:
        return extractor(steps, scispacy_when=scispacy_when, top_k=top_k_umls)  # type: ignore
    except TypeError:
        pass
    try:
        return extractor(steps, scispacy_when=scispacy_when, top_k_umls=top_k_umls)  # type: ignore
    except TypeError:
        pass
    try:
        return extractor(steps)  # type: ignore
    except Exception as e:
        logging.warning("[concepts] extraction failed: %s", e)
        return [[] for _ in steps]

# ─────────────────────────────────────────────────────────────────────────────
# Entailment (hybrid with UMLS if available; else plain; else unknown)
# ─────────────────────────────────────────────────────────────────────────────

def _entailment_with_umls(steps: List[str], umls_per_step: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if len(steps) < 2:
        return []

    hybrid_build = _safe_import_hybrid_builder()
    if hybrid_build:
        try:
            recs = hybrid_build(steps, umls_per_step) or []
            out = []
            for r in recs:
                i, j = 0, 1
                if isinstance(r.get("step_pair"), (list, tuple)) and len(r["step_pair"]) == 2:
                    i, j = int(r["step_pair"][0]), int(r["step_pair"][1])
                else:
                    i = int(r.get("i", i)); j = int(r.get("j", j))
                probs = r.get("probs") or {}
                label = str(r.get("final_label") or r.get("label") or (max(probs, key=probs.get) if probs else "neutral"))
                out.append({
                    "i": i, "j": j, "label": label,
                    "scores": {
                        "E": float(probs.get("entailment", 0.0)),
                        "N": float(probs.get("neutral", 0.0)),
                        "C": float(probs.get("contradiction", 0.0)),
                    },
                    "meta": r.get("meta", {}),
                })
            return out
        except Exception as e:
            logging.warning("[entailment] hybrid failed; falling back: %s", e)

    check_entailment, check_entailment_bidirectional, _attach = _safe_import_entailment_fns()
    if check_entailment:
        try:
            base = check_entailment(steps) or []
            out = []
            for k, r in enumerate(base):
                probs = r.get("probs") or {}
                label = str(r.get("label") or (max(probs, key=probs.get) if probs else "neutral"))
                out.append({
                    "i": k, "j": k + 1, "label": label,
                    "scores": {
                        "E": float(probs.get("entailment", 0.0)),
                        "N": float(probs.get("neutral", 0.0)),
                        "C": float(probs.get("contradiction", 0.0)),
                    }
                })
            return out
        except Exception as e:
            logging.warning("[entailment] plain NLI failed: %s", e)

    return [{"i": i, "j": i + 1, "label": "unknown", "scores": {"E": 0.0, "N": 1.0, "C": 0.0}}
            for i in range(len(steps) - 1)]

# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────────────────

_CUI_RE = re.compile(r"^C\d{3,}$", re.IGNORECASE)

def _cand_is_valid(c: Dict[str, Any]) -> bool:
    if not isinstance(c, dict):
        return False
    if c.get("valid") is True:
        return True
    cui = str(c.get("cui") or c.get("concept_id") or "").strip()
    if _CUI_RE.match(cui):
        stys = c.get("semantic_types") or c.get("sty") or c.get("tui")
        if stys is None:
            return True
        if isinstance(stys, (list, tuple)) and len(stys) > 0:
            return True
        if isinstance(stys, str) and stys.strip():
            return True
    return False

def smoke_test_entailment() -> Tuple[bool, str]:
    steps = ["A causes B.", "B is caused by A.", "A prevents B.", "C is a drug."]
    pairs = _entailment_with_umls(steps, umls_per_step=[[], [], [], []])
    if not pairs:
        return False, "Entailment returned no pairs"
    labels = [p["label"].lower() for p in pairs]
    if all(l == "unknown" for l in labels):
        return False, "All entailment labels are 'unknown'"
    if len(set(labels)) >= 2:
        return True, f"OK (labels={sorted(set(labels))})"
    any_signal = any((p["scores"].get("E", 0) > 0.01 or p["scores"].get("C", 0) > 0.01) for p in pairs)
    if any_signal:
        return True, f"OK (score signal present; labels={labels})"
    return False, f"Weak entailment signal (labels={labels})"

def smoke_test_umls() -> Tuple[bool, str]:
    link_batch = _safe_import_umls_link_texts_batch()
    surfaces = ["aspirin", "myocardial infarction", "metformin"]

    if link_batch:
        try:
            res = link_batch(surfaces, top_k=3)  # should be a dict
            if not isinstance(res, dict):
                return False, f"link_texts_batch returned {type(res).__name__}, expected dict"
            if not all(isinstance(k, str) for k in res.keys()):
                return False, "link_texts_batch keys must be strings"
            any_valid = any(
                any(_cand_is_valid(c) for c in (cand_list or []))
                for cand_list in res.values()
            )
            return (True, "OK") if any_valid else (False, "No valid UMLS candidates for probe surfaces")
        except Exception as e:
            return False, f"link_texts_batch raised: {e}"

    extractor = _safe_import_concept_extractor()
    if not extractor:
        return False, "concept_extractor unavailable"
    try:
        linked = _call_concept_extractor(surfaces, scispacy_when="always", top_k_umls=3)
        if not isinstance(linked, list) or len(linked) != len(surfaces):
            return False, "concept_extractor returned unexpected shape"
        any_valid = any(any(_cand_is_valid(c) for c in (arr or [])) for arr in linked)
        return (True, "OK") if any_valid else (False, "No valid UMLS candidates via extractor")
    except Exception as e:
        return False, f"concept_extractor raised: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# Metrics & end-of-run gating
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    total_pairs = 0
    covered_pairs = 0
    total_steps = 0
    steps_with_valid = 0

    for rec in records:
        pairs = rec.get("entailment_pairs") or []
        total_pairs += len(pairs)
        for p in pairs:
            lab = str(p.get("label") or "").lower()
            if lab in {"entails", "entailment", "contradiction", "neutral"}:
                covered_pairs += 1
            elif lab not in {"", "unknown"}:
                covered_pairs += 1

        steps = rec.get("steps") or []
        total_steps += len(steps)
        per_step = (((rec.get("concepts") or {}).get("per_step")) or [])
        for arr in per_step[:len(steps)]:
            if any(_cand_is_valid(c) for c in (arr or [])):
                steps_with_valid += 1

    entailment_coverage = (covered_pairs / total_pairs) if total_pairs else 0.0
    concept_valid_rate = (steps_with_valid / total_steps) if total_steps else 0.0
    return {
        "entailment_coverage": round(entailment_coverage, 4),
        "concept_valid_rate": round(concept_valid_rate, 4),
        "total_pairs": total_pairs,
        "total_steps": total_steps,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Per-question processing
# ─────────────────────────────────────────────────────────────────────────────

def process_question(
    question: str,
    prefer: str = "anthropic",
    scispacy_when: str = "auto",
    top_k_umls: int = 3,
    out_dir: Path = Path("data"),
    write_artifacts: bool = True,
) -> Tuple[bool, Dict[str, Any], Optional[Path]]:
    t0 = time.time()
    record: Dict[str, Any] = {
        "question": question,
        "started_utc": _utc_stamp(),
        "generator": {},
        "steps": [],
        "final": "",
        "entailment_pairs": [],
        "concepts": {},
        "umls_configured": _safe_import_umls_linker_configured()(),
        "umls_density": {},
        "umls_specificity": {},
        "errors": [],
        "duration_s": None,
    }

    # 1) CoT
    cg_res = _call_cot_generator(question, prefer=prefer)
    record["generator"] = {
        "provider": cg_res.get("provider"),
        "model": cg_res.get("model"),
        "errors": cg_res.get("errors") or [],
    }
    if cg_res.get("errors"):
        record["errors"].extend([f"cot: {e}" for e in cg_res["errors"]])

    steps: List[str] = list(cg_res.get("steps") or [])
    record["steps"] = steps
    record["final"] = cg_res.get("final") or ""

    # 2) Concepts
    try:
        per_step = _call_concept_extractor(steps, scispacy_when=scispacy_when, top_k_umls=top_k_umls)
        record["concepts"] = {"per_step": per_step, "meta": {"impl": "concept_extractor"}}
    except Exception as e:
        logging.warning("[concepts] failed: %s", e)
        record["errors"].append(f"concepts: {e}")
        record["concepts"] = {"per_step": [[] for _ in steps], "meta": {"error": str(e)}}

    # 3) Entailment
    umls_per_step = list(record["concepts"].get("per_step") or [[] for _ in steps])
    try:
        record["entailment_pairs"] = _entailment_with_umls(steps, umls_per_step)
    except Exception as e:
        logging.warning("[entailment] failed: %s", e)
        record["errors"].append(f"entailment: {e}")

    # 3c) UMLS standalone signals
    try:
        from utils.umls_density_scorer import score_density
        record["umls_density"] = score_density(umls_per_step, steps=steps)
    except Exception as e:
        logging.warning("[umls_density] failed: %s", e)
        record["umls_density"] = {"configured": False, "per_step": [], "slope": None,
                                   "leakage_onset_step": None, "overall_risk": 0.0}

    try:
        from utils.umls_specificity_scorer import score_specificity
        record["umls_specificity"] = score_specificity(umls_per_step)
    except Exception as e:
        logging.warning("[umls_specificity] failed: %s", e)
        record["umls_specificity"] = {"configured": False, "per_step": [], "depth_slope": None,
                                       "abstraction_leaps": [], "overall_specificity_score": 0.0}

    # 3b) Guard signals (using UMLS relation metadata from hybrid checker)
    try:
        from utils.guards import derive_guards
        for pair_rec in record.get("entailment_pairs", []):
            si = steps[pair_rec.get("i", 0)] if pair_rec.get("i", 0) < len(steps) else ""
            sj = steps[pair_rec.get("j", 1)] if pair_rec.get("j", 1) < len(steps) else ""
            scores = pair_rec.get("scores", {})
            probs_for_guard = {
                "entailment": scores.get("E", 0.0),
                "neutral": scores.get("N", 0.0),
                "contradiction": scores.get("C", 0.0),
            }
            meta = pair_rec.get("meta") or {}
            guards = derive_guards(
                premise=si,
                hypothesis=sj,
                probs=probs_for_guard,
                relation_violation=bool(meta.get("relation_violation", False)),
                ontology_override_signal=bool(meta.get("ontology_support", False)),
            )
            pair_rec["guards"] = guards
    except Exception as e:
        logging.debug("[guards] guard computation skipped: %s", e)

    record["duration_s"] = round(time.time() - t0, 3)

    # Artifact writing (respect --out_dir and ensure dirs)
    if not write_artifacts:
        return True, record, None

    try:
        # base out_dir
        _ensure_dir(out_dir)

        tm = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_provider = str(record["generator"].get("provider") or "local")
        safe_model = str(record["generator"].get("model") or "fallback").replace("/", "_")

        # artifacts under: <out_dir>/<provider>-<model>/data/
        subdir = out_dir / f"{safe_provider}-{safe_model}" / "data"
        _ensure_dir(subdir)

        unique = f"{os.getpid()}-{time.time_ns()%1_000_000}"
        path = subdir / f"report-{tm}-{unique}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return True, record, path
    except Exception as e:
        logging.warning("[io] could not write per question JSON: %s", e)
        return True, record, None

# ─────────────────────────────────────────────────────────────────────────────
# Batch runner (with strict gates, metrics JSON)
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    questions: List[str],
    workers: int,
    json_only: bool,
    log_jsonl: Optional[str],
    prefer: str,
    scispacy_when: str,
    top_k_umls: int,
    strict: bool,
    min_entail_cov: float,
    min_concept_rate: float,
    out_dir: Path,
    write_artifacts: bool,
    metrics_json: Optional[str],
) -> int:
    logging.info(
        "Starting run for %d questions with %d workers | prefer=%s | json_only=%s | scispacy_umls=%s | out_dir=%s",
        len(questions), workers, prefer, json_only, scispacy_when, str(out_dir)
    )

    jsonl_fp = None
    if log_jsonl:
        log_path = Path(os.path.expanduser(os.path.expandvars(log_jsonl)))
        _ensure_dir(log_path.parent)
        jsonl_fp = open(log_path, "a", encoding="utf-8")

    records: List[Dict[str, Any]] = []
    success = 0
    failures = 0

    def _task(q: str):
        return process_question(
            q, prefer=prefer, scispacy_when=scispacy_when, top_k_umls=top_k_umls,
            out_dir=out_dir, write_artifacts=write_artifacts
        )

    with cf.ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {ex.submit(_task, q): q for q in questions}
        done = 0
        for fut in cf.as_completed(futs):
            q = futs[fut]
            ok, rec, path = fut.result()
            records.append(rec)
            done += 1
            if ok: success += 1
            else: failures += 1
            if jsonl_fp:
                try:
                    jsonl_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    jsonl_fp.flush()
                except Exception as e:
                    logging.warning("[io] jsonl write failed: %s", e)
            if not json_only:
                title = q if len(q) < 100 else q[:97] + "..."
                print(f"[{'OK' if ok else 'FAIL'}] {done}/{len(questions)}  {title}")
                if path:
                    print(f"  saved: {path}")

    if jsonl_fp:
        jsonl_fp.close()

    metrics = _compute_metrics(records)
    logging.info("[METRICS] entailment_coverage=%.3f (pairs=%d) | concept_valid_rate=%.3f (steps=%d)",
                 metrics["entailment_coverage"], metrics["total_pairs"],
                 metrics["concept_valid_rate"], metrics["total_steps"])

    if metrics_json:
        try:
            mpath = Path(os.path.expanduser(os.path.expandvars(metrics_json)))
            _ensure_dir(mpath.parent)
            with open(mpath, "w", encoding="utf-8") as fp:
                json.dump(metrics, fp, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning("[io] metrics_json write failed: %s", e)

    if strict:
        failed_reasons = []
        if metrics["entailment_coverage"] < float(min_entail_cov):
            failed_reasons.append(f"entailment_coverage {metrics['entailment_coverage']:.3f} < {min_entail_cov:.3f}")
        if metrics["concept_valid_rate"] < float(min_concept_rate):
            failed_reasons.append(f"concept_valid_rate {metrics['concept_valid_rate']:.3f} < {min_concept_rate:.3f}")
        if failed_reasons:
            logging.error("[STRICT FAIL] " + " ; ".join(failed_reasons))
            return 5

    logging.info("[DONE] success=%d failures=%d", success, failures)
    return 0

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI server (optional)
# ─────────────────────────────────────────────────────────────────────────────

app = None
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(title="Bio Ontology CoT API", version="1.3.1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class BatchIn(BaseModel):
        questions: List[str]
        prefer: Optional[str] = "anthropic"
        scispacy_when: Optional[str] = "auto"
        top_k_umls: int = 3

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "utc": _utc_stamp()}

    @app.get("/cot")
    def api_cot(question: str, prefer: str = "anthropic") -> Dict[str, Any]:
        cg = _call_cot_generator(question, prefer=prefer)
        return {
            "question": question,
            "provider": cg.get("provider"),
            "model": cg.get("model"),
            "steps": cg.get("steps"),
            "final": cg.get("final"),
            "errors": cg.get("errors", []),
        }

    @app.get("/analyze")
    def api_analyze(question: str, prefer: str = "anthropic", scispacy_when: str = "auto", top_k_umls: int = 3) -> Dict[str, Any]:
        ok, rec, _ = process_question(
            question, prefer=prefer, scispacy_when=scispacy_when, top_k_umls=int(top_k_umls),
            write_artifacts=False
        )
        rec["ok"] = bool(ok)
        return rec

    @app.post("/batch")
    def api_batch(body: BatchIn) -> Dict[str, Any]:
        results = []
        for q in body.questions[:200]:  # soft limit
            ok, rec, _ = process_question(
                q, prefer=body.prefer or "anthropic", scispacy_when=body.scispacy_when or "auto",
                top_k_umls=int(body.top_k_umls), write_artifacts=False
            )
            rec["ok"] = bool(ok)
            results.append(rec)
        metrics = _compute_metrics(results)
        return {"count": len(results), "metrics": metrics, "results": results}

except Exception:
    app = None  # FastAPI not installed

def _run_server(host: str, port: int) -> int:
    if app is None:
        print("[ERROR] FastAPI/uvicorn not installed. pip install fastapi uvicorn pydantic")
        return 2
    try:
        import uvicorn  # type: ignore
    except Exception:
        print("[ERROR] uvicorn not installed. pip install uvicorn")
        return 2
    uvicorn.run(app, host=host, port=int(port), log_level="info")
    return 0

# ─────────────────────────────────────────────────────────────────────────────
# Arg parser & main
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Biomedical CoT leakage detector (CLI + API + strict gates)")

    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument("--serve", action="store_true", help="Run FastAPI server (requires fastapi+uvicorn)")
    mode.add_argument("--batch", action="store_true", help="Run in batch mode reading from file")
    mode.add_argument("--question", help="Single question string")

    # I/O & execution
    p.add_argument("--questions_file", help="Path to .jsonl or .json or .txt with questions")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    p.add_argument("--json_only", action="store_true", help="Only emit JSON to stdout (quiet human printing)")
    p.add_argument("--log_jsonl", help="Append each record to this JSONL file")
    p.add_argument("--out_dir", default="data", help="Base directory for per-question artifacts")
    p.add_argument("--no_artifacts", action="store_true", help="Do not write per-question JSON files")
    p.add_argument("--metrics_json", help="If set, write run-level metrics JSON here")
    p.add_argument("--max_questions", type=int, help="If set, limit the number of questions processed")
    p.add_argument("--seed", type=int, default=0, help="Random seed (for any sampling in downstream utils)")

    # Model options
    p.add_argument("--prefer", choices=["anthropic", "openai", "o4", "gemini"], default="anthropic", help="Preferred provider for CoT generation")
    p.add_argument("--top_k_umls", type=int, default=3, help="Top K UMLS candidates per surface")
    p.add_argument("--scispacy_when", choices=["auto", "always", "never"], default="auto", help="When to use scispaCy corroboration")

    # Server
    p.add_argument("--host", default="127.0.0.1", help="(server) host to bind")
    p.add_argument("--port", type=int, default=5005, help="(server) port to bind")

    # Logging & gates
    p.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    p.add_argument("--strict", action="store_true", help="Fail run if coverage/validity gates are not met")
    p.add_argument("--require_entailment", action="store_true", help="Run startup entailment smoke test; exit non-zero if it fails")
    p.add_argument("--require_umls", action="store_true", help="Run startup UMLS smoke test; exit non-zero if it fails")
    p.add_argument("--min_entailment_coverage", type=float, default=0.80, help="Minimum fraction of non-unknown entailment pairs (micro-average)")
    p.add_argument("--min_concept_valid_rate", type=float, default=0.70, help="Minimum fraction of steps with ≥1 valid UMLS concept")
    return p

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Logging
    logging.basicConfig(level=getattr(logging, args.log), format="%(asctime)s [%(levelname)s] %(message)s")

    # Seed
    if args.seed:
        random.seed(args.seed)

    # Quick env summary
    try:
        cg = _safe_import_cot_generator()
        openai_ok = bool(os.getenv("OPENAI_API_KEY", "")) or bool(getattr(cg, "OPENAI_READY", False)) if cg else False
        anthropic_ok = bool(os.getenv("ANTHROPIC_API_KEY", "")) or bool(getattr(cg, "ANTHROPIC_READY", False)) if cg else False
        gemini_ok = bool(os.getenv("GOOGLE_API_KEY", "")) or bool(getattr(cg, "GEMINI_READY", False)) if cg else False
        print(f"Python: {sys.executable}")
        if cg:
            try:
                print(f"cot_generator file: {inspect.getsourcefile(cg) or ''}")
            except Exception:
                pass
        print(f"OPENAI key ready? {bool(openai_ok)}")
        print(f"ANTHROPIC key ready? {bool(anthropic_ok)}")
        print(f"GOOGLE key ready? {bool(gemini_ok)}")
    except Exception:
        pass

    # Env overrides for smoke tests
    require_ent = args.require_entailment or (os.getenv("REQUIRE_ENTAILMENT", "") == "1")
    require_umls = args.require_umls or (os.getenv("REQUIRE_UMLS", "") == "1")

    # Server mode
    if args.serve:
        if require_ent:
            ok, msg = smoke_test_entailment()
            if not ok:
                print(f"[ERROR] Entailment smoke test failed: {msg}")
                return 3
            else:
                print(f"[OK] Entailment smoke test: {msg}")
        if require_umls:
            ok, msg = smoke_test_umls()
            if not ok:
                print(f"[ERROR] UMLS smoke test failed: {msg}")
                return 3
            else:
                print(f"[OK] UMLS smoke test: {msg}")
        return _run_server(args.host, int(args.port))

    # Determine questions
    questions: List[str] = []
    if args.batch:
        if not args.questions_file:
            print("[ERROR] --batch requires --questions_file")
            return 2
        questions = load_questions_file(args.questions_file)
        if args.max_questions and len(questions) > args.max_questions:
            questions = questions[: int(args.max_questions)]
        if not questions:
            print(f"[ERROR] No questions found in {args.questions_file}")
            return 2
    else:
        if args.question:
            questions = [args.question.strip()]
        elif args.questions_file:
            questions = load_questions_file(args.questions_file)
            if not questions:
                print(f"[ERROR] No questions found in {args.questions_file}")
                return 2
        else:
            print("[ERROR] Provide --question or --batch with --questions_file (or use --serve for API)")
            return 2

    # Startup smoke tests for CLI mode if required
    if require_ent:
        ok, msg = smoke_test_entailment()
        if not ok:
            print(f"[ERROR] Entailment smoke test failed: {msg}")
            return 3
        else:
            print(f"[OK] Entailment smoke test: {msg}")
    if require_umls:
        ok, msg = smoke_test_umls()
        if not ok:
            print(f"[ERROR] UMLS smoke test failed: {msg}")
            return 3
        else:
            print(f"[OK] UMLS smoke test: {msg}")

    out_dir = Path(os.path.expanduser(os.path.expandvars(args.out_dir)))
    write_artifacts = not bool(args.no_artifacts)
    _ensure_dir(out_dir)  # ensure base out_dir exists up front

    # Run
    if len(questions) == 1:
        ok, rec, path = process_question(
            questions[0],
            prefer=args.prefer,
            scispacy_when=args.scispacy_when,
            top_k_umls=int(args.top_k_umls),
            out_dir=out_dir,
            write_artifacts=write_artifacts,
        )
        if args.log_jsonl:
            log_path = Path(os.path.expanduser(os.path.expandvars(args.log_jsonl)))
            _ensure_dir(log_path.parent)
            with open(log_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if args.json_only:
            print(json.dumps(rec, ensure_ascii=False, indent=2))
        else:
            print("\n=== CoT Demo ===")
            print(f"Q: {rec['question']}")
            gm = rec.get("generator", {})
            print(f"Generator model: {gm.get('provider')} {gm.get('model')}")
            print("\nSteps:")
            for i, s in enumerate(rec.get("steps") or []):
                print(f"  {i+1}. {s}")
            if rec.get("entailment_pairs"):
                print("\nPairwise entailment (i → i+1):")
                for p_ in rec["entailment_pairs"]:
                    lab = p_["label"]; sc = p_["scores"]
                    print(f"  {p_['i']}->{p_['j']}:  {lab} | E {sc['E']:.1%}  N {sc['N']:.1%}  C {sc['C']:.1%}")
            if path:
                print(f"\nSaved: {path}")
        return 0

    # Batch mode
    code = run_batch(
        questions=questions,
        workers=int(args.workers),
        json_only=bool(args.json_only),
        log_jsonl=args.log_jsonl,
        prefer=args.prefer,
        scispacy_when=args.scispacy_when,
        top_k_umls=int(args.top_k_umls),
        strict=bool(args.strict),
        min_entail_cov=float(args.min_entailment_coverage),
        min_concept_rate=float(args.min_concept_valid_rate),
        out_dir=out_dir,
        write_artifacts=write_artifacts,
        metrics_json=args.metrics_json,
    )
    return code

if __name__ == "__main__":
    raise SystemExit(main())
