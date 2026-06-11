#!/usr/bin/env python3
# utils/concept_extractor_v2.py
# -*- coding: utf-8 -*-
"""
Standalone UMLS concept extractor — drop-in replacement for concept_extractor.py.

Key difference from concept_extractor.py (the original):
  Reads UMLS_API_KEY from os.environ at **call-time** rather than at module-import
  time, so it works correctly in notebook environments where the key is pasted in
  after the module has already been imported.

Same extraction strategies as concept_extractor.py:
  • N-gram extraction (2–5 tokens) with stopword filtering
  • Parenthetical content (abbreviations / expansions)
  • Standalone acronym detection (2–6 uppercase letters)
  • Full-step phrase when the step contains long biomedical tokens
  • CUI-level deduplication per step

Same output schema (dicts compatible with umls_density_scorer and
umls_specificity_scorer):
  {
    "text": <surface queried>,
    "cui": "Cxxxxxx",
    "canonical": "...",
    "semantic_types": [],
    "kb_sources": [...],
    "valid": True,
    "scores": {"api": 1.0, "link": 1.0, "confidence": 0.8}
  }
"""
from __future__ import annotations

import os
import re
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger("concepts_v2")

_search_cache: Dict[Tuple, List] = {}  # (query, top_k, search_type) → results

# ── Knobs (env-overridable, mirrors concept_extractor.py) ─────────────────────
MAX_SURFACES_PER_STEP   = int(os.getenv("CE_MAX_SURFACES_PER_STEP",   "12"))
MAX_CANDIDATES_PER_STEP = int(os.getenv("CE_MAX_CANDIDATES_PER_STEP", "32"))
RATE_SLEEP              = float(os.getenv("CE_RATE_SLEEP", "0.05"))   # sec between UMLS calls
NGRAM_MIN               = 2
NGRAM_MAX               = 5
UMLS_SEARCH_URL         = "https://uts-ws.nlm.nih.gov/rest/search/current"

# ── Stopwords (mirrors concept_extractor.py) ──────────────────────────────────
STOPWORDS = {
    "a","an","the","of","on","in","to","is","are","and","with","for","as","at","by","or","from",
    "via","into","than","that","this","those","these","be","been","being","was","were","will","would",
    "can","could","should","may","might","not","no","yes","it","its","their","there","then","thus",
    "we","our","you","your","i","he","she","they","them","his","her","which","when","where","how",
    "what","if","so","also","both","such","each","only","more","most","less","per","about","due",
}

# ── Regex helpers ─────────────────────────────────────────────────────────────
_SPLIT_PUNCT = re.compile(r"[.;:!?]|(?:\s-\s)")
_PARENS      = re.compile(r"\(([^)]{1,60})\)")
_ACRO        = re.compile(r"\b[A-Z]{2,6}\b")
_TOKEN_SPLIT = re.compile(r"[\s/,\-;]+")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _is_trivial(surface: str) -> bool:
    t = surface.lower()
    if not t or len(t) <= 1:
        return True
    if t in STOPWORDS:
        return True
    if re.fullmatch(r"\W+", t):
        return True
    return False


def _tokens(s: str) -> List[str]:
    parts = [re.sub(r"^[^\w]+|[^\w]+$", "", p) for p in _TOKEN_SPLIT.split(s)]
    return [p for p in parts if p and p.lower() not in STOPWORDS]


def _ngrams(ts: List[str], nmin: int = NGRAM_MIN, nmax: int = NGRAM_MAX) -> List[str]:
    out: List[str] = []
    L = len(ts)
    if L < nmin:
        return out
    for n in range(nmin, min(nmax, L) + 1):
        for i in range(L - n + 1):
            phrase = " ".join(ts[i:i+n])
            if not _is_trivial(phrase):
                out.append(phrase)
    seen, uniq = set(), []
    for p in out:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def _surface_candidates_from_step(step: str) -> List[str]:
    """Same strategy as concept_extractor._surface_candidates_from_step."""
    s = _norm(step)
    if not s:
        return []
    cands: List[str] = []
    # 1. Parenthetical content
    for inside in _PARENS.findall(s):
        inside = _norm(inside)
        if inside and not _is_trivial(inside):
            cands.append(inside)
    # 2. Phrase chunks split by major punctuation
    for chunk in _SPLIT_PUNCT.split(s):
        chunk = _norm(chunk)
        if chunk and not _is_trivial(chunk):
            cands.append(chunk)
    # 3. Standalone acronyms
    for ac in _ACRO.findall(s):
        if not _is_trivial(ac):
            cands.append(ac)
    # 4. N-grams
    ts = _tokens(s)
    cands.extend(_ngrams(ts))
    # 5. Full step when it contains long biomedical tokens
    if any(len(t) >= 6 for t in ts):
        cands.append(s[:120])
    # Deduplicate and cap
    seen, uniq = set(), []
    for v in cands:
        vv = _norm(v)
        k = vv.lower()
        if vv and not _is_trivial(vv) and k not in seen:
            seen.add(k)
            uniq.append(vv)
        if len(uniq) >= MAX_SURFACES_PER_STEP:
            break
    return uniq


def _search_umls(
    query: str,
    apikey: str,
    top_k: int = 3,
    search_type: str = "normalizedString",
) -> List[Dict[str, Any]]:
    """Call UMLS /search/current for a single surface. Returns concept dicts."""
    if not query or len(query) < 3:
        return []
    if len(query.split()) > 10 or sum(ch.isalpha() for ch in query) < 3:
        return []
    _key = (query, top_k, search_type if apikey else "local")
    if _key in _search_cache:
        return _search_cache[_key]
    if not apikey:
        try:
            from utils import umls_api_linker
            results = umls_api_linker.umls_search("", query, page_size=top_k)
            out = [
                {
                    "text":           query,
                    "cui":            r["cui"],
                    "canonical":      r.get("canonical", query),
                    "semantic_types": r.get("semantic_types", []),
                    "kb_sources":     r.get("kb_sources", ["UMLS"]),
                    "valid":          True,
                    "scores":         {"api": 1.0, "link": 1.0, "confidence": 0.8},
                }
                for r in results if r.get("cui")
            ]
        except Exception as exc:
            log.debug("[concepts_v2] local DB fallback failed for %r: %s", query, exc)
            out = []
        _search_cache[_key] = out
        return out
    try:
        r = requests.get(
            UMLS_SEARCH_URL,
            params={
                "string":       query,
                "apiKey":       apikey,
                "returnIdType": "concept",
                "searchType":   search_type,
                "pageSize":     top_k,
            },
            timeout=12,
        )
        if r.status_code != 200:
            return []
        results = r.json().get("result", {}).get("results", [])
        out: List[Dict[str, Any]] = []
        for res in results:
            ui = res.get("ui", "")
            if not ui or ui.upper() == "NONE":
                continue
            out.append({
                "text":           query,
                "cui":            ui,
                "canonical":      res.get("name", query),
                "semantic_types": [],
                "kb_sources":     [res.get("rootSource", "UMLS")],
                "valid":          True,
                "scores":         {"api": 1.0, "link": 1.0, "confidence": 0.8},
            })
        _search_cache[_key] = out
        return out
    except Exception as exc:
        log.debug("[concepts_v2] UMLS search failed for %r: %s", query, exc)
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def extract_concepts(
    steps: List[str],
    *,
    scispacy_when: str = "auto",
    top_k: int = 3,
    top_k_umls: Optional[int] = None,
    allowed_kb_sources: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[List[Dict[str, Any]]]:
    """
    Extract UMLS concepts for each reasoning step.

    Drop-in replacement for concept_extractor.extract_concepts.
    Reads UMLS_API_KEY from os.environ at call-time (fixes the stale
    module-level import issue common in notebooks).

    Returns per_step_concepts: List[List[Dict]] aligned to steps.
    Each dict has keys: text, cui, canonical, semantic_types, kb_sources,
    valid, scores.confidence — exactly what density and specificity scorers need.
    """
    apikey = os.getenv("UMLS_API_KEY", "")
    if not apikey:
        from utils import umls_api_linker
        if not umls_api_linker.is_configured():
            log.info("[concepts_v2] UMLS not configured; returning empty concept sets.")
            return [[] for _ in steps]

    K = int(top_k_umls or top_k or 3)
    per_step: List[List[Dict[str, Any]]] = []

    for step in steps:
        candidates: List[Dict[str, Any]] = []
        seen_cuis: set = set()
        surfaces = _surface_candidates_from_step(step or "")

        for surface in surfaces:
            if len(candidates) >= MAX_CANDIDATES_PER_STEP:
                break
            concepts = _search_umls(surface, apikey, top_k=K, search_type="normalizedString")
            # words fallback only needed for REST API; local DB handles its own fallback internally
            if not concepts and apikey:
                concepts = _search_umls(surface, apikey, top_k=K, search_type="words")
            for c in concepts:
                if c["cui"] not in seen_cuis:
                    seen_cuis.add(c["cui"])
                    candidates.append(c)
            if RATE_SLEEP > 0 and apikey:  # no sleep needed for local SQLite
                time.sleep(RATE_SLEEP)

        # Sort by confidence, cap
        candidates.sort(
            key=lambda r: float(((r.get("scores") or {}).get("confidence")) or 0.0),
            reverse=True,
        )
        per_step.append(candidates[:MAX_CANDIDATES_PER_STEP])

    linked = sum(1 for step in per_step if step)
    log.info("[concepts_v2] %d/%d steps linked", linked, len(steps))
    return per_step
