# utils/umls_oscillation_scorer.py
# -*- coding: utf-8 -*-
"""
CUI Semantic-Cluster Oscillation Scorer.

Measures how frequently the reasoning chain "switches" between distinct
UMLS semantic-type clusters across consecutive steps.  A high oscillation
score indicates the model is flip-flopping between unrelated medical
domains — a hallmark of semantic interference or conflicted reasoning.

Algorithm
---------
1. For each step, collect the set of unique CUI semantic types from
   high-confidence, valid concepts.
2. Represent each step as a binary vector over all observed semantic
   types (the "cluster fingerprint").
3. Compute the Jaccard *distance* (1 − Jaccard similarity) between
   every pair of consecutive steps.
4. The **oscillation score** is the mean of these consecutive-step
   distances.  A perfectly consistent chain scores 0; a chain that
   changes its entire semantic profile on every step scores 1.

Output schema (matches the key used in local_test HTML results):
    {
        "configured": True,
        "per_step": [
            {"step_index": 0, "semantic_types": [...], "n_types": 3},
            ...
        ],
        "consecutive_distances": [0.4, 0.6, ...],
        "oscillation_score": 0.45,
    }
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger("oscillation")

# ─── Tunable constants ────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5   # min scores["confidence"] to include a concept


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _valid_high_conf(step_concepts: List[Dict]) -> List[Dict]:
    """Return concepts that are marked valid AND have confidence >= threshold."""
    out = []
    for c in step_concepts:
        if not c.get("valid"):
            continue
        conf = (c.get("scores") or {}).get("confidence", 0.0)
        if conf >= CONFIDENCE_THRESHOLD:
            out.append(c)
    return out


def _semantic_types_for_step(step_concepts: List[Dict]) -> Set[str]:
    """Collect unique semantic type strings from valid high-confidence concepts."""
    types: Set[str] = set()
    for c in _valid_high_conf(step_concepts):
        for st in (c.get("semantic_types") or []):
            if st:
                types.add(st)
    return types


def _jaccard_distance(a: Set[str], b: Set[str]) -> float:
    """Jaccard distance = 1 - |A∩B| / |A∪B|.  Returns 0 if both empty."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


# ─── Public API ───────────────────────────────────────────────────────────────

def score_oscillation(
    per_step_concepts: List[List[Dict]],
) -> Dict[str, Any]:
    """
    Compute the CUI Semantic-Cluster Oscillation Score for a reasoning chain.

    Parameters
    ----------
    per_step_concepts:
        Already-extracted concept list (from concept_extractor).
        Each element is a list of concept dicts for one step.

    Returns
    -------
    Dict with keys: configured, per_step, consecutive_distances,
    oscillation_score.
    """
    # Guard: UMLS must be configured for concept data to be meaningful
    try:
        from utils.umls_api_linker import is_configured
        configured = is_configured()
    except Exception:
        configured = False

    null_schema: Dict[str, Any] = {
        "configured": False,
        "per_step": [],
        "consecutive_distances": [],
        "oscillation_score": 0.0,
    }

    if not configured:
        return null_schema

    if not per_step_concepts:
        return {**null_schema, "configured": True}

    # 1) Build per-step semantic-type sets
    per_step_records = []
    step_type_sets: List[Set[str]] = []

    for i, step_concepts in enumerate(per_step_concepts):
        stypes = _semantic_types_for_step(step_concepts)
        step_type_sets.append(stypes)
        per_step_records.append({
            "step_index": i,
            "semantic_types": sorted(stypes),
            "n_types": len(stypes),
        })

    # 2) Compute consecutive Jaccard distances
    distances: List[float] = []
    for i in range(len(step_type_sets) - 1):
        d = _jaccard_distance(step_type_sets[i], step_type_sets[i + 1])
        distances.append(round(d, 6))

    # 3) Oscillation score = mean of consecutive distances
    if distances:
        osc_score = sum(distances) / len(distances)
    else:
        osc_score = 0.0

    return {
        "configured": True,
        "per_step": per_step_records,
        "consecutive_distances": distances,
        "oscillation_score": round(osc_score, 6),
    }
