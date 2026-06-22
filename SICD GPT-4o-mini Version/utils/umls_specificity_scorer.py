
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

# ─── Tunable constants ────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD       = 0.5               # min scores["confidence"] to include a concept
PREFERRED_SOURCES          = ["SNOMEDCT_US", "MSH"]  # vocab preference for depth queries
ABSTRACTION_LEAP_THRESHOLD = 2.0               # depth drop magnitude to flag as sudden abstraction
SPECIFICITY_SLOPE_SCALE    = 0.2               # converts slope → score; slope of -5 → full risk
MAX_CONCEPTS_PER_STEP      = 1                 # top-N concepts (by confidence) processed per step




def _extract_source_code(code_val: str) -> str:
    """Extract just the source code identifier from a full UMLS REST URL or plain code."""
    if code_val and code_val.startswith("http"):
        return code_val.rstrip("/").split("/")[-1]
    return code_val


@lru_cache(maxsize=512)
def _cached_atoms_for_cui(apikey: str, version: str, cui: str) -> Tuple[Tuple[str, str], ...]:
    try:
        from utils.umls_api_linker import atoms_for_cui_direct
        atoms = atoms_for_cui_direct(apikey, version, cui, max_pages=1)
        return tuple(
            (a.get("rootSource", ""), _extract_source_code(a.get("code", "")))
            for a in atoms
            if a.get("rootSource") and a.get("code")
        )
    except Exception as e:
        print(f"  [specificity_err] atoms_for_cui({cui}) failed: {e}")
        return ()


@lru_cache(maxsize=2048)
def _cached_ancestor_depth(apikey: str, version: str, source: str, code: str) -> Optional[int]:
    try:
        from utils.umls_api_linker import hierarchy_for_source_code
        ancestors = hierarchy_for_source_code(apikey, version, source, code, "ancestors", max_pages=1)
        return len(ancestors)
    except Exception as e:
        print(f"  [specificity_err] hierarchy({source}/{code}) failed: {e}")
        return None


#  helpers 

def _get_source_code(
    concept: Dict,
    apikey: str,
    version: str,
) -> Optional[Tuple[str, str]]:
    cui = concept.get("cui")
    if not cui:
        return None

    atom_pairs = _cached_atoms_for_cui(apikey, version, cui)
    if not atom_pairs:
        return None

    # Check preferred sources in order
    for preferred in PREFERRED_SOURCES:
        for src, code in atom_pairs:
            if src == preferred and code:
                return (src, code)

    # Fallback: first non-UMLS source
    for src, code in atom_pairs:
        if src and src.upper() != "UMLS" and code:
            return (src, code)

    return None


def _step_avg_depth(
    step_concepts: List[Dict],
    apikey: str,
    version: str,
) -> Tuple[Optional[float], int]:
    """
    Compute the average ancestor depth for all valid, high-confidence concepts
    in a single step.

    Returns (avg_depth, concept_count) where avg_depth is None if no depths
    could be retrieved.
    """
    depths: List[int] = []
    # Take top-N concepts by confidence to limit API calls per step
    candidates = [
        c for c in step_concepts
        if c.get("valid") and (c.get("scores") or {}).get("confidence", 0.0) >= CONFIDENCE_THRESHOLD
    ]
    candidates.sort(key=lambda c: (c.get("scores") or {}).get("confidence", 0.0), reverse=True)
    for concept in candidates[:MAX_CONCEPTS_PER_STEP]:
        src_code = _get_source_code(concept, apikey, version)
        if src_code is None:
            continue

        src, code = src_code
        depth = _cached_ancestor_depth(apikey, version, src, code)
        if depth is not None:
            depths.append(depth)

    if not depths:
        return None, 0
    return sum(depths) / len(depths), len(depths)


def _linear_slope(xs: List[float], ys: List[float]) -> Optional[float]:
    """
    Fit a linear slope over (xs, ys).
    """
    if len(xs) < 2:
        return None
    try:
        import numpy as np  # type: ignore
        coeffs = np.polyfit(xs, ys, 1)
        return float(coeffs[0])
    except Exception:
        pass
    # Pure-Python least squares
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def _abstraction_leaps(step_depths: List[Optional[float]]) -> List[Dict[str, Any]]:
    """
    Find consecutive step pairs where the depth drops by more than
    ABSTRACTION_LEAP_THRESHOLD.  Skips pairs where either depth is None.
    """
    leaps = []
    for i in range(len(step_depths) - 1):
        d_i = step_depths[i]
        d_j = step_depths[i + 1]
        if d_i is None or d_j is None:
            continue
        drop = d_i - d_j
        if drop > ABSTRACTION_LEAP_THRESHOLD:
            leaps.append({
                "from_step": i,
                "to_step": i + 1,
                "depth_drop": round(drop, 4),
            })
    return leaps


def _overall_score(slope: Optional[float]) -> float:
    """
    Convert depth slope to a 0–1 score.
    A slope of 0 or positive → 0.0.
    A slope of -1/SPECIFICITY_SLOPE_SCALE or steeper → 1.0.
    """
    if slope is None or slope >= 0:
        return 0.0
    return min(1.0, abs(slope) * SPECIFICITY_SLOPE_SCALE)


# ─── Public API ───────────────────────────────────────────────────────────────

def score_specificity(
    per_step_concepts: List[List[Dict]],
) -> Dict[str, Any]:
    """
    Compute the MRHIER Specificity Depth Trajectory for a reasoning chain.

    Parameters
    
    per_step_concepts:
        Already-extracted concept list from concept_extractor.py.
        Each element is a list of concept dicts for one step.

    """
    null_schema: Dict[str, Any] = {
        "configured": False,
        "per_step": [],
        "depth_slope": None,
        "abstraction_leaps": [],
        "overall_specificity_score": 0.0,
    }

    # Guard: UMLS must be configured
    try:
        import os as _os
        from utils.umls_api_linker import is_configured, UMLS_API_KEY, DEFAULT_VERSION
        configured = is_configured()
    except Exception:
        return null_schema

    if not configured:
        return null_schema

    if not per_step_concepts:
        return {**null_schema, "configured": True}

    # Read key at call-time so keys set after module import are picked up
    apikey: str = _os.getenv("UMLS_API_KEY", "") or UMLS_API_KEY
    version: str = DEFAULT_VERSION

    # Build per-step depth records
    per_step_records = []
    step_depths: List[Optional[float]] = []

    from concurrent.futures import ThreadPoolExecutor

    # Because UMLS API requests can be slow, parallelize the fetching
    # across steps. This cuts down the total response time drastically.
    with ThreadPoolExecutor(max_workers=min(12, len(per_step_concepts) + 1)) as executor:
        futures = [
            executor.submit(_step_avg_depth, step_concepts, apikey, version)
            for step_concepts in per_step_concepts
        ]
        results = [fut.result() for fut in futures]

    for i, (avg_depth, concept_count) in enumerate(results):
        step_depths.append(avg_depth)
        per_step_records.append({
            "step_index": i,
            "avg_depth": round(avg_depth, 4) if avg_depth is not None else None,
            "concept_count": concept_count,
        })

    # Slope over non-None depths only
    valid_pairs = [
        (float(i), d) for i, d in enumerate(step_depths) if d is not None
    ]
    if len(valid_pairs) >= 2:
        xs, ys = zip(*valid_pairs)
        slope = _linear_slope(list(xs), list(ys))
    else:
        slope = None

    return {
        "configured": True,
        "per_step": per_step_records,
        "depth_slope": round(slope, 6) if slope is not None else None,
        "abstraction_leaps": _abstraction_leaps(step_depths),
        "overall_specificity_score": round(_overall_score(slope), 4),
    }
