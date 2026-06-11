

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

# ─── Tunable constants ────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5   # min scores["confidence"] to count a concept
DENSITY_SLOPE_SCALE  = 5.0   # converts slope → risk; slope of -0.2 → full risk
MIN_WORD_COUNT       = 1     # floor to avoid division by zero


# Helpers 

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


def _word_count(step_text: str) -> int:
    """Word count of a step string, floored at MIN_WORD_COUNT."""
    return max(MIN_WORD_COUNT, len(step_text.split()))


def _density(valid_concepts: List[Dict], word_count: int) -> float:
    """Ratio of valid high-confidence concepts to words."""
    return len(valid_concepts) / word_count


def _linear_slope(xs: List[float], ys: List[float]) -> Optional[float]:
    """
    Fit a linear slope over (xs, ys).
    Tries numpy.polyfit first; falls back to pure-Python least squares.
    Returns None if fewer than 2 points.
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


def _leakage_onset(densities: List[float]) -> Optional[int]:
    """
    Return the 0-based step index where the sharpest single-step density
    drop occurs.  Returns None if there are no drops.
    """
    if len(densities) < 2:
        return None
    drops = [
        (densities[i] - densities[i + 1], i + 1)
        for i in range(len(densities) - 1)
    ]
    negative_drops = [(d, idx) for d, idx in drops if d > 0]
    if not negative_drops:
        return None
    return max(negative_drops, key=lambda x: x[0])[1]


def _overall_risk(slope: Optional[float]) -> float:
    """
    Convert slope to a 0–1 risk score.
    A slope of 0 or positive → risk 0.0.
    A slope of -1/DENSITY_SLOPE_SCALE or steeper → risk 1.0.
    """
    if slope is None or slope >= 0:
        return 0.0
    return min(1.0, abs(slope) * DENSITY_SLOPE_SCALE)


# ─── Public API ───────────────────────────────────────────────────────────────

def score_density(
    per_step_concepts: List[List[Dict]],
    steps: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute the CUI Grounding Density Slope for a reasoning chain.

    Parameters
    ----------
    per_step_concepts:
        Already-extracted concept list from concept_extractor.py.
        Each element is a list of concept dicts for one step.
    steps:
        Raw step text list (same order).  Used for accurate word counts.
        If omitted, word count is estimated from concept count alone.
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
        "slope": None,
        "leakage_onset_step": None,
        "overall_risk": 0.0,
    }

    if not configured:
        return null_schema

    if not per_step_concepts:
        return {**null_schema, "configured": True}

    # Build per-step density records
    per_step_records = []
    densities: List[float] = []

    for i, step_concepts in enumerate(per_step_concepts):
        valid = _valid_high_conf(step_concepts)
        if steps and i < len(steps):
            wc = _word_count(steps[i])
        else:
            # Fallback: use total concept count as proxy word count floor
            wc = max(MIN_WORD_COUNT, len(step_concepts))
        d = _density(valid, wc)
        densities.append(d)
        per_step_records.append({
            "step_index": i,
            "valid_concept_count": len(valid),
            "word_count": wc,
            "density": round(d, 6),
        })

    # Slope over step indices
    xs = list(range(len(densities)))
    slope = _linear_slope(xs, densities)

    return {
        "configured": True,
        "per_step": per_step_records,
        "slope": round(slope, 6) if slope is not None else None,
        "leakage_onset_step": _leakage_onset(densities),
        "overall_risk": round(_overall_risk(slope), 4),
    }
