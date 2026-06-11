# utils/umls_regression_scorer.py
# -*- coding: utf-8 -*-
"""
CUI Specificity Regression Scorer.

Detects whether the reasoning chain "regresses" toward more generic /
less specific UMLS concepts as it progresses.  While the specificity
scorer measures the overall depth-slope, the regression scorer focuses
on a different signal: **concept novelty decay**.

Algorithm
---------
1. Maintain a running set of CUIs seen so far (the "knowledge frontier").
2. For each step, measure:
   a) **Novelty ratio** — fraction of the step's valid CUIs that have
      NOT been seen in any previous step.
   b) **Repeat ratio** — fraction of valid CUIs that are repeats.
3. Fit a linear slope over the novelty-ratio trajectory.
4. A **negative slope** means the model is progressively rehashing old
   concepts instead of introducing new ones — i.e., semantic regression.

The regression score is derived from the magnitude of the negative
novelty slope: a steeper decline → higher regression score.

Output schema (matches the key used in local_test HTML results):
    {
        "configured": True,
        "per_step": [
            {"step_index": 0, "new_cuis": 5, "repeat_cuis": 0,
             "novelty_ratio": 1.0},
            ...
        ],
        "novelty_slope": -0.12,
        "regression_score": 0.6,
    }
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger("regression")

# ─── Tunable constants ────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD    = 0.5   # min scores["confidence"] to include
REGRESSION_SLOPE_SCALE  = 3.0   # slope → score multiplier; slope of -0.33 → full risk


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


def _linear_slope(xs: List[float], ys: List[float]) -> Optional[float]:
    """Fit a linear slope over (xs, ys).  Returns None if < 2 points."""
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


def _overall_regression_score(slope: Optional[float]) -> float:
    """
    Convert novelty slope to a 0–1 regression score.
    Slope >= 0 (improving or stable novelty) → 0.0.
    Slope of -1/REGRESSION_SLOPE_SCALE or steeper → 1.0.
    """
    if slope is None or slope >= 0:
        return 0.0
    return min(1.0, abs(slope) * REGRESSION_SLOPE_SCALE)


# ─── Public API ───────────────────────────────────────────────────────────────

def score_regression(
    per_step_concepts: List[List[Dict]],
) -> Dict[str, Any]:
    """
    Compute the CUI Novelty Regression Score for a reasoning chain.

    Parameters
    ----------
    per_step_concepts:
        Already-extracted concept list (from concept_extractor).
        Each element is a list of concept dicts for one step.

    Returns
    -------
    Dict with keys: configured, per_step, novelty_slope,
    regression_score.
    """
    # Guard: UMLS must be configured
    try:
        from utils.umls_api_linker import is_configured
        configured = is_configured()
    except Exception:
        configured = False

    null_schema: Dict[str, Any] = {
        "configured": False,
        "per_step": [],
        "novelty_slope": None,
        "regression_score": 0.0,
    }

    if not configured:
        return null_schema

    if not per_step_concepts:
        return {**null_schema, "configured": True}

    # 1) Track CUI novelty across the chain
    seen_cuis: Set[str] = set()
    per_step_records = []
    novelty_ratios: List[float] = []

    for i, step_concepts in enumerate(per_step_concepts):
        valid = _valid_high_conf(step_concepts)
        step_cuis = {c["cui"] for c in valid if c.get("cui")}

        new_cuis = step_cuis - seen_cuis
        repeat_cuis = step_cuis & seen_cuis

        if step_cuis:
            novelty = len(new_cuis) / len(step_cuis)
        else:
            novelty = 0.0

        novelty_ratios.append(novelty)
        per_step_records.append({
            "step_index": i,
            "new_cuis": len(new_cuis),
            "repeat_cuis": len(repeat_cuis),
            "total_cuis": len(step_cuis),
            "novelty_ratio": round(novelty, 6),
        })

        # Update frontier
        seen_cuis |= step_cuis

    # 2) Fit slope over novelty ratios
    xs = list(range(len(novelty_ratios)))
    slope = _linear_slope([float(x) for x in xs], novelty_ratios)

    return {
        "configured": True,
        "per_step": per_step_records,
        "novelty_slope": round(slope, 6) if slope is not None else None,
        "regression_score": round(_overall_regression_score(slope), 4),
    }
