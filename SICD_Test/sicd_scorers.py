# sicd_scorers.py
# -*- coding: utf-8 -*-
"""
SICD-specific scoring: Split Density signal.

The novel contribution of the SICD experiment.  Instead of measuring raw
UMLS concept density (which conflates target-relevant and interference-
relevant concepts), we split the density into two components:

    Target-Relevant Density   — concepts in the correct diagnostic domain
    Interference-Relevant Density — concepts in the interference domain

The **Density Ratio** = Target / (Target + Interference) is expected to
drop from ~1.0 (control) to ~0.0 (full dissonance) across interference
levels, providing a much stronger signal than raw density.

Classification uses two strategies (OR logic):
1. UMLS Semantic Type matching (primary, when semantic types are available)
2. Keyword matching on concept canonical names (fallback)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# ─── Helpers ──────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.5


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


def _classify_concept(
    concept: Dict,
    target_stypes: Set[str],
    interf_stypes: Set[str],
    target_keywords: Set[str],
    interf_keywords: Set[str],
) -> str:
    """
    Classify a single concept as 'target', 'interference', or 'neutral'.

    Strategy:
    1. Check semantic types first (more reliable).
    2. Fall back to keyword matching on canonical name / surface text.
    3. If a concept matches BOTH domains, classify as 'interference'
       (conservative — we want to detect when interference is present).
    4. If it matches neither, classify as 'neutral'.
    """
    # Semantic type classification
    concept_stypes = set(concept.get("semantic_types") or [])
    in_target = bool(concept_stypes & target_stypes)
    in_interf = bool(concept_stypes & interf_stypes)

    # Keyword fallback (check canonical name and surface text)
    if not in_target and not in_interf:
        text_lower = (
            (concept.get("canonical") or "") + " " + (concept.get("text") or "")
        ).lower()

        for kw in target_keywords:
            if kw in text_lower:
                in_target = True
                break

        for kw in interf_keywords:
            if kw in text_lower:
                in_interf = True
                break

    # Resolution
    if in_interf and in_target:
        # Concept straddles both domains — count as interference
        # (conservative: we want to detect semantic leakage)
        return "interference"
    elif in_interf:
        return "interference"
    elif in_target:
        return "target"
    else:
        return "neutral"


def _linear_slope(xs: List[float], ys: List[float]) -> Optional[float]:
    """Fit a linear slope.  Returns None if < 2 points."""
    if len(xs) < 2:
        return None
    try:
        import numpy as np
        coeffs = np.polyfit(xs, ys, 1)
        return float(coeffs[0])
    except Exception:
        pass
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


# ─── Public API ───────────────────────────────────────────────────────────────

def score_split_density(
    per_step_concepts: List[List[Dict]],
    target_domain: str,
    interference_domain: str,
    domain_semantic_types: Dict[str, set],
    domain_keywords: Dict[str, set],
    steps: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute Split Density: target-relevant vs interference-relevant concept density.

    Parameters
    ----------
    per_step_concepts : list of list of concept dicts
    target_domain : e.g. 'respiratory'
    interference_domain : e.g. 'endocrine'
    domain_semantic_types : from sicd_cases.DOMAIN_SEMANTIC_TYPES
    domain_keywords : from sicd_cases.DOMAIN_KEYWORDS
    steps : raw step text list for word counts

    Returns
    -------
    Dict with:
        per_step: list of per-step classification breakdowns
        target_density_slope: slope of target density across steps
        interference_density_slope: slope of interference density across steps
        density_ratio_slope: slope of ratio across steps
        mean_density_ratio: mean of target/(target+interference) across steps
    """
    target_stypes = domain_semantic_types.get(target_domain, set())
    interf_stypes = domain_semantic_types.get(interference_domain, set())
    target_kw = domain_keywords.get(target_domain, set())
    interf_kw = domain_keywords.get(interference_domain, set())

    per_step_records = []
    target_densities: List[float] = []
    interf_densities: List[float] = []
    density_ratios: List[float] = []

    for i, step_concepts in enumerate(per_step_concepts):
        valid = _valid_high_conf(step_concepts)

        # Word count for density normalisation
        if steps and i < len(steps):
            wc = max(1, len((steps[i] or "").split()))
        else:
            wc = max(1, len(valid))

        # Classify each concept
        n_target = 0
        n_interf = 0
        n_neutral = 0
        classifications = []

        for c in valid:
            cls = _classify_concept(c, target_stypes, interf_stypes, target_kw, interf_kw)
            classifications.append(cls)
            if cls == "target":
                n_target += 1
            elif cls == "interference":
                n_interf += 1
            else:
                n_neutral += 1

        td = n_target / wc
        id_ = n_interf / wc
        total_relevant = n_target + n_interf
        ratio = n_target / total_relevant if total_relevant > 0 else 1.0

        target_densities.append(td)
        interf_densities.append(id_)
        density_ratios.append(ratio)

        per_step_records.append({
            "step_index": i,
            "word_count": wc,
            "n_target": n_target,
            "n_interference": n_interf,
            "n_neutral": n_neutral,
            "target_density": round(td, 6),
            "interference_density": round(id_, 6),
            "density_ratio": round(ratio, 6),
        })

    # Slopes
    xs = [float(x) for x in range(len(per_step_records))]
    td_slope = _linear_slope(xs, target_densities)
    id_slope = _linear_slope(xs, interf_densities)
    ratio_slope = _linear_slope(xs, density_ratios)

    mean_ratio = (
        sum(density_ratios) / len(density_ratios) if density_ratios else 1.0
    )

    return {
        "target_domain": target_domain,
        "interference_domain": interference_domain,
        "per_step": per_step_records,
        "target_density_slope": round(td_slope, 6) if td_slope is not None else None,
        "interference_density_slope": round(id_slope, 6) if id_slope is not None else None,
        "density_ratio_slope": round(ratio_slope, 6) if ratio_slope is not None else None,
        "mean_density_ratio": round(mean_ratio, 6),
    }
