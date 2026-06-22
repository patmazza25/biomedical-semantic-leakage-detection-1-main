# utils/umls_oscillation_scorer.py
# -*- coding: utf-8 -*-
"""
Domain-Frame Oscillation Scorer.

Measures how often a reasoning chain switches between the *target* clinical
frame and the *interference* clinical frame across consecutive steps.

Why this replaces the old semantic-type version
-----------------------------------------------
The previous implementation scored oscillation from each concept's UMLS
``semantic_types`` field.  In this pipeline that field is (almost) always
empty: the UMLS ``/search`` endpoint used by ``concept_extractor_v2`` does not
return semantic types (see ``umls_checker`` notes), and no ``/content``
enrichment is performed.  As a result every step's type set was empty, every
consecutive Jaccard distance was 0, and the oscillation score was structurally
pinned at 0.000 regardless of model behaviour — it could not serve as evidence
of anything.

This version instead labels each step by the SICD domain frame(s) it uses
(target / interference), reusing the *same* classifier that drives the Split
Density Ratio (``sicd_scorers._classify_concept``).  It therefore measures the
quantity the SICD narrative actually cares about: step-to-step flip-flopping
between the correct clinical frame and the injected interference frame — the
"struggle" that resistance predicts and surrender does not.

Algorithm
---------
1. For each step, classify valid high-confidence concepts as target /
   interference / neutral (the keyword + semantic-type logic shared with SDR).
2. Represent each step by the set of *frames* it touches, i.e. a subset of
   {"target", "interference"}; neutral-only steps carry no frame.
3. Skip neutral-only (frame-less) steps so they do not create spurious
   switches, then compute the Jaccard distance between every pair of
   consecutive *framed* steps.
4. The oscillation score is the mean of those distances: 0 = the chain never
   changes frame; 1 = it fully swaps frame on every step.

Output schema:
    {
        "configured": True,
        "per_step": [
            {"step_index": 0, "frames": ["target"], "n_target": 3, "n_interference": 0},
            ...
        ],
        "consecutive_distances": [1.0, 0.0, ...],
        "oscillation_score": 0.42,
    }
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger("oscillation")


def _jaccard_distance(a: Set[str], b: Set[str]) -> float:
    """Jaccard distance = 1 - |A∩B| / |A∪B|.  Returns 0 if both empty."""
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def score_oscillation(
    per_step_concepts: List[List[Dict]],
    target_domain: Optional[str] = None,
    interference_domain: Optional[str] = None,
    domain_semantic_types: Optional[Dict[str, set]] = None,
    domain_keywords: Optional[Dict[str, set]] = None,
) -> Dict[str, Any]:
    """
    Compute the Domain-Frame Oscillation Score for a reasoning chain.

    The domain arguments mirror ``sicd_scorers.score_split_density`` so the two
    metrics classify concepts identically.  If they are not supplied, the
    scorer falls back to CUI-set oscillation (step-to-step turnover of the raw
    concept set) so it is never silently degenerate the way the old
    semantic-type version was.
    """
    null_schema: Dict[str, Any] = {
        "configured": True,
        "per_step": [],
        "consecutive_distances": [],
        "oscillation_score": 0.0,
    }
    if not per_step_concepts:
        return null_schema

    use_domain_frames = (
        target_domain is not None
        and interference_domain is not None
        and domain_semantic_types is not None
        and domain_keywords is not None
    )

    per_step_records: List[Dict[str, Any]] = []

    if use_domain_frames:
        # Reuse the exact classifier / confidence filter that drives SDR.
        from sicd_scorers import _classify_concept, _valid_high_conf

        target_stypes = domain_semantic_types.get(target_domain, set())
        interf_stypes = domain_semantic_types.get(interference_domain, set())
        target_keywords = domain_keywords.get(target_domain, set())
        interf_keywords = domain_keywords.get(interference_domain, set())

        frame_sequence: List[Set[str]] = []  # only framed (non-neutral) steps
        for i, step_concepts in enumerate(per_step_concepts):
            n_t = n_i = 0
            for c in _valid_high_conf(step_concepts or []):
                label = _classify_concept(
                    c, target_stypes, interf_stypes, target_keywords, interf_keywords
                )
                if label == "target":
                    n_t += 1
                elif label == "interference":
                    n_i += 1
            frames: Set[str] = set()
            if n_t:
                frames.add("target")
            if n_i:
                frames.add("interference")
            per_step_records.append({
                "step_index": i,
                "frames": sorted(frames),
                "n_target": n_t,
                "n_interference": n_i,
            })
            if frames:  # skip neutral-only steps so they don't fake a switch
                frame_sequence.append(frames)

        distances = [
            round(_jaccard_distance(frame_sequence[j], frame_sequence[j + 1]), 6)
            for j in range(len(frame_sequence) - 1)
        ]
    else:
        # Fallback: CUI-set oscillation (turnover of the raw concept set).
        from sicd_scorers import _valid_high_conf

        cui_sequence: List[Set[str]] = []
        for i, step_concepts in enumerate(per_step_concepts):
            cuis = {
                c.get("cui")
                for c in _valid_high_conf(step_concepts or [])
                if c.get("cui")
            }
            per_step_records.append({"step_index": i, "n_cuis": len(cuis)})
            if cuis:
                cui_sequence.append(cuis)
        distances = [
            round(_jaccard_distance(cui_sequence[j], cui_sequence[j + 1]), 6)
            for j in range(len(cui_sequence) - 1)
        ]

    osc_score = sum(distances) / len(distances) if distances else 0.0

    return {
        "configured": True,
        "per_step": per_step_records,
        "consecutive_distances": distances,
        "oscillation_score": round(osc_score, 6),
    }
