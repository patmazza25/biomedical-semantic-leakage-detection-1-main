# guards.py
# Guard derivation utilities for step-to-step NLI decisions.
# Emits light-weight tags that the final-label policy can use
# to widen neutrality bands, suppress weak contradictions when
# ontology support exists, and highlight directional conflicts.

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


__all__ = [
    "GuardConfig",
    "derive_guards",
    "lexical_jaccard",
    "tokenize_for_jaccard",
]


@dataclass(frozen=True)
class GuardConfig:
    """
    Tunable knobs for guard emission. All can be overridden by env vars.

    Env overrides:
      LEXICAL_DUP_THRESHOLD   float, default 0.90
      LEXICAL_MIN_TOKEN_LEN   int,   default 2
      CAUTION_BAND_DELTA      float, default 0.07
      DIRECTION_CONFLICT_MARGIN float, default 0.10
      STRONG_CONFIDENCE       float, default 0.70
    """
    lexical_dupe_threshold: float = float(os.getenv("LEXICAL_DUP_THRESHOLD", "0.90"))
    min_token_len: int = int(os.getenv("LEXICAL_MIN_TOKEN_LEN", "2"))
    caution_delta: float = float(os.getenv("CAUTION_BAND_DELTA", "0.07"))
    direction_margin: float = float(os.getenv("DIRECTION_CONFLICT_MARGIN", "0.10"))
    strong_confidence: float = float(os.getenv("STRONG_CONFIDENCE", "0.70"))


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize_for_jaccard(text: str, min_len: int = 2) -> Set[str]:
    """
    Simple alnum tokenizer for similarity. Lowercase, no stopwording
    to stay model agnostic. Filter very short tokens to reduce noise.
    """
    if not text:
        return set()
    toks = (t.lower() for t in _WORD_RE.findall(text))
    return {t for t in toks if len(t) >= min_len}


def lexical_jaccard(a: str, b: str, *, min_len: int = 2) -> float:
    """
    Jaccard similarity of token sets, used for duplicate-step guard.
    """
    sa = tokenize_for_jaccard(a, min_len=min_len)
    sb = tokenize_for_jaccard(b, min_len=min_len)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    inter = sa & sb
    return len(inter) / len(union) if union else 0.0


def _sign_of_support(pe: float, pc: float, margin: float) -> int:
    """
    Returns +1 if entailment beats contradiction by >= margin,
            -1 if contradiction beats entailment by >= margin,
             0 otherwise (tie or within margin).
    """
    if pe - pc >= margin:
        return 1
    if pc - pe >= margin:
        return -1
    return 0


def derive_guards(
    premise: str,
    hypothesis: str,
    probs: Dict[str, float],
    *,
    relation_violation: bool = False,
    ontology_override_signal: bool = False,
    reverse_probs: Optional[Dict[str, float]] = None,
    config: GuardConfig = GuardConfig(),
) -> List[str]:
    """
    Compute qualitative guard tags for a premise → hypothesis pair.

    Inputs
    ------
    premise, hypothesis : raw strings of the compared statements
    probs               : dict with keys "entailment", "neutral", "contradiction"
    relation_violation  : True when ontology check finds no permitted relation
    ontology_override_signal : True when ontology indicates a permitted relation
                               (supported or provisional). This is the only case
                               where we emit "ontology_override".
    reverse_probs       : optional probs for the reverse direction (hyp → prem).
                          If provided, we may emit "direction_conflict".
    config              : GuardConfig with thresholds

    Returns
    -------
    list of string tags from the set:
      - "lexical_duplicate"
      - "caution_band"
      - "direction_conflict"
      - "relation_violation"
      - "ontology_override"
      - "provisional_support"
    """
    guards: List[str] = []

    # 1) Lexical duplicate guard (helps downweight near-copies)
    j = lexical_jaccard(
        premise or "",
        hypothesis or "",
        min_len=config.min_token_len,
    )
    if j >= config.lexical_dupe_threshold:
        guards.append("lexical_duplicate")

    # 2) Caution band near ties between entailment and contradiction
    pe = float(probs.get("entailment", 0.0))
    pc = float(probs.get("contradiction", 0.0))
    if abs(pe - pc) < config.caution_delta:
        guards.append("caution_band")

    # 3) Direction conflict: forward and reverse disagree with margin
    if reverse_probs is not None:
        pe_r = float(reverse_probs.get("entailment", 0.0))
        pc_r = float(reverse_probs.get("contradiction", 0.0))
        s_fwd = _sign_of_support(pe, pc, config.direction_margin)
        s_rev = _sign_of_support(pe_r, pc_r, config.direction_margin)
        if s_fwd != 0 and s_rev != 0 and s_fwd != s_rev:
            guards.append("direction_conflict")

    # 4) Ontology-informed tags
    #    We always record violations, but we only emit "ontology_override"
    #    when there is actual support (supported or provisional).
    if relation_violation:
        guards.append("relation_violation")

    if ontology_override_signal:
        guards.append("ontology_override")
        # If ontology supports the relation but model support is weak,
        # mark as provisional to let the decider nudge toward neutral
        # rather than hard entailment.
        if pe < config.strong_confidence:
            guards.append("provisional_support")

    return guards
