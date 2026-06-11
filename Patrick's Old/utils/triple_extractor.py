# utils/triple_extractor.py
from __future__ import annotations
from typing import List, Dict, Optional
import re

# Very light heuristic: subject (1–4 words) + verb + object (1–6 words)
_PAT = re.compile(r"^\s*([A-Z]?[A-Za-z0-9\- ]{1,40}?)\s+(reduces|increases|causes|treats|prevents|blocks|inhibits|leads|provokes)\s+([A-Za-z0-9\-\s,]{1,60})[.\s]*$", re.I)

def extract_triples(text: str) -> List[Dict[str, str]]:
    m = _PAT.match(text or "")
    if not m:
        return []
    subj, rel, obj = m.group(1).strip(), m.group(2).lower().strip(), m.group(3).strip().rstrip(".")
    if subj and rel and obj:
        return [{"subj": subj, "rel": rel, "obj": obj}]
    return []

# For compatibility, we might also define a single-triple extractor:
def extract_triple(text: str) -> Optional[Dict[str, str]]:
    triples = extract_triples(text)
    return triples[0] if triples else None

# Optionally, alias for expected function name
extract_triple_for_step = extract_triple
