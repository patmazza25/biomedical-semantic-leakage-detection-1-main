# utils/retrieval_stub.py — simple local sentence retrieval for evidence fallback
from __future__ import annotations
from typing import List
import glob

# Expect text files in data/corpus/ for evidence retrieval (if any)
_CORPUS = [p for p in glob.glob("data/corpus/*.txt")]

def retrieve_sentences(term_a: str, term_b: str, k: int = 5) -> List[str]:
    q1, q2 = term_a.lower(), term_b.lower()
    hits: List[str] = []
    for path in _CORPUS:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    L = s.lower()
                    if q1 in L and q2 in L:
                        hits.append(s)
                        if len(hits) >= k:
                            return hits
        except Exception:
            continue
    return hits
