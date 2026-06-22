"""
utils/umls_local_db.py
Local UMLS SQLite query module — drop-in replacement for the REST API calls.

Activated by setting the environment variable:
    UMLS_LOCAL_DB_PATH=/path/to/umls_local.db

When active, the three core REST-API functions used by the pipeline
(umls_search, atoms_for_cui_direct, hierarchy_for_source_code) are
redirected here — no network, no rate limits, no timeouts.

Build the database first with:
    python scripts/build_local_umls.py
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# ── Configuration ─────────────────────────────────────────────────────────────

# Sources to prefer when looking up source codes for a CUI (must match what
# was loaded into the DB by build_local_umls.py).
PREFERRED_SOURCES: List[str] = ["SNOMEDCT_US", "MSH"]

_DB_PATH: str = os.getenv("UMLS_LOCAL_DB_PATH", "")
_conn: Optional[sqlite3.Connection] = None
_conn_lock = threading.Lock()


# ── Connection management ─────────────────────────────────────────────────────

def _get_conn() -> Optional[sqlite3.Connection]:
    """Return a cached read-only SQLite connection, or None if not configured."""
    global _conn, _DB_PATH
    # Re-read env at call time so keys set after import are picked up
    db_path = _DB_PATH or os.getenv("UMLS_LOCAL_DB_PATH", "")
    if not db_path or not os.path.exists(db_path):
        return None
    with _conn_lock:
        if _conn is None:
            _conn = sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True,
                check_same_thread=False,
            )
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA cache_size=-65536")   # 64 MB read cache
            _conn.execute("PRAGMA temp_store=MEMORY")
    return _conn


def is_available() -> bool:
    """True if UMLS_LOCAL_DB_PATH is set and the DB file exists."""
    return _get_conn() is not None


# ── FTS5 helpers ──────────────────────────────────────────────────────────────

_FTS_SPECIAL = re.compile(r'["\^\*\+\-]')


def _fts_phrase(term: str) -> str:
    """Wrap term in double-quotes for FTS5 phrase search; escape inner quotes."""
    escaped = term.replace('"', '""')
    return f'"{escaped}"'


def _fts_safe(term: str) -> str:
    """Strip FTS5 special characters for a plain token query."""
    return _FTS_SPECIAL.sub(" ", term).strip()


# ── Public API ────────────────────────────────────────────────────────────────

def search_strings_local(
    term: str,
    sabs: List[str],
    ttys: List[str],
    top_k: int = 7,
) -> List[Dict]:
    """
    Search MRCONSO for concepts matching `term`.

    Returns a list of candidate dicts in the same shape as umls_search():
      { cui, canonical, score, semantic_types, kb_sources }

    Strategy:
      1. FTS5 phrase match  →  "term"
      2. FTS5 token match   →  individual tokens (fallback)
      3. Exact LIKE match   →  LOWER(str) = LOWER(term)  (final fallback)
    """
    conn = _get_conn()
    if conn is None:
        return []

    term = term.strip()
    if not term:
        return []

    sabs_set = set(sabs) if sabs else set(PREFERRED_SOURCES)
    sabs_ph  = ",".join("?" * len(sabs_set))

    rows: List[sqlite3.Row] = []

    # 1. FTS5 phrase search
    try:
        q = (
            f"SELECT cui, str, sab, tty, code FROM mrconso_fts "
            f"WHERE mrconso_fts MATCH ? AND sab IN ({sabs_ph}) "
            f"LIMIT ?"
        )
        rows = conn.execute(q, [_fts_phrase(term)] + list(sabs_set) + [top_k * 3]).fetchall()
    except sqlite3.OperationalError:
        rows = []

    # 2. FTS5 token search (if phrase gave nothing)
    if not rows:
        safe = _fts_safe(term)
        if safe:
            try:
                q = (
                    f"SELECT cui, str, sab, tty, code FROM mrconso_fts "
                    f"WHERE mrconso_fts MATCH ? AND sab IN ({sabs_ph}) "
                    f"LIMIT ?"
                )
                rows = conn.execute(q, [safe] + list(sabs_set) + [top_k * 3]).fetchall()
            except sqlite3.OperationalError:
                rows = []

    if not rows:
        return []

    # Deduplicate by CUI — keep first occurrence (highest FTS5 rank)
    seen: Dict[str, sqlite3.Row] = {}
    for r in rows:
        cui = r["cui"]
        if cui not in seen:
            seen[cui] = r

    top_cuis = list(seen.keys())[:top_k]

    # Batch-fetch semantic types for all unique CUIs
    sty_map: Dict[str, List[str]] = {c: [] for c in top_cuis}
    if top_cuis:
        ph = ",".join("?" * len(top_cuis))
        for sty_row in conn.execute(
            f"SELECT cui, sty FROM mrsty WHERE cui IN ({ph})", top_cuis
        ).fetchall():
            sty_map[sty_row["cui"]].append(sty_row["sty"])

    # Build output dicts — shape matches _umls_search_multi() output
    results = []
    for rank, cui in enumerate(top_cuis):
        row = seen[cui]
        score = max(0.1, 1.0 - rank * 0.1)   # simple rank-decay score
        results.append({
            "cui":            cui,
            "canonical":      row["str"],
            "score":          score,
            "semantic_types": sty_map.get(cui, []),
            "kb_sources":     [row["sab"]],
        })

    return results


@lru_cache(maxsize=4096)
def atoms_for_cui_local(
    cui: str,
    sabs: Tuple[str, ...] = tuple(PREFERRED_SOURCES),
) -> List[Tuple[str, str]]:
    """
    Return list of (rootSource, code) pairs for a CUI.
    Mirrors the data returned by atoms_for_cui_direct().

    Prefers ISPREF='Y' (preferred atom) for each source.
    """
    conn = _get_conn()
    if conn is None:
        return []

    sabs_ph = ",".join("?" * len(sabs))
    rows = conn.execute(
        f"""SELECT sab, code FROM mrconso
            WHERE cui = ? AND sab IN ({sabs_ph}) AND suppress = 'N'
            ORDER BY CASE ispref WHEN 'Y' THEN 0 ELSE 1 END, sab""",
        [cui] + list(sabs),
    ).fetchall()

    # Return one (sab, code) per source vocabulary — the preferred atom
    seen: Dict[str, str] = {}
    for r in rows:
        if r["sab"] not in seen:
            seen[r["sab"]] = r["code"]

    return list(seen.items())


@lru_cache(maxsize=4096)
def ancestor_depth_local(sab: str, code: str) -> Optional[int]:
    """
    Return the ontology depth of a concept identified by (sab, code).

    Depth = number of ancestors = number of AUIs in the PTR path field of
    MRHIER.  PTR is a dot-delimited list of ancestor AUIs from the concept
    up to the root: "A1.A2.A3..." → depth 3.

    Returns None if the concept has no hierarchy entry.
    """
    conn = _get_conn()
    if conn is None:
        return None

    # Join mrconso → mrhier directly: the preferred atom often lacks a hierarchy
    # entry (only certain term types, e.g. MH for MSH, are indexed in MRHIER).
    # A single JOIN finds any AUI for this (sab, code) that has a PTR row.
    hier_row = conn.execute(
        """SELECT h.ptr FROM mrconso m
           JOIN mrhier h ON h.sab = m.sab AND h.aui = m.aui
           WHERE m.sab = ? AND m.code = ? AND m.suppress = 'N'
             AND h.ptr IS NOT NULL AND h.ptr != ''
           LIMIT 1""",
        [sab, code],
    ).fetchone()

    if hier_row is None:
        return None

    # PTR is dot-delimited: "AUI1.AUI2.AUI3" → len = depth
    ptr: str = hier_row["ptr"]
    depth = len([p for p in ptr.split(".") if p.strip()])
    return depth
