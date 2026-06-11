#!/usr/bin/env python3
# scripts/build_local_umls.py
"""
One-time script: loads UMLS Metathesaurus RRF files into a local SQLite DB.

Only keeps English rows for the requested source vocabularies (default:
SNOMEDCT_US and MSH) to keep the resulting DB small (~400-600 MB).

Usage (from project root):
    python scripts/build_local_umls.py
    python scripts/build_local_umls.py --rrf-dir C:\\umls\\META --db-path utils\\umls_local.db
    python scripts/build_local_umls.py --sabs SNOMEDCT_US,MSH,RXNORM
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ── Schema ────────────────────────────────────────────────────────────────────

DDL = [
    # MRCONSO: concept strings, AUIs, source codes
    """CREATE TABLE IF NOT EXISTS mrconso (
        cui      TEXT NOT NULL,
        lat      TEXT,
        aui      TEXT NOT NULL,
        sab      TEXT NOT NULL,
        tty      TEXT,
        code     TEXT,
        str      TEXT,
        ispref   TEXT,
        suppress TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_mrconso_cui      ON mrconso(cui)",
    "CREATE INDEX IF NOT EXISTS idx_mrconso_str      ON mrconso(str COLLATE NOCASE)",
    "CREATE INDEX IF NOT EXISTS idx_mrconso_sab_code ON mrconso(sab, code)",
    "CREATE INDEX IF NOT EXISTS idx_mrconso_cui_sab  ON mrconso(cui, sab)",

    # MRHIER: hierarchy paths — PTR is pipe-delimited path to ontology root
    """CREATE TABLE IF NOT EXISTS mrhier (
        cui  TEXT NOT NULL,
        aui  TEXT NOT NULL,
        sab  TEXT NOT NULL,
        ptr  TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_mrhier_sab_aui ON mrhier(sab, aui)",
    "CREATE INDEX IF NOT EXISTS idx_mrhier_cui     ON mrhier(cui)",

    # MRSTY: semantic types
    """CREATE TABLE IF NOT EXISTS mrsty (
        cui TEXT NOT NULL,
        tui TEXT,
        sty TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_mrsty_cui ON mrsty(cui)",

    # FTS5 virtual table for fast string search
    """CREATE VIRTUAL TABLE IF NOT EXISTS mrconso_fts
       USING fts5(str, cui, aui, sab, tty, code,
                  content='mrconso', content_rowid='rowid')""",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_lines(path: Path) -> int:
    """Fast byte-level line count."""
    count = 0
    with open(path, "rb") as f:
        buf = f.read(1 << 20)
        while buf:
            count += buf.count(b"\n")
            buf = f.read(1 << 20)
    return count


def _make_bar(total: int, desc: str):
    if _HAS_TQDM:
        return tqdm(total=total, desc=desc, unit="lines", unit_scale=True)
    # Minimal fallback
    class _FakeBar:
        def __init__(self): self._n = 0
        def __enter__(self): return self
        def __exit__(self, *a):
            print(f"  {desc.strip()} — {self._n:,} lines processed")
        def update(self, n=1): self._n += n
        def set_postfix(self, **kw): pass
    return _FakeBar()


# ── Loaders ───────────────────────────────────────────────────────────────────

_BATCH = 50_000


def load_mrconso(conn: sqlite3.Connection, path: Path, sabs: set) -> int:
    """
    MRCONSO.RRF columns (pipe-delimited, trailing pipe):
    0:CUI  1:LAT  2:TS  3:LUI  4:STT  5:SUI  6:ISPREF  7:AUI  8:SAUI
    9:SCUI 10:SDUI 11:SAB 12:TTY 13:CODE 14:STR 15:SRL 16:SUPPRESS 17:CVF
    """
    cur = conn.cursor()
    inserted = 0
    rows: list = []

    with _make_bar(_count_lines(path), "MRCONSO") as bar:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                bar.update(1)
                parts = line.rstrip("\n").split("|")
                if len(parts) < 17:
                    continue
                if parts[1] != "ENG":        # LAT — English only
                    continue
                if parts[11] not in sabs:    # SAB — requested sources only
                    continue
                if parts[16] != "N":         # SUPPRESS — keep unsuppressed only
                    continue
                rows.append((
                    parts[0],   # cui
                    parts[1],   # lat
                    parts[7],   # aui
                    parts[11],  # sab
                    parts[12],  # tty
                    parts[13],  # code
                    parts[14],  # str
                    parts[6],   # ispref
                    parts[16],  # suppress
                ))
                if len(rows) >= _BATCH:
                    cur.executemany("INSERT INTO mrconso VALUES (?,?,?,?,?,?,?,?,?)", rows)
                    inserted += len(rows)
                    rows.clear()
                    bar.set_postfix(inserted=f"{inserted:,}")

    if rows:
        cur.executemany("INSERT INTO mrconso VALUES (?,?,?,?,?,?,?,?,?)", rows)
        inserted += len(rows)

    conn.commit()
    print(f"  mrconso : {inserted:,} rows")
    return inserted


def load_mrhier(conn: sqlite3.Connection, path: Path, sabs: set) -> int:
    """
    MRHIER.RRF columns:
    0:CUI  1:AUI  2:CXN  3:PAUI  4:SAB  5:RELA  6:PTR  7:HCD  8:CVF
    """
    cur = conn.cursor()
    inserted = 0
    rows: list = []

    with _make_bar(_count_lines(path), "MRHIER ") as bar:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                bar.update(1)
                parts = line.rstrip("\n").split("|")
                if len(parts) < 7:
                    continue
                if parts[4] not in sabs:     # SAB filter
                    continue
                rows.append((
                    parts[0],  # cui
                    parts[1],  # aui
                    parts[4],  # sab
                    parts[6],  # ptr  (pipe-delimited ancestor path)
                ))
                if len(rows) >= _BATCH:
                    cur.executemany("INSERT INTO mrhier VALUES (?,?,?,?)", rows)
                    inserted += len(rows)
                    rows.clear()
                    bar.set_postfix(inserted=f"{inserted:,}")

    if rows:
        cur.executemany("INSERT INTO mrhier VALUES (?,?,?,?)", rows)
        inserted += len(rows)

    conn.commit()
    print(f"  mrhier  : {inserted:,} rows")
    return inserted


def load_mrsty(conn: sqlite3.Connection, path: Path) -> int:
    """
    MRSTY.RRF columns:
    0:CUI  1:TUI  2:STN  3:STY  4:ATUI  5:CVF
    """
    cur = conn.cursor()
    inserted = 0
    rows: list = []

    with _make_bar(_count_lines(path), "MRSTY  ") as bar:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                bar.update(1)
                parts = line.rstrip("\n").split("|")
                if len(parts) < 4:
                    continue
                rows.append((parts[0], parts[1], parts[3]))  # cui, tui, sty
                if len(rows) >= _BATCH:
                    cur.executemany("INSERT INTO mrsty VALUES (?,?,?)", rows)
                    inserted += len(rows)
                    rows.clear()
                    bar.set_postfix(inserted=f"{inserted:,}")

    if rows:
        cur.executemany("INSERT INTO mrsty VALUES (?,?,?)", rows)
        inserted += len(rows)

    conn.commit()
    print(f"  mrsty   : {inserted:,} rows")
    return inserted


def build_fts(conn: sqlite3.Connection) -> None:
    print("Building FTS5 index (may take a few minutes)...")
    conn.execute("INSERT INTO mrconso_fts(mrconso_fts) VALUES('rebuild')")
    conn.commit()
    print("  FTS5 index built.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Default rrf_dir = project root (parent of this script's directory)
    _project_root = Path(__file__).resolve().parent.parent
    _default_rrf  = str(_project_root)
    _default_db   = str(_project_root / "utils" / "umls_local.db")

    parser = argparse.ArgumentParser(
        description="Build a local UMLS SQLite DB from RRF files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rrf-dir", default=_default_rrf,
                        help="Directory containing MRCONSO.RRF, MRHIER.RRF, MRSTY.RRF")
    parser.add_argument("--db-path", default=_default_db,
                        help="Output SQLite database path")
    parser.add_argument("--sabs", default="SNOMEDCT_US,MSH",
                        help="Comma-separated source vocabularies to include")
    args = parser.parse_args()

    rrf_dir = Path(args.rrf_dir)
    db_path = Path(args.db_path)
    sabs    = {s.strip() for s in args.sabs.split(",") if s.strip()}

    # Validate RRF files exist
    for fname in ("MRCONSO.RRF", "MRHIER.RRF", "MRSTY.RRF"):
        p = rrf_dir / fname
        if not p.exists():
            sys.exit(
                f"ERROR: {p} not found.\n"
                f"Set --rrf-dir to the directory containing the RRF files."
            )

    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        print(f"Removing existing DB at {db_path}")
        db_path.unlink()

    print(f"\nBuilding UMLS local DB")
    print(f"  RRF dir  : {rrf_dir}")
    print(f"  DB path  : {db_path}")
    print(f"  Sources  : {', '.join(sorted(sabs))}")
    print()

    conn = sqlite3.connect(db_path)
    # Performance pragmas for bulk insert
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-131072")   # 128 MB page cache

    for stmt in DDL:
        conn.execute(stmt)
    conn.commit()

    load_mrconso(conn, rrf_dir / "MRCONSO.RRF", sabs)
    load_mrhier (conn, rrf_dir / "MRHIER.RRF",  sabs)
    load_mrsty  (conn, rrf_dir / "MRSTY.RRF")
    build_fts   (conn)

    conn.close()

    size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"\nDone.  DB size: {size_mb:.0f} MB  →  {db_path}")
    print(f"\nNext step — set this env var before running the pipeline:")
    print(f'  UMLS_LOCAL_DB_PATH="{db_path}"')


if __name__ == "__main__":
    main()
