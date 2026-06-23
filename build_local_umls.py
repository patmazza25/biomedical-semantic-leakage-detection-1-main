#!/usr/bin/env python3
# build_local_umls.py  (repo-root copy; adds `--sabs ALL`)
"""
One-time script: loads UMLS Metathesaurus RRF files into a local SQLite DB so the
SICD pipeline can run fully offline (seconds/chain instead of minutes over the REST API).

Only keeps English rows for the requested source vocabularies. Use `--sabs ALL` to keep
ALL English sources (closest to the REST API's all-source coverage; larger DB), or pass a
comma-separated list to keep the DB small.

Usage (from repo root):
    python build_local_umls.py --rrf-dir "C:\\umls\\META" --sabs ALL --db-path umls_local.db
    python build_local_umls.py --rrf-dir "C:\\umls\\META" --sabs SNOMEDCT_US,MSH,RXNORM
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

    """CREATE TABLE IF NOT EXISTS mrhier (
        cui  TEXT NOT NULL,
        aui  TEXT NOT NULL,
        sab  TEXT NOT NULL,
        ptr  TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_mrhier_sab_aui ON mrhier(sab, aui)",
    "CREATE INDEX IF NOT EXISTS idx_mrhier_cui     ON mrhier(cui)",

    """CREATE TABLE IF NOT EXISTS mrsty (
        cui TEXT NOT NULL,
        tui TEXT,
        sty TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_mrsty_cui ON mrsty(cui)",

    """CREATE VIRTUAL TABLE IF NOT EXISTS mrconso_fts
       USING fts5(str, cui, aui, sab, tty, code,
                  content='mrconso', content_rowid='rowid')""",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_lines(path: Path) -> int:
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
    class _FakeBar:
        def __init__(self): self._n = 0
        def __enter__(self): return self
        def __exit__(self, *a):
            print(f"  {desc.strip()} - {self._n:,} lines processed")
        def update(self, n=1): self._n += n
        def set_postfix(self, **kw): pass
    return _FakeBar()


# ── Loaders ───────────────────────────────────────────────────────────────────

_BATCH = 50_000


def _keep_sab(sab: str, sabs) -> bool:
    """sabs is None -> keep all sources; otherwise keep only listed sources."""
    return sabs is None or sab in sabs


def load_mrconso(conn: sqlite3.Connection, path: Path, sabs) -> int:
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
                if parts[1] != "ENG":        # English only
                    continue
                if not _keep_sab(parts[11], sabs):
                    continue
                if parts[16] != "N":         # unsuppressed only
                    continue
                rows.append((
                    parts[0], parts[1], parts[7], parts[11], parts[12],
                    parts[13], parts[14], parts[6], parts[16],
                ))
                if len(rows) >= _BATCH:
                    cur.executemany("INSERT INTO mrconso VALUES (?,?,?,?,?,?,?,?,?)", rows)
                    inserted += len(rows); rows.clear()
                    bar.set_postfix(inserted=f"{inserted:,}")
    if rows:
        cur.executemany("INSERT INTO mrconso VALUES (?,?,?,?,?,?,?,?,?)", rows)
        inserted += len(rows)
    conn.commit()
    print(f"  mrconso : {inserted:,} rows")
    return inserted


def load_mrhier(conn: sqlite3.Connection, path: Path, sabs) -> int:
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
                if not _keep_sab(parts[4], sabs):
                    continue
                rows.append((parts[0], parts[1], parts[4], parts[6]))
                if len(rows) >= _BATCH:
                    cur.executemany("INSERT INTO mrhier VALUES (?,?,?,?)", rows)
                    inserted += len(rows); rows.clear()
                    bar.set_postfix(inserted=f"{inserted:,}")
    if rows:
        cur.executemany("INSERT INTO mrhier VALUES (?,?,?,?)", rows)
        inserted += len(rows)
    conn.commit()
    print(f"  mrhier  : {inserted:,} rows")
    return inserted


def load_mrsty(conn: sqlite3.Connection, path: Path) -> int:
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
                rows.append((parts[0], parts[1], parts[3]))
                if len(rows) >= _BATCH:
                    cur.executemany("INSERT INTO mrsty VALUES (?,?,?)", rows)
                    inserted += len(rows); rows.clear()
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
    _project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build a local UMLS SQLite DB from RRF files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rrf-dir", default=str(_project_root),
                        help="Directory containing MRCONSO.RRF, MRHIER.RRF, MRSTY.RRF")
    parser.add_argument("--db-path", default=str(_project_root / "umls_local.db"),
                        help="Output SQLite database path")
    parser.add_argument("--sabs", default="SNOMEDCT_US,MSH,RXNORM",
                        help="Comma-separated source vocabularies, or ALL for every English source")
    args = parser.parse_args()

    rrf_dir = Path(args.rrf_dir)
    db_path = Path(args.db_path)
    raw_sabs = {s.strip() for s in args.sabs.split(",") if s.strip()}
    sabs = None if any(s.upper() == "ALL" for s in raw_sabs) else raw_sabs

    for fname in ("MRCONSO.RRF", "MRHIER.RRF", "MRSTY.RRF"):
        p = rrf_dir / fname
        if not p.exists():
            sys.exit(f"ERROR: {p} not found.\nSet --rrf-dir to the directory containing the RRF files.")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        print(f"Removing existing DB at {db_path}")
        db_path.unlink()

    print("\nBuilding UMLS local DB")
    print(f"  RRF dir  : {rrf_dir}")
    print(f"  DB path  : {db_path}")
    print(f"  Sources  : {'ALL English sources' if sabs is None else ', '.join(sorted(sabs))}")
    print()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-131072")

    for stmt in DDL:
        conn.execute(stmt)
    conn.commit()

    load_mrconso(conn, rrf_dir / "MRCONSO.RRF", sabs)
    load_mrhier (conn, rrf_dir / "MRHIER.RRF",  sabs)
    load_mrsty  (conn, rrf_dir / "MRSTY.RRF")
    build_fts   (conn)

    conn.close()

    size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"\nDone.  DB size: {size_mb:.0f} MB  ->  {db_path}")
    print(f"\nNext: the notebooks auto-detect ./umls_local.db (repo root). "
          f"Or set UMLS_LOCAL_DB_PATH={db_path}")


if __name__ == "__main__":
    main()
