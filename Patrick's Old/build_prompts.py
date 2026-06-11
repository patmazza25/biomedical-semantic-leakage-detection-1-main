#!/usr/bin/env python3
"""
build_prompts.py — JSON-only exporter (fixed)

Creates an evaluation pack of biomedical prompts from public QA sources.

Writes (JSON ONLY):
  - prompts_<N>.jsonl   (prompt + provenance; one JSON per line, with prompt_id)
  - answers_<N>.jsonl   (MCQ answer key only; one JSON per line)
  - manifest.json       (summary + paths + split listing)

Optional JSON splits:
  - out_dir/json_splits/<start-end>/prompts.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from datasets import load_dataset
except Exception as e:
    raise SystemExit(
        "This script requires the 'datasets' package. Install with:\n"
        "    pip install datasets\n"
        f"Import error: {e}"
    )

# ----------------------------- Config & Helpers -----------------------------

CORE_KEYS = ("pubmedqa", "medqa", "medmcqa", "liveqa")


def to_multiple_of_4(n: int) -> int:
    """Round UP to nearest multiple of 4."""
    return ((n + 3) // 4) * 4


def norm_q(text: str) -> str:
    """Normalize a question string for cross-dataset de-duplication."""
    if not isinstance(text, str):
        text = str(text or "")
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\s]+", "", t)
    return t


def near_dupe(a: str, b: str, *, j_thresh: float) -> bool:
    """Near-duplicate check via token-overlap Jaccard with threshold j_thresh."""
    if j_thresh <= 0:
        return False
    sa = set(norm_q(a).split())
    sb = set(norm_q(b).split())
    if not sa or not sb:
        return False
    j = len(sa & sb) / max(1, len(sa | sb))
    return j >= j_thresh


def sample_indices(n_total: int, n_want: int, rng: random.Random) -> List[int]:
    """Sample without replacement (clamped to available)."""
    n = min(n_total, max(0, n_want))
    if n <= 0:
        return []
    return rng.sample(range(n_total), n)


def clean(s: Optional[str]) -> str:
    return html.unescape(str(s or "").strip())


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------- Prompt templates -----------------------------

def to_prompt_free_response(question: str, context: Optional[str] = None) -> str:
    question = clean(question)
    ctx = clean(context)
    lines = [f"Question: {question}"]
    if ctx:
        lines.append(f"Context: {ctx}")
    lines.append(
        "Instructions: Think step by step using biomedical knowledge. "
        "Provide 3–6 reasoning steps that reference key concepts (diseases, drugs, genes, anatomy). "
        "End with a concise final answer."
    )
    return "\n".join(lines)


def to_prompt_mcq(question: str, options: List[str]) -> str:
    question = clean(question)
    opts = [clean(o) for o in options if isinstance(o, str)]
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[: len(opts)]
    opt_str = " | ".join([f"{lab}) {opt}" for lab, opt in zip(labels, opts)])
    lines = [
        f"Question: {question}",
        f"Options: {opt_str}",
        "Instructions: Reason step by step. Eliminate wrong options with specific medical facts. "
        "Then pick the single best option, and finish with 'Final: <LETTER>' on the last line.",
    ]
    return "\n".join(lines)


# ----------------------------- Loaders -----------------------------

def load_pubmedqa(n: int, rng: random.Random) -> List[Dict]:
    out: List[Dict] = []
    tried = False
    # Labeled (smaller, higher-quality)
    try:
        dsets = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
        split = next(iter(dsets.values()))
        idx = sample_indices(len(split), n, rng)
        for i in idx:
            row = split[i]
            q = row.get("question", "")
            ctx = None
            if "context" in row and isinstance(row["context"], list):
                ctx = " ".join(map(clean, row["context"][:2]))
            elif "contexts" in row and isinstance(row["contexts"], list):
                ctx = " ".join(map(clean, row["contexts"][:2]))
            prompt = to_prompt_free_response(q, ctx)
            out.append(
                {
                    "dataset": "PubMedQA(pqa_labeled)",
                    "source_id": str(row.get("pubid", row.get("pubmed_id", i))),
                    "question": q,
                    "prompt": prompt,
                    "options": "",
                    "gold": row.get("final_decision", ""),
                    "license": "MIT (per HF card)",
                }
            )
        tried = True
        if len(out) >= n:
            return out[:n]
        n = n - len(out)
    except Exception:
        pass

    # Artificial (very large)
    try:
        dsets = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
    except Exception as e:
        if tried:
            return out
        raise e
    split = next(iter(dsets.values()))
    idx = sample_indices(len(split), n, rng)
    for i in idx:
        row = split[i]
        q = row.get("question", "")
        ctx = None
        if "context" in row and isinstance(row["context"], list):
            ctx = " ".join(map(clean, row["context"][:2]))
        elif "contexts" in row and isinstance(row["contexts"], list):
            ctx = " ".join(map(clean, row["contexts"][:2]))
        prompt = to_prompt_free_response(q, ctx)
        out.append(
            {
                "dataset": "PubMedQA(pqa_artificial)",
                "source_id": str(row.get("pubid", row.get("pubmed_id", i))),
                "question": q,
                "prompt": prompt,
                "options": "",
                "gold": row.get("final_decision", ""),
                "license": "MIT (per HF card)",
            }
        )
    return out


def load_medqa(n: int, rng: random.Random) -> List[Dict]:
    dsets = load_dataset("GBaker/MedQA-USMLE-4-options-hf")
    all_rows = []
    for split_name, ds in dsets.items():
        all_rows.extend([(split_name, i, ds[i]) for i in range(len(ds))])
    idx = sample_indices(len(all_rows), n, rng)
    out: List[Dict] = []
    for k in idx:
        split_name, i, row = all_rows[k]
        q = row.get("sent1", "")
        opts = [row.get(f"ending{j}", "") for j in range(4)]
        opts = [clean(o) for o in opts if clean(o)]
        if len(opts) < 2:
            continue
        prompt = to_prompt_mcq(q, opts)
        label = row.get("label", "")  # often 0..3
        gold_letter = _gold_to_letter(label)
        out.append(
            {
                "dataset": f"MedQA({split_name})",
                "source_id": str(row.get("id", i)),
                "question": q,
                "prompt": prompt,
                "options": " || ".join(opts),
                "gold": gold_letter,
                "license": "CC BY-SA 4.0 (per HF card)",
            }
        )
    return out


def load_medmcqa(n: int, rng: random.Random) -> List[Dict]:
    dsets = load_dataset("openlifescienceai/medmcqa")
    all_rows = []
    for split_name, ds in dsets.items():
        all_rows.extend([(split_name, i, ds[i]) for i in range(len(ds))])
    idx = sample_indices(len(all_rows), n, rng)
    out: List[Dict] = []
    for k in idx:
        split_name, i, row = all_rows[k]
        q = row.get("question", "")
        opts = [row.get("opa", ""), row.get("opb", ""), row.get("opc", ""), row.get("opd", "")]
        opts = [clean(o) for o in opts if clean(o)]
        if len(opts) < 2:
            continue
        prompt = to_prompt_mcq(q, opts)
        gold = row.get("cop", "")  # often a letter already
        gold_letter = _gold_to_letter(gold)
        out.append(
            {
                "dataset": f"MedMCQA({split_name})",
                "source_id": str(row.get("id", i)),
                "question": q,
                "prompt": prompt,
                "options": " || ".join(opts),
                "gold": gold_letter,
                "license": "Apache-2.0 (per HF card)",
            }
        )
    return out


def load_liveqa(n: int, rng: random.Random) -> List[Dict]:
    dsets = load_dataset("hyesunyun/liveqa_medical_trec2017")
    split_name = next(iter(dsets.keys()))
    ds = dsets[split_name]
    idx = sample_indices(len(ds), n, rng)
    out: List[Dict] = []
    for i in idx:
        row = ds[i]
        q = row.get("ORIGINAL_QUESTION_MESSAGE", "") or row.get("NIST_PARAPHRASE", "")
        ctx = row.get("NIST_PARAPHRASE", "")
        q_out = q + ("?" if q and not q.strip().endswith("?") else "")
        prompt = to_prompt_free_response(q_out, ctx)
        out.append(
            {
                "dataset": f"LiveQA_Medical({split_name})",
                "source_id": str(row.get("QUESTION_ID", i)),
                "question": q_out,
                "prompt": prompt,
                "options": "",
                "gold": "",
                "license": "Research use; see HF card / TREC",
            }
        )
    return out


def load_medquad_local(n: int, rng: random.Random, medquad_dir: Optional[str]) -> List[Dict]:
    out: List[Dict] = []
    if not medquad_dir:
        return out
    root = Path(medquad_dir).expanduser()
    if not root.exists():
        print(f"[WARN] MedQuAD path not found: {root}")
        return out

    files = list(root.rglob("*.xml")) + list(root.rglob("*.txt"))
    rng.shuffle(files)
    picked = 0
    for fp in files:
        if picked >= n:
            break
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        q: Optional[str] = None
        m = re.search(r"<Question[^>]*>(.*?)</Question>", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            q = clean(m.group(1))
        else:
            qm = re.search(r"(?im)^(?:Q(?:uestion)?\s*[:\-]\s*)(.+)$", text)
            if qm:
                q = clean(qm.group(1))
        if not q:
            continue
        prompt = to_prompt_free_response(q, None)
        out.append(
            {
                "dataset": "MedQuAD(local)",
                "source_id": str(fp.relative_to(root)),
                "question": q,
                "prompt": prompt,
                "options": "",
                "gold": "",
                "license": "CC BY 4.0 (per repo)",
            }
        )
        picked += 1

    return out


# ---------- MeDAL backup (optional) -----------------------------

def _iter_medal_streaming(split: str):
    if split == "full":
        for s in ("train", "validation", "test"):
            ds = load_dataset("McGill-NLP/medal", split=s, streaming=True)
            for row in ds:
                yield row
    else:
        ds = load_dataset("McGill-NLP/medal", split=split, streaming=True)
        for row in ds:
            yield row


def load_medal_backup(
    n: int,
    rng: random.Random,
    *,
    split: str = "validation",
    stream: bool = True,
    max_scan: int = 200_000,
) -> List[Dict]:
    out: List[Dict] = []
    try:
        if stream:
            rows = _iter_medal_streaming(split)
            reservoir: List[Tuple[int, Dict]] = []
            seen = 0
            for row in rows:
                seen += 1
                if seen > max_scan:
                    break
                if not isinstance(row, dict):
                    continue
                text = clean(row.get("text", ""))
                locs = row.get("location") or []
                labels = row.get("label") or []
                if not text:
                    continue
                q = "In the following text, what is the correct expansion of the medical abbreviation at the given character position(s)?"
                ctx = text + ("\nPositions: " + ",".join(str(int(x)) for x in locs) if locs else "")
                prompt = to_prompt_free_response(q, ctx)
                gold = str(labels[0]) if labels else ""
                rec = {
                    "dataset": f"MeDAL({split})",
                    "source_id": str(row.get("abstract_id", seen)),
                    "question": q,
                    "prompt": prompt,
                    "options": "",
                    "gold": gold,
                    "license": "Unknown / see HF card",
                }
                if len(reservoir) < n:
                    reservoir.append((seen, rec))
                else:
                    j = rng.randint(1, seen)
                    if j <= n:
                        reservoir[j - 1] = (seen, rec)
            out = [r for _, r in reservoir]
        else:
            dsets = load_dataset("McGill-NLP/medal")
            split_name = split if split in dsets else "validation"
            ds = dsets[split_name]
            idx = sample_indices(len(ds), n, rng)
            for i in idx:
                row = ds[i]
                text = clean(row.get("text", ""))
                locs = row.get("location") or []
                labels = row.get("label") or []
                if not text:
                    continue
                q = "In the following text, what is the correct expansion of the medical abbreviation at the given character position(s)?"
                ctx = text + ("\nPositions: " + ",".join(str(int(x)) for x in locs) if locs else "")
                prompt = to_prompt_free_response(q, ctx)
                gold = str(labels[0]) if labels else ""
                out.append(
                    {
                        "dataset": f"MeDAL({split_name})",
                        "source_id": str(row.get("abstract_id", i)),
                        "question": q,
                        "prompt": prompt,
                        "options": "",
                        "gold": gold,
                        "license": "Unknown / see HF card",
                    }
                )
        return out[:n]
    except Exception as e:
        print(f"[WARN] MeDAL backup not available: {e}")
        return []


# ----------------------------- Assembly -----------------------------

def _parse_weights(spec: Optional[str], active_keys: List[str]) -> Dict[str, float]:
    w = {k: (1.0 if k in active_keys else 0.0) for k in CORE_KEYS}
    if not spec:
        return {k: w[k] for k in active_keys}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("Bad weights spec. Use k=v, comma separated.")
        k, v = part.split("=", 1)
        k = k.strip().lower()
        if k not in CORE_KEYS:
            raise ValueError(f"Unknown weight key: {k}")
        try:
            w[k] = float(v)
        except Exception:
            raise ValueError(f"Bad weight value for {k}: {v}")
    return {k: w[k] for k in active_keys}


def _allocate_by_weights(target: int, weights: Dict[str, float]) -> Dict[str, int]:
    keys = list(weights.keys())
    if not keys:
        return {}
    total_w = sum(max(0.0, float(weights.get(k, 0.0))) for k in keys)
    if total_w <= 0:
        base = target // len(keys)
        rem = target - base * len(keys)
        out = {k: base for k in keys}
        for i, k in enumerate(keys[:rem]):
            out[k] += 1
        return out

    raw = {k: target * (max(0.0, float(weights.get(k, 0.0))) / total_w) for k in keys}
    floor = {k: int(math.floor(raw[k])) for k in keys}
    rem = target - sum(floor.values())
    fracs = sorted(((raw[k] - floor[k], k) for k in keys), reverse=True)
    for i in range(rem):
        floor[fracs[i % len(fracs)][1]] += 1
    return floor


def _quality_ok(q: str, *, min_chars: int, max_chars: int, require_qmark: bool) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    if len(q) < min_chars or len(q) > max_chars:
        return False
    if require_qmark and not q.endswith("?"):
        return False
    return True


def _dedupe(records: List[Dict], *, min_chars: int, max_chars: int, require_qmark: bool, j_thresh: float) -> List[Dict]:
    """BUGFIX: use min_chars/max_chars (not min_q_chars/max_q_chars) inside this function."""
    seen = set()
    kept: List[Dict] = []
    for r in records:
        q = r.get("question", "")
        # (fixed line below)
        if not _quality_ok(q, min_chars=min_chars, max_chars=max_chars, require_qmark=require_qmark):
            continue
        key = norm_q(q)
        if not key or key in seen:
            continue
        if kept and j_thresh > 0:
            recent = kept[-200:] if len(kept) > 200 else kept
            if any(near_dupe(q, k.get("question", ""), j_thresh=j_thresh) for k in recent):
                continue
        seen.add(key)
        r["norm_q"] = key
        kept.append(r)
    return kept


def _gold_to_letter(gold: str) -> str:
    if gold is None:
        return ""
    g = str(gold).strip()
    if g.isdigit():
        idx = int(g)
        if 0 <= idx <= 25:
            return chr(ord('A') + idx)
    g = g.upper()
    if g in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        return g
    m = re.search(r"([A-Z])", g)
    return m.group(1) if m else ""


def _shuffle_mcq_if_needed(r: Dict, rng: random.Random, do_shuffle: bool) -> Dict:
    if not do_shuffle:
        return r
    opts = [o.strip() for o in (r.get("options", "").split(" || ") if r.get("options") else []) if o.strip()]
    if len(opts) < 2:
        return r
    old = list(opts)
    rng.shuffle(opts)
    gold = _gold_to_letter(r.get("gold", ""))
    if gold:
        try:
            old_idx = ord(gold) - ord('A')
            old_correct = old[old_idx] if 0 <= old_idx < len(old) else None
            new_idx = opts.index(old_correct) if old_correct in opts else -1
            new_gold = chr(ord('A') + new_idx) if 0 <= new_idx < len(opts) else ""
        except Exception:
            new_gold = ""
    else:
        new_gold = ""
    r2 = {**r}
    r2["options"] = " || ".join(opts)
    r2["gold"] = new_gold
    r2["prompt"] = to_prompt_mcq(r.get("question", ""), opts)
    return r2


@dataclass
class AssemblyPlan:
    target: int
    per_bucket: Dict[str, int]


def assemble_prompts(
    total_n: int,
    seed: int,
    out_dir: Path,
    medquad_dir: Optional[str],
    *,
    weights: Optional[str] = None,
    min_q_chars: int = 12,
    max_q_chars: int = 420,
    require_qmark: bool = False,
    shuffle_mcq: bool = False,
    use_medal: bool = True,
    medal_split: str = "validation",
    medal_stream: bool = True,
    medal_max_scan: int = 200_000,
    force_target: bool = False,
    near_dupe_jaccard: float = 0.92,
    oversample_factor: float = 2.5,
    topup_passes: int = 12,
    sources: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    rng = random.Random(seed)
    target = to_multiple_of_4(total_n)
    if target != total_n:
        print(f"[INFO] Increased requested total from {total_n} to {target} to make it divisible by 4.")

    active_keys = list(CORE_KEYS) if not sources else [s.strip().lower() for s in sources.split(",") if s.strip()]
    active_keys = [k for k in active_keys if k in CORE_KEYS]
    if not active_keys:
        raise SystemExit("No valid sources selected. Choose any of: pubmedqa, medqa, medmcqa, liveqa")

    W = _parse_weights(weights, active_keys)
    plan = _allocate_by_weights(target, W)

    recs: Dict[str, List[Dict]] = {k: [] for k in CORE_KEYS}
    if "pubmedqa" in active_keys:
        recs["pubmedqa"] = load_pubmedqa(plan.get("pubmedqa", 0), rng)
    if "medqa" in active_keys:
        recs["medqa"] = load_medqa(plan.get("medqa", 0), rng)
    if "medmcqa" in active_keys:
        recs["medmcqa"] = load_medmcqa(plan.get("medmcqa", 0), rng)
    if "liveqa" in active_keys:
        recs["liveqa"] = load_liveqa(plan.get("liveqa", 0), rng)

    got = {k: len(recs[k]) for k in CORE_KEYS}

    shortages = [(k, plan.get(k, 0) - got.get(k, 0)) for k in active_keys if plan.get(k, 0) - got.get(k, 0) > 0]
    if shortages:
        total_short = sum(s for _, s in shortages)
        can_take = {k: W[k] for k in active_keys if plan.get(k, 0) - got.get(k, 0) <= 0 and W.get(k, 0) > 0}
        if not can_take:
            can_take = {k: 1.0 for k in ("pubmedqa", "medqa", "medmcqa") if k in active_keys}
        add_plan = _allocate_by_weights(total_short, can_take)
        for key, add in add_plan.items():
            if add <= 0:
                continue
            if key == "pubmedqa":
                more = load_pubmedqa(add, rng)
                recs["pubmedqa"].extend(more)
                got[key] = len(recs["pubmedqa"])
            elif key == "medqa":
                more = load_medqa(add, rng)
                recs["medqa"].extend(more)
                got[key] = len(recs["medqa"])
            elif key == "medmcqa":
                more = load_medmcqa(add, rng)
                recs["medmcqa"].extend(more)
                got[key] = len(recs["medmcqa"])
            elif key == "liveqa":
                more = load_liveqa(add, rng)
                recs["liveqa"].extend(more)
                got[key] = len(recs["liveqa"])

    all_recs = []
    for k in CORE_KEYS:
        all_recs.extend(recs[k])

    if shuffle_mcq:
        tmp: List[Dict] = []
        for r in all_recs:
            tmp.append(_shuffle_mcq_if_needed(r, rng, True) if r.get("options") else r)
        all_recs = tmp

    deduped = _dedupe(
        all_recs,
        min_chars=min_q_chars,
        max_chars=max_q_chars,
        require_qmark=require_qmark,
        j_thresh=near_dupe_jaccard,
    )

    def top_up_needed() -> int:
        return max(0, target - len(deduped))

    seen = {r["norm_q"] for r in deduped}

    # Optional: MedQuAD
    if top_up_needed() > 0 and medquad_dir:
        extra = load_medquad_local(top_up_needed(), rng, medquad_dir)
        for r in extra:
            key = r.get("norm_q") or norm_q(r.get("question", ""))
            if key and key not in seen and _quality_ok(r.get("question", ""), min_chars=min_q_chars, max_chars=max_q_chars, require_qmark=require_qmark):
                seen.add(key)
                r["norm_q"] = key
                deduped.append(r)
                if top_up_needed() <= 0:
                    break

    # Optional: MeDAL
    if top_up_needed() > 0 and use_medal:
        extra = load_medal_backup(top_up_needed(), rng, split=medal_split, stream=medal_stream, max_scan=medal_max_scan)
        for r in extra:
            key = r.get("norm_q") or norm_q(r.get("question", ""))
            if key and key not in seen and _quality_ok(r.get("question", ""), min_chars=min_q_chars, max_chars=max_q_chars, require_qmark=require_qmark):
                seen.add(key)
                r["norm_q"] = key
                deduped.append(r)
                if top_up_needed() <= 0:
                    break

    # FINAL: repeated PubMedQA passes
    attempts = 0
    while top_up_needed() > 0 and attempts < max(1, int(topup_passes)):
        need = top_up_needed()
        oversample = int(max(need * float(oversample_factor), need + 64))
        extra = load_pubmedqa(oversample, rng)
        added = 0
        for r in extra:
            key = r.get("norm_q") or norm_q(r.get("question", ""))
            if not key or key in seen:
                continue
            if not _quality_ok(r.get("question", ""), min_chars=min_q_chars, max_chars=max_q_chars, require_qmark=require_qmark):
                continue
            seen.add(key)
            r["norm_q"] = key
            deduped.append(r)
            added += 1
            if top_up_needed() <= 0:
                break

        # Small side nudge if needed
        if added == 0 and need > 0:
            side = []
            if "medqa" in active_keys:
                side += load_medqa(min(need, 500), rng)
            if "medmcqa" in active_keys:
                side += load_medmcqa(min(need, 500), rng)
            for r in side:
                key = r.get("norm_q") or norm_q(r.get("question", ""))
                if key and key not in seen and _quality_ok(r.get("question", ""), min_chars=min_q_chars, max_chars=max_q_chars, require_qmark=require_qmark):
                    seen.add(key)
                    r["norm_q"] = key
                    deduped.append(r)
                    if top_up_needed() <= 0:
                        break

        attempts += 1

    # Last resort: allow duplicates to force exact N
    if top_up_needed() > 0 and force_target:
        gap = top_up_needed()
        rng2 = random.Random(seed + 999)
        pool = deduped[:] or load_pubmedqa(gap, rng2)
        while gap > 0:
            deduped.append(rng2.choice(pool))
            gap -= 1

    deduped = deduped[:target]

    from collections import Counter
    cnt = Counter([(r.get("dataset") or "?").split("(")[0] for r in deduped])
    summary_by_source = {k: sum(cnt[kk] for kk in cnt if kk.lower().startswith(k)) for k in CORE_KEYS}

    short_log: List[Dict] = []
    if len(deduped) < target:
        short_log.append(
            {
                "note": "Could not reach target after repeated top-ups (no duplicates allowed). Consider --force_target.",
                "target": target,
                "final_count": len(deduped),
            }
        )

    return deduped, short_log, summary_by_source


# ----------------------------- JSON I/O -----------------------------

def _write_jsonl(iter_objs: List[Dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in iter_objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return out_path


def _write_outputs_json_only(records: List[Dict], out_dir: Path) -> Tuple[Path, Path]:
    """
    Write:
      - prompts_<N>.jsonl  (full objects with prompt_id)
      - answers_<N>.jsonl  (MCQ answer key records only)
    """
    final_n = len(records)
    prompts_jsonl_path = out_dir / f"prompts_{final_n}.jsonl"
    answers_jsonl_path = out_dir / f"answers_{final_n}.jsonl"

    # prompts jsonl (full)
    prompts_objs = []
    for i, r in enumerate(records, 1):
        obj = {**r, "prompt_id": f"p{i:04d}"}
        prompts_objs.append(obj)
    _write_jsonl(prompts_objs, prompts_jsonl_path)

    # answers jsonl (MCQ only)
    answers_objs = []
    for i, r in enumerate(records, 1):
        if r.get("options"):
            options_raw = r.get("options", "")
            # robust split for " || " or "||" with variable whitespace
            opts_list = [x.strip() for x in re.split(r"\s*\|\|\s*", options_raw) if x.strip()]
            answers_objs.append(
                {
                    "prompt_id": f"p{i:04d}",
                    "dataset": r.get("dataset", ""),
                    "source_id": r.get("source_id", ""),
                    "gold": _gold_to_letter(r.get("gold", "")),
                    "options": options_raw,
                    "options_list": opts_list,
                }
            )
    _write_jsonl(answers_objs, answers_jsonl_path)
    return prompts_jsonl_path, answers_jsonl_path


# --------- JSON split helpers (prompts only) ---------

def _chunk_ranges_for_prompts(total: int, *, split_into: int = 0, chunk_size: int = 0) -> List[Tuple[int, int]]:
    if total <= 0:
        return []
    if chunk_size and chunk_size > 0:
        split_into = (total + chunk_size - 1) // chunk_size
    split_into = max(0, int(split_into))
    if split_into <= 1:
        return [(1, total)]

    base = total // split_into
    rem = total % split_into
    out: List[Tuple[int, int]] = []
    start = 1
    for i in range(split_into):
        size = base + (1 if i < rem else 0)
        end = start + size - 1
        out.append((start, end))
        start = end + 1
    return [(s, e) for (s, e) in out if s <= e]


def _write_json_splits_only(records: List[Dict], out_dir: Path, *, split_into: int, chunk_size: int, split_subdir: str = "json_splits") -> List[Dict]:
    total = len(records)
    ranges = _chunk_ranges_for_prompts(total, split_into=split_into, chunk_size=chunk_size)
    container = out_dir / split_subdir
    ensure_dir(container)

    manifest: List[Dict] = []
    for (s, e) in ranges:
        sub = container / f"{s}-{e}"
        ensure_dir(sub)
        objs = []
        for j, r in enumerate(records[s - 1:e], s):
            objs.append({**r, "prompt_id": f"p{j:04d}"})
        pth = sub / "prompts.jsonl"
        _write_jsonl(objs, pth)
        manifest.append({"range": f"{s}-{e}", "file": str(pth)})
    return manifest


def _write_manifest(
    out_dir: Path,
    *,
    total_requested: int,
    final_count: int,
    prompts_jsonl_path: Optional[Path],
    answers_jsonl_path: Optional[Path],
    splits_manifest: List[Dict],
    summary_by_source: Dict[str, int],
    shortages: List[Dict],
    args_dict: Dict,
):
    man = {
        "total_requested": total_requested,
        "final_count": final_count,
        "outputs": {
            "prompts_jsonl_path": str(prompts_jsonl_path) if prompts_jsonl_path else None,
            "answers_jsonl_path": str(answers_jsonl_path) if answers_jsonl_path else None,
        },
        "splits": splits_manifest,
        "summary_by_source": summary_by_source,
        "shortages": shortages,
        "args": args_dict,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)


def _print_summary(records: List[Dict], short_log: List[Dict], summary_by_source: Dict[str, int]):
    total = len(records)
    print("\nSummary by source:")
    for k in ("pubmedqa", "medqa", "medmcqa", "liveqa"):
        v = int(summary_by_source.get(k, 0))
        print(f"  {k:8s}: {v:5d}")
    print(f"  {'TOTAL':8s}: {total:5d}")

    if short_log:
        print("\n[WARN] Shortage details:")
        for item in short_log:
            print("  -", json.dumps(item))


# ----------------------------- Main -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_n", type=int, required=True, help="Requested number of prompts (rounded up to /4)")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--medquad_dir", type=str, default=None, help="Optional local path to MedQuAD clone for top-ups")
    ap.add_argument("--weights", type=str, default=None, help="Weights for buckets: 'pubmedqa=1,medqa=1,medmcqa=1,liveqa=1'")
    ap.add_argument("--sources", type=str, default="pubmedqa,medqa,medmcqa,liveqa", help="Comma list of sources to include")
    ap.add_argument("--min_q_chars", type=int, default=12, help="Minimum question length filter")
    ap.add_argument("--max_q_chars", type=int, default=420, help="Maximum question length filter")
    ap.add_argument("--require_qmark", action="store_true", help="Require question to end with '?' (quality filter)")
    ap.add_argument("--shuffle_mcq", action="store_true", help="Shuffle MCQ options and remap gold labels")
    # MeDAL backup controls
    ap.add_argument("--no_medal", dest="use_medal", action="store_false", help="Disable MeDAL backup (enabled by default)")
    ap.add_argument("--medal_split", type=str, default="validation", choices=["train", "validation", "test", "full"], help="MeDAL split to mine from")
    ap.add_argument("--no_medal_stream", dest="medal_stream", action="store_false", help="Disable streaming for MeDAL backup (enabled by default)")
    ap.add_argument("--medal_max_scan", type=int, default=200_000, help="Max rows to scan from MeDAL when streaming backup")
    ap.set_defaults(use_medal=True, medal_stream=True)

    # Robust fill controls
    ap.add_argument("--force_target", action="store_true", help="As a last resort, allow duplicates to reach exact target")
    ap.add_argument("--near_dupe_jaccard", type=float, default=0.92, help="Jaccard threshold for near-duplicate pruning; <=0 disables")
    ap.add_argument("--oversample_factor", type=float, default=2.5, help="Oversample factor in each PubMedQA top-up pass")
    ap.add_argument("--topup_passes", type=int, default=12, help="Max number of PubMedQA top-up passes")

    # JSON splits
    ap.add_argument("--prompts_split_into", type=int, default=0, help="Split prompts JSONL into this many even chunks")
    ap.add_argument("--prompts_chunk_size", type=int, default=0, help="Alternative: fixed chunk size for prompts JSONL")
    ap.add_argument("--prompts_split_subdir", type=str, default="json_splits", help="Subfolder under --out_dir for JSON chunks")

    # Optional: skip top-level pack (write only splits + manifest)
    ap.add_argument("--no_full_pack", action="store_true", help="Do not write full prompts_<N>.jsonl / answers_<N>.jsonl")
    return ap


def main():
    args = build_arg_parser().parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    records, short_log, summary_by_source = assemble_prompts(
        total_n=args.total_n,
        seed=args.seed,
        out_dir=out_dir,
        medquad_dir=args.medquad_dir,
        weights=args.weights,
        min_q_chars=args.min_q_chars,
        max_q_chars=args.max_q_chars,
        require_qmark=args.require_qmark,
        shuffle_mcq=args.shuffle_mcq,
        use_medal=args.use_medal,
        medal_split=args.medal_split,
        medal_stream=args.medal_stream,
        medal_max_scan=args.medal_max_scan,
        force_target=args.force_target,
        near_dupe_jaccard=args.near_dupe_jaccard,
        oversample_factor=args.oversample_factor,
        topup_passes=args.topup_passes,
        sources=args.sources,
    )

    prompts_jsonl_path = answers_jsonl_path = None
    if not args.no_full_pack:
        prompts_jsonl_path, answers_jsonl_path = _write_outputs_json_only(records, out_dir)
        print(f"Wrote {len(records)} prompts to:       {prompts_jsonl_path}")
        print(f"Wrote MCQ answers to:              {answers_jsonl_path}")

    splits_manifest: List[Dict] = []
    if (args.prompts_split_into and args.prompts_split_into > 1) or (args.prompts_chunk_size and args.prompts_chunk_size > 0):
        splits_manifest = _write_json_splits_only(
            records,
            out_dir,
            split_into=int(args.prompts_split_into or 0),
            chunk_size=int(args.prompts_chunk_size or 0),
            split_subdir=(args.prompts_split_subdir or "json_splits").strip() or "json_splits",
        )
        print("\nJSON prompt splits:")
        for m in splits_manifest:
            print(f"  {m['range']}: {m['file']}")

    _write_manifest(
        out_dir,
        total_requested=args.total_n,
        final_count=len(records),
        prompts_jsonl_path=prompts_jsonl_path,
        answers_jsonl_path=answers_jsonl_path,
        splits_manifest=splits_manifest,
        summary_by_source=summary_by_source,
        shortages=short_log,
        args_dict=vars(args),
    )

    _print_summary(records, short_log, summary_by_source)


if __name__ == "__main__":
    main()
