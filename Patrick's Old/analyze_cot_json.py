# analyze_cot_json.py
# Biomedical CoT JSON Analyzer — v4.3
# Robust JSON loading (objects/arrays, comments, trailing commas, concatenated blobs),
# per-file fault isolation (bad files skipped + logged), and robust label derivation.
'''
python analyze_cot_json.py \
  --root "reports/claude-3-5-haiku-latest/data" \
  --out "analysis_out" \
  --workers 8 \
  --excel \
  --argmax-probs

'''
import argparse
import concurrent.futures as cf
import glob
import io
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- CLI ---------------------------

def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Analyze Biomedical CoT JSON reports")
    ap.add_argument("--root", default="reports/claude-3-5-sonnet-latest/data",
                    help="Directory with JSON reports. If pointing to 'reports/', scans */data/*.json.")
    ap.add_argument("--glob", nargs="*", default=None,
                    help="Optional glob(s) for JSON files; overrides --root.")
    ap.add_argument("--out", default="analysis_out_v4", help="Output directory.")
    ap.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4)),
                    help="Parallel workers.")
    ap.add_argument("--excel", action="store_true",
                    help="Also write an Excel workbook with key tables (requires openpyxl).")
    ap.add_argument("--topn-examples", type=int, default=60,
                    help="Top-N contradiction examples to save.")
    ap.add_argument("--argmax-probs", action="store_true",
                    help="If set, derive final labels by argmax(probs) when missing or requested.")
    ap.add_argument("--contra-thresh", type=float, default=None,
                    help="Optional threshold: if prob_contradiction >= T => label 'contradiction'.")
    ap.add_argument("--entail-thresh", type=float, default=None,
                    help="Optional threshold: if prob_entailment >= T => label 'entailment'.")
    ap.add_argument("--neutral-fallback", action="store_true",
                    help="If thresholds used and neither hits, label 'neutral' (otherwise argmax).")
    ap.add_argument("--skip-bad", action="store_true",
                    help="Skip files that cannot be parsed instead of aborting the run.")
    return ap

# --------------------------- JSON loading ---------------------------

OBJ_RE = re.compile(r'\{(?:[^{}"]|"[^"\\]*(?:\\.[^"\\]*)*")*\}', re.DOTALL)
ARR_RE = re.compile(r'\[(?:[^\[\]"]|"[^"\\]*(?:\\.[^"\\]*)*")*\]', re.DOTALL)

def _json_try(s: str) -> Optional[Dict[str, Any]]:
    try:
        x = json.loads(s)
        # Allow object root or a single-object array root
        if isinstance(x, dict):
            return x
        if isinstance(x, list):
            # Prefer last dict if multiple dicts in list, else first dict
            dicts = [e for e in x if isinstance(e, dict)]
            if dicts:
                return dicts[-1]
    except Exception:
        pass
    return None

def _strip_comments_and_ctrl(text: str) -> str:
    # Remove //... and /* ... */ comments
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Remove non-printable control chars (except \t \n \r)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)
    return text

def _fix_trailing_commas(text: str) -> str:
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text

def _balance_slice(text: str, start_idx: int) -> Optional[str]:
    """
    Return the smallest balanced JSON block ({} or []) starting at start_idx,
    respecting quotes and escapes. If none, return None.
    """
    open_char = text[start_idx]
    close_char = '}' if open_char == '{' else ']'
    depth = 0
    in_str = False
    esc = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx:i+1]
    return None

def _extract_last_balanced(text: str) -> Optional[str]:
    # Search from the end for '{' or '[' and try to balance
    for idx in range(len(text) - 1, -1, -1):
        ch = text[idx]
        if ch in '{[':
            block = _balance_slice(text, idx)
            if block:
                return block
    return None

def _load_json_robust(path: str) -> Dict[str, Any]:
    """
    Robust loader:
      1) direct json.loads
      2) strip comments + trailing commas, then json.loads
      3) last balanced {...} or [...] block (balanced by scanner)
      4) last regex-matched {...} or [...] block
      5) JSONL scan: first line that parses to dict or array with dicts
    """
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig", errors="replace")
    except Exception:
        text = raw.decode("utf-8", errors="replace")

    # 1) direct
    obj = _json_try(text)
    if obj is not None:
        return obj

    # 2) cleaned
    cleaned = _fix_trailing_commas(_strip_comments_and_ctrl(text))
    obj = _json_try(cleaned)
    if obj is not None:
        return obj

    # 3) balanced scanner (from end)
    block = _extract_last_balanced(cleaned)
    if block:
        obj = _json_try(block)
        if obj is not None:
            return obj

    # 4) regex (object or array), prefer last match that parses
    for m in reversed(list(OBJ_RE.finditer(cleaned))):
        obj = _json_try(m.group(0))
        if obj is not None:
            return obj
    for m in reversed(list(ARR_RE.finditer(cleaned))):
        obj = _json_try(m.group(0))
        if obj is not None:
            return obj

    # 5) JSONL/line-wise
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = _json_try(line)
        if obj is not None:
            return obj

    raise ValueError(f"Could not parse JSON: {path}")

# --------------------------- Utilities ---------------------------

def _safe_list(x, default=None):
    return x if isinstance(x, list) else (default or [])

def _safe_dict(x, default=None):
    return x if isinstance(x, dict) else (default or {})

def _norm_sem_types(st) -> List[str]:
    out: List[str] = []
    if isinstance(st, list):
        for x in st:
            if isinstance(x, dict):
                v = x.get("name") or x.get("Name") or x.get("tui") or x.get("TUI")
                if v:
                    out.append(str(v))
            elif isinstance(x, str):
                out.append(x)
    return out

def _cui_set(step_concepts: List[Dict[str, Any]]) -> set:
    return {c.get("cui") for c in step_concepts if c.get("cui")}

def _canon_set(step_concepts: List[Dict[str, Any]]) -> set:
    s = {(c.get("canonical") or c.get("text") or "").strip().lower() for c in step_concepts}
    s.discard("")
    return s

def _topic_overlap(umls_a: List[Dict[str, Any]], umls_b: List[Dict[str, Any]]) -> bool:
    a, b = _cui_set(umls_a), _cui_set(umls_b)
    if a & b: return True
    an, bn = _canon_set(umls_a), _canon_set(umls_b)
    return bool(an & bn)

def _ensure_out(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "charts"), exist_ok=True)

def _plot_bar_counts(counts: Dict[str, int], title: str, out_png: str) -> None:
    if not counts:
        return
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def _plot_series(xs: List[int], ys: List[float], title: str, xlabel: str, ylabel: str, out_png: str) -> None:
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def _derive_model_name(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    try:
        data_idx = len(parts) - 1 - parts[::-1].index("data")
        model = parts[data_idx - 1] if data_idx - 1 >= 0 else ""
        return model or ""
    except ValueError:
        return Path(path).parent.name

def _iter_json_files(root: Optional[str], globs: List[str]) -> List[str]:
    files: List[str] = []
    if globs:
        for pat in globs:
            files.extend(glob.glob(pat))
    elif root:
        if os.path.isdir(root):
            files.extend(glob.glob(os.path.join(root, "*.json")))
        else:
            files.extend(glob.glob(os.path.join(root, "*", "data", "*.json")))
    return sorted(set(files))

def _two_prop_z(x1, n1, x2, n2):
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if denom == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / denom
    Phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    p_two = 2 * (1 - Phi) if z >= 0 else 2 * Phi
    return z, p_two

# --------------------------- Label normalization ---------------------------

def _norm_label(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip().lower()
    if t in {"contra", "contradict", "contradictn", "contradictions"}:
        t = "contradiction"
    if t in {"entails", "entail"}:
        t = "entailment"
    if t in {"neut", "none"}:
        t = "neutral"
    return t

def _best_label_from_probs(e: Dict[str, Any],
                           contra_thresh: Optional[float],
                           entail_thresh: Optional[float],
                           neutral_fallback: bool) -> Optional[str]:
    probs = _safe_dict(e.get("probs"))
    if "contradict" in probs and "contradiction" not in probs:
        probs["contradiction"] = probs.get("contradict")
    if not probs:
        return None

    pe = float(probs.get("entailment") or 0.0)
    pn = float(probs.get("neutral") or 0.0)
    pc = float(probs.get("contradiction") or 0.0)

    if contra_thresh is not None and pc >= float(contra_thresh):
        return "contradiction"
    if entail_thresh is not None and pe >= float(entail_thresh):
        return "entailment"
    if (contra_thresh is not None or entail_thresh is not None):
        if neutral_fallback:
            return "neutral"
    # argmax
    m = max(pe, pn, pc)
    if m == pe:
        return "entailment"
    if m == pc:
        return "contradiction"
    return "neutral"

# --------------------------- Per-file analysis ---------------------------

def analyze_one(args: Tuple[int, str],
                contra_thresh: Optional[float],
                entail_thresh: Optional[float],
                use_argmax: bool,
                neutral_fallback: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    report_id, path = args
    data = _load_json_robust(path)

    question = (data.get("question") or "").strip()
    steps: List[str] = _safe_list(data.get("steps"))
    entailments: List[Dict[str, Any]] = _safe_list(data.get("entailments"))
    umls: List[List[Dict[str, Any]]] = _safe_list(data.get("umls"))
    generated_at = data.get("generated_at")
    model_name = _derive_model_name(path)

    n_steps = len(steps)
    n_pairs = max(0, n_steps - 1)

    # ---------- Concepts long ----------
    concepts_rows: List[Dict[str, Any]] = []
    total_concepts = 0
    total_valid = 0
    source_counter = Counter()
    stype_counter = Counter()

    for si, step_concepts in enumerate(umls):
        for rec in _safe_list(step_concepts):
            sem_types = _norm_sem_types(rec.get("semantic_types"))
            kb_sources = rec.get("kb_sources") or []
            valid = bool(rec.get("valid"))
            cui = rec.get("cui")
            concepts_rows.append({
                "report_id": report_id,
                "model": model_name,
                "file": os.path.basename(path),
                "step_idx": si,
                "text": rec.get("text"),
                "cui": cui,
                "canonical": rec.get("canonical"),
                "valid": valid,
                "kb_sources": "|".join(kb_sources) if isinstance(kb_sources, list) else "",
                "semantic_types": "|".join(sem_types),
                "confidence": (_safe_dict(rec.get("scores")).get("confidence")),
            })
            total_concepts += 1
            total_valid += int(valid)
            for s in kb_sources or []:
                source_counter[str(s)] += 1
            for t in sem_types or []:
                stype_counter[str(t)] += 1

    concept_valid_rate = (total_valid / total_concepts) if total_concepts else np.nan
    unique_cuis = len({r["cui"] for r in concepts_rows if r["cui"]})

    # ---------- Pairs long ----------
    pairs_rows: List[Dict[str, Any]] = []
    guard_rows: List[Dict[str, Any]] = []

    label_counter = Counter()
    final_label_counter = Counter()
    guards_counter = Counter()

    avg_probs = {"entailment": [], "neutral": [], "contradiction": []}
    depth_contra = defaultdict(lambda: {"contradictions": 0, "total": 0})
    contra_no_topic_overlap = 0
    contra_total = 0

    for i in range(n_pairs):
        e = entailments[i] if i < len(entailments) else {}
        probs = dict(_safe_dict(e.get("probs")))
        if "contradict" in probs and "contradiction" not in probs:
            probs["contradiction"] = probs.get("contradict")
        for k in ("entailment", "neutral", "contradiction"):
            if k in probs:
                try:
                    avg_probs[k].append(float(probs[k]))
                except Exception:
                    pass

        label = _norm_label(e.get("label")) or "neutral"
        final_label = _norm_label(e.get("final_label")) or label
        if use_argmax or final_label not in {"entailment","neutral","contradiction"}:
            derived = _best_label_from_probs(e, contra_thresh, entail_thresh, neutral_fallback)
            if derived:
                final_label = derived

        reason = (e.get("reason") or "").strip()
        guards = _safe_list(e.get("guards"))
        triple_prev = bool(e.get("triple_prev"))
        triple_next = bool(e.get("triple_next"))

        label_counter[label] += 1
        final_label_counter[final_label] += 1
        for g in guards:
            guards_counter[g] += 1
            guard_rows.append({
                "report_id": report_id,
                "model": model_name,
                "file": os.path.basename(path),
                "pair_idx": i,
                "guard": g
            })

        prev_concepts = _safe_list(umls[i]) if i < len(umls) else []
        next_concepts = _safe_list(umls[i+1]) if (i+1) < len(umls) else []
        prev_total = len(prev_concepts)
        next_total = len(next_concepts)
        prev_valid = sum(1 for c in prev_concepts if bool(c.get("valid")))
        next_valid = sum(1 for c in next_concepts if bool(c.get("valid")))
        prev_valid_rate = (prev_valid / prev_total) if prev_total else np.nan
        next_valid_rate = (next_valid / next_total) if next_total else np.nan

        try:
            topic_olap = _topic_overlap(prev_concepts, next_concepts)
        except Exception:
            topic_olap = False

        depth_contra[i]["total"] += 1
        if final_label == "contradiction":
            depth_contra[i]["contradictions"] += 1
            contra_total += 1
            if not topic_olap:
                contra_no_topic_overlap += 1

        pairs_rows.append({
            "report_id": report_id,
            "model": model_name,
            "file": os.path.basename(path),
            "pair_idx": i,
            "label": label,
            "final_label": final_label,
            "prob_entailment": probs.get("entailment"),
            "prob_neutral": probs.get("neutral"),
            "prob_contradiction": probs.get("contradiction"),
            "reason": reason,
            "guards_pipe": "|".join(guards),
            "has_triple_prev": bool(triple_prev),
            "has_triple_next": bool(triple_next),
            "topic_overlap": bool(topic_olap),
            "prev_concepts_total": prev_total,
            "prev_concepts_valid": prev_valid,
            "prev_valid_rate": prev_valid_rate,
            "next_concepts_total": next_total,
            "next_concepts_valid": next_valid,
            "next_valid_rate": next_valid_rate,
            "step_text_prev": steps[i] if i < len(steps) else None,
            "step_text_next": steps[i+1] if (i + 1) < len(steps) else None,
        })

    avg_ent = float(np.nanmean(avg_probs["entailment"])) if avg_probs["entailment"] else np.nan
    avg_neu = float(np.nanmean(avg_probs["neutral"])) if avg_probs["neutral"] else np.nan
    avg_con = float(np.nanmean(avg_probs["contradiction"])) if avg_probs["contradiction"] else np.nan

    row_summary = {
        "report_id": report_id,
        "model": model_name,
        "file": os.path.basename(path),
        "question": question,
        "n_steps": n_steps,
        "n_pairs": n_pairs,
        "concepts_total": total_concepts,
        "concepts_valid": total_valid,
        "concept_valid_rate": (total_valid / total_concepts) if total_concepts else np.nan,
        "unique_cuis": unique_cuis,
        "avg_prob_entailment": avg_ent,
        "avg_prob_neutral": avg_neu,
        "avg_prob_contradiction": avg_con,
        "label_entailment": label_counter.get("entailment", 0),
        "label_neutral": label_counter.get("neutral", 0),
        "label_contradiction": label_counter.get("contradiction", 0),
        "final_entailment": final_label_counter.get("entailment", 0),
        "final_neutral": final_label_counter.get("neutral", 0),
        "final_contradiction": final_label_counter.get("contradiction", 0),
        "guards_total": int(sum(guards_counter.values())),
        "pairs_with_triple_prev": int(sum(1 for r in pairs_rows if r["has_triple_prev"])),
        "pairs_with_triple_next": int(sum(1 for r in pairs_rows if r["has_triple_next"])),
        "contradictions_no_topic_overlap": contra_no_topic_overlap,
        "contradictions_total": contra_total,
        "generated_at": generated_at,
    }

    df_concepts = pd.DataFrame(concepts_rows, columns=[
        "report_id","model","file","step_idx","text","cui","canonical",
        "valid","kb_sources","semantic_types","confidence"
    ])
    df_pairs = pd.DataFrame(pairs_rows, columns=[
        "report_id","model","file","pair_idx","label","final_label",
        "prob_entailment","prob_neutral","prob_contradiction",
        "reason","guards_pipe","has_triple_prev","has_triple_next",
        "topic_overlap",
        "prev_concepts_total","prev_concepts_valid","prev_valid_rate",
        "next_concepts_total","next_concepts_valid","next_valid_rate",
        "step_text_prev","step_text_next"
    ])
    df_guards = pd.DataFrame(guard_rows, columns=["report_id","model","file","pair_idx","guard"])

    sources = [{"report_id": report_id, "model": model_name, "kb_source": k, "count": v} for k, v in source_counter.items()]
    stypes = [{"report_id": report_id, "model": model_name, "semantic_type": k, "count": v} for k, v in stype_counter.items()]
    df_sources = pd.DataFrame(sources, columns=["report_id","model","kb_source","count"])
    df_stypes = pd.DataFrame(stypes, columns=["report_id","model","semantic_type","count"])

    depth_stats = {d: (v["contradictions"] / v["total"]) if v["total"] else np.nan for d, v in depth_contra.items()}
    extras = {"summary": row_summary, "depth": depth_stats}
    return df_pairs, df_concepts, df_guards, (df_sources, df_stypes), extras

# --------------------------- HTML builders ---------------------------

def build_html(out_dir: str, topline: Dict[str, Any], charts: List[str], insights: List[str]) -> None:
    html_path = os.path.join(out_dir, "analytics_summary.html")
    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'><title>CoT Analytics Summary</title>")
    parts.append("<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;margin:24px;line-height:1.5} h1{font-size:1.6rem} .kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:12px 0} .card{border:1px solid #e5e7eb;border-radius:10px;padding:12px} img{max-width:100%;height:auto;border:1px solid #e5e7eb;border-radius:10px;padding:6px;background:#fafafa} ul{margin:0 0 0 18px}</style>")
    parts.append("</head><body>")
    parts.append("<h1>Biomedical CoT — Analytics Summary</h1>")
    parts.append("<div class='kpi'>")
    for k, v in topline.items():
        parts.append(f"<div class='card'><div style='font-size:0.85rem;color:#6b7280'>{k}</div><div style='font-size:1.2rem;font-weight:600'>{v}</div></div>")
    parts.append("</div>")
    if insights:
        parts.append("<h2>Actionable insights</h2>")
        parts.append("<div class='card'><ul>")
        for li in insights[:8]:
            parts.append(f"<li>{li}</li>")
        parts.append("</ul></div>")
    parts.append("<h2>Charts</h2>")
    for rel in charts:
        fn = os.path.basename(rel)
        parts.append(f"<div class='card'><div style='font-weight:600;margin-bottom:6px'>{fn}</div><img src='charts/{fn}' alt='{fn}'></div>")
    parts.append("</body></html>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

def build_examples_html(out_dir: str, df_pairs: pd.DataFrame, topn: int = 40) -> None:
    html_path = os.path.join(out_dir, "contradiction_examples.html")
    if df_pairs.empty or (df_pairs["final_label"]!="contradiction").all():
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<html><body><p>No contradiction pairs found.</p></body></html>")
        return

    df = df_pairs.copy()
    for col in ("prob_contradiction","prob_entailment"):
        if col not in df or df[col].isnull().all():
            df[col] = np.nan
    df["score"] = df["prob_contradiction"].fillna(0) - 0.25*df["prob_entailment"].fillna(0)
    df = df[df["final_label"]=="contradiction"].sort_values("score", ascending=False).head(topn)

    rows_html = []
    for _, r in df.iterrows():
        rows_html.append(f"""
        <tr>
          <td><code>{r.get('model','')}</code></td>
          <td><code>{r.get('file','')}</code></td>
          <td>{int(r.get('pair_idx',-1))}</td>
          <td style="white-space:pre-wrap">{(r.get('step_text_prev') or '')}</td>
          <td style="white-space:pre-wrap">{(r.get('step_text_next') or '')}</td>
          <td>{r.get('prob_entailment')}</td>
          <td>{r.get('prob_contradiction')}</td>
          <td style="white-space:pre-wrap">{(r.get('reason') or '')}</td>
          <td>{(r.get('guards_pipe') or '').replace('|','<br>')}</td>
        </tr>
        """)

    html = f"""
    <!doctype html><html><head><meta charset="utf-8">
      <title>Contradiction Examples</title>
      <style>body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;margin:24px;line-height:1.5}}
      table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #e5e7eb;padding:8px;text-align:left}} th{{background:#fafafa}}
      </style>
    </head><body>
      <h1>Most Confident Contradictions</h1>
      <table>
       <thead><tr>
         <th>Model</th><th>File</th><th>Pair</th>
         <th>Step i</th><th>Step i+1</th>
         <th>P(ent)</th><th>P(contra)</th>
         <th>Reason</th><th>Guards</th>
       </tr></thead>
       <tbody>
         {''.join(rows_html)}
       </tbody>
      </table>
    </body></html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

# --------------------------- Main ---------------------------

def main():
    args = build_cli().parse_args()

    files = _iter_json_files(args.root, args.glob or [])
    if not files:
        print("No JSON files found. Try: --root reports/ or --glob 'reports/*/data/*.json'")
        return

    out = args.out
    _ensure_out(out)
    skipped_log = os.path.join(out, "skipped_files.txt")
    skipped = []

    tasks = list(enumerate(files))
    all_pairs = []
    all_concepts = []
    all_guards = []
    all_sources = []
    all_stypes = []
    summaries = []
    depth_rows = []

    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        fut2path = {}
        for t in tasks:
            fut = ex.submit(
                analyze_one, t, args.contra_thresh, args.entail_thresh,
                args.argmax_probs, args.neutral_fallback
            )
            fut2path[fut] = t[1]

        for fut in cf.as_completed(fut2path):
            path = fut2path[fut]
            try:
                df_pairs, df_concepts, df_guards, (df_sources, df_stypes), extras = fut.result()
            except Exception as e:
                if args.skip_bad:
                    skipped.append(f"{path}\t{repr(e)}")
                    continue
                else:
                    raise

            if df_pairs is not None:
                all_pairs.append(df_pairs)
            if df_concepts is not None:
                all_concepts.append(df_concepts)
            if df_guards is not None and not df_guards.empty:
                all_guards.append(df_guards)
            if df_sources is not None and not df_sources.empty:
                all_sources.append(df_sources)
            if df_stypes is not None and not df_stypes.empty:
                all_stypes.append(df_stypes)
            if extras and extras.get("summary"):
                summaries.append(extras["summary"])
            if extras and extras.get("depth"):
                for d, rate in extras["depth"].items():
                    depth_rows.append({"depth": int(d), "contradiction_rate": rate})

    if skipped:
        with open(skipped_log, "w", encoding="utf-8") as f:
            f.write("\n".join(skipped))
        print(f"[WARN] Skipped {len(skipped)} files. See: {skipped_log}")

    if not summaries:
        print("No valid JSON reports parsed.")
        return

    # Aggregate frames
    df_summary = pd.DataFrame(summaries).sort_values(["model","file"]).reset_index(drop=True)
    df_pairs_all = pd.concat([df for df in all_pairs if df is not None], ignore_index=True) if all_pairs else pd.DataFrame()
    df_concepts_all = pd.concat([df for df in all_concepts if df is not None], ignore_index=True) if all_concepts else pd.DataFrame()
    df_guards_all = pd.concat([df for df in all_guards if df is not None], ignore_index=True) if all_guards else pd.DataFrame(columns=["report_id","model","file","pair_idx","guard"])
    df_sources_all = pd.concat([df for df in all_sources if df is not None], ignore_index=True) if all_sources else pd.DataFrame()
    df_stypes_all = pd.concat([df for df in all_stypes if df is not None], ignore_index=True) if all_stypes else pd.DataFrame()
    df_depth = pd.DataFrame(depth_rows)

    # Save base CSVs
    df_summary.to_csv(os.path.join(out, "reports_summary.csv"), index=False)
    df_pairs_all.to_csv(os.path.join(out, "pairs_long.csv"), index=False)
    df_concepts_all.to_csv(os.path.join(out, "concepts_long.csv"), index=False)
    df_guards_all.to_csv(os.path.join(out, "guards_long.csv"), index=False)
    if not df_sources_all.empty:
        df_sources_all.to_csv(os.path.join(out, "concept_sources_long.csv"), index=False)
    if not df_stypes_all.empty:
        df_stypes_all.to_csv(os.path.join(out, "semantic_types_long.csv"), index=False)

    # Derived aggregates
    depth_agg = pd.DataFrame(columns=["depth","contradiction_rate"])
    if not df_depth.empty:
        depth_agg = df_depth.groupby("depth", as_index=False)["contradiction_rate"].mean().sort_values("depth")
    depth_agg.to_csv(os.path.join(out, "leakage_by_depth.csv"), index=False)

    # Guard lift analysis
    insights = []
    if not df_pairs_all.empty:
        global_n = df_pairs_all.shape[0]
        global_x = int((df_pairs_all["final_label"] == "contradiction").sum())
        global_rate = global_x / global_n if global_n else float("nan")

        if not df_guards_all.empty:
            gf = df_guards_all.merge(
                df_pairs_all[["report_id","model","file","pair_idx","final_label"]],
                on=["report_id","model","file","pair_idx"],
                how="left"
            )
            out_rows = []
            for g, grp in gf.groupby("guard"):
                n = grp.shape[0]
                x = int((grp["final_label"] == "contradiction").sum())
                rate = x / n if n else float("nan")
                z, p = _two_prop_z(x, n, global_x, global_n)
                out_rows.append({"guard": g, "n_pairs": n, "n_contra": x, "rate": rate,
                                 "global_rate": global_rate, "lift": (rate - global_rate) if (not math.isnan(rate) and not math.isnan(global_rate)) else float("nan"),
                                 "z": z, "p_value": p})
            pd.DataFrame(out_rows).sort_values(["rate","n_pairs"], ascending=[False, False]).to_csv(os.path.join(out, "guard_lift.csv"), index=False)

        # Triple coverage
        tc = []
        for col in ["has_triple_prev","has_triple_next"]:
            n = df_pairs_all.shape[0]
            x = int((df_pairs_all["final_label"]=="contradiction").sum())
            base_rate = x / n if n else float("nan")
            for flag in [True, False]:
                sub = df_pairs_all[df_pairs_all[col]==flag]
                n1 = sub.shape[0]
                x1 = int((sub["final_label"]=="contradiction").sum())
                r1 = x1 / n1 if n1 else float("nan")
                z, p = _two_prop_z(x1, n1, x, n)
                tc.append({"feature": col, "flag": flag, "n_pairs": n1, "n_contra": x1, "rate": r1, "global_rate": base_rate, "z": z, "p_value": p})
        pd.DataFrame(tc).sort_values(["feature","flag"], ascending=[True, False]).to_csv(os.path.join(out, "triple_coverage_effects.csv"), index=False)

        # Topic overlap
        if "topic_overlap" in df_pairs_all:
            rows = []
            for flag in [True, False]:
                s = df_pairs_all[df_pairs_all["topic_overlap"]==flag]
                n1 = s.shape[0]
                x1 = int((s["final_label"]=="contradiction").sum())
                r1 = x1 / n1 if n1 else float("nan")
                z, p = _two_prop_z(x1, n1, global_x, global_n)
                rows.append({"topic_overlap": flag, "n_pairs": n1, "n_contra": x1, "rate": r1, "global_rate": global_rate, "z": z, "p_value": p})
            pd.DataFrame(rows).to_csv(os.path.join(out, "topic_overlap_effects.csv"), index=False)

        # Validity vs contradiction
        if {"prev_valid_rate","next_valid_rate"}.issubset(df_pairs_all.columns):
            def bucket(v):
                if pd.isna(v): return "na"
                if v >= 0.8:  return "high"
                if v >= 0.2:  return "mid"
                return "low"
            tmp = df_pairs_all.copy()
            tmp["prev_bucket"] = tmp["prev_valid_rate"].apply(bucket)
            tmp["next_bucket"] = tmp["next_valid_rate"].apply(bucket)

            rows = []
            x = int((df_pairs_all["final_label"]=="contradiction").sum())
            n = df_pairs_all.shape[0]
            base = x / n if n else float("nan")

            for (pb, nb), s in tmp.groupby(["prev_bucket","next_bucket"]):
                n1 = s.shape[0]
                x1 = int((s["final_label"]=="contradiction").sum())
                r1 = x1 / n1 if n1 else float("nan")
                z, p = _two_prop_z(x1, n1, x, n)
                rows.append({"prev_bucket": pb, "next_bucket": nb, "n_pairs": n1, "n_contra": x1, "rate": r1, "global_rate": base, "z": z, "p_value": p})
            pd.DataFrame(rows).sort_values(["prev_bucket","next_bucket"]).to_csv(os.path.join(out, "validity_vs_contradiction.csv"), index=False)

    # Charts
    charts = []
    if not df_pairs_all.empty:
        counts = df_pairs_all["final_label"].value_counts().to_dict()
        out_png = os.path.join(out, "charts", "final_label_distribution.png")
        _plot_bar_counts(counts, "Final label distribution (all pairs)", out_png)
        charts.append(out_png)

    if not depth_agg.empty:
        xs = depth_agg["depth"].astype(int).tolist()
        ys = depth_agg["contradiction_rate"].astype(float).tolist()
        out_png = os.path.join(out, "charts", "leakage_by_depth.png")
        _plot_series(xs, ys, "Contradiction rate by reasoning depth", "Depth (pair index)", "Avg contradiction rate", out_png)
        charts.append(out_png)

    if not df_sources_all.empty:
        top_src = (df_sources_all.groupby("kb_source", as_index=False)["count"].sum()
                   .sort_values("count", ascending=False).head(15))
        counts = dict(zip(top_src["kb_source"], top_src["count"]))
        out_png = os.path.join(out, "charts", "top_kb_sources.png")
        _plot_bar_counts(counts, "Top knowledge-base sources (concept yield)", out_png)
        charts.append(out_png)

    topline = {
        "Files analyzed": len(files),
        "Models observed": ", ".join(sorted(set(df_summary["model"].dropna()))) if "model" in df_summary else "",
        "Total pairs": int(df_pairs_all.shape[0]) if not df_pairs_all.empty else 0,
        "Total concept mentions": int(df_concepts_all.shape[0]) if not df_concepts_all.empty else 0,
        "Mean concept validity rate": f"{pd.to_numeric(df_summary['concept_valid_rate'], errors='coerce').mean():.3f}",
        "Mean P(entailment)": f"{pd.to_numeric(df_summary['avg_prob_entailment'], errors='coerce').mean():.3f}",
        "Mean P(contradiction)": f"{pd.to_numeric(df_summary['avg_prob_contradiction'], errors='coerce').mean():.3f}",
        "Pairs with triples (prev/next)": f"{int(df_summary.get('pairs_with_triple_prev', pd.Series(dtype=int)).sum())}/{int(df_summary.get('pairs_with_triple_next', pd.Series(dtype=int)).sum())}",
    }
    build_html(out, topline, charts, [])

    build_examples_html(out, df_pairs_all, topn=int(args.topn_examples))

    if args.excel:
        xlsx_path = os.path.join(out, "cot_analytics.xlsx")
        try:
            with pd.ExcelWriter(xlsx_path) as xw:
                df_summary.to_excel(xw, "reports_summary", index=False)
                df_pairs_all.head(100000).to_excel(xw, "pairs_long_head", index=False)
                df_concepts_all.head(100000).to_excel(xw, "concepts_long_head", index=False)
                if not df_guards_all.empty:
                    df_guards_all.to_excel(xw, "guards_long", index=False)
                for name in ["leakage_by_depth.csv","guard_lift.csv","triple_coverage_effects.csv","topic_overlap_effects.csv","validity_vs_contradiction.csv"]:
                    p = os.path.join(out, name)
                    if os.path.exists(p):
                        pd.read_csv(p).to_excel(xw, Path(name).stem, index=False)
            print("Excel written:", xlsx_path)
        except ModuleNotFoundError:
            print("[WARN] openpyxl not installed; skipping Excel export. Install: pip install openpyxl")
        except Exception as e:
            print("[WARN] Excel export failed:", e)

    print("Wrote base CSVs and HTML into:", out)


if __name__ == "__main__":
    main()
