# utils/reporter.py
from __future__ import annotations
import os, json, csv, time
from typing import List, Dict, Any, Optional
from config import REPORTS_DIR

def save_report(
    question: str,
    steps: List[str],
    entailments: List[Dict[str, Any]],
    umls: Any,
    settings: Optional[Dict[str, Any]] = None,
    *,
    model_dir: Optional[str] = None,
) -> tuple[str, str, str]:
    """
    Writes three artifacts and returns their **relative** paths from REPORTS_DIR:
      - HTML is written by the caller; we return the name to use
      - data/<name>.json
      - data/<name>.csv
    Where <model_dir>/data is created if model_dir is provided.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.join(REPORTS_DIR, model_dir) if model_dir else REPORTS_DIR
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    base_html = f"report-{ts}.html"
    json_name = base_html.replace(".html", ".json")
    csv_name  = base_html.replace(".html", ".csv")

    payload = {
        "question": question,
        "steps": steps,
        "entailments": entailments,
        "umls": umls,
        "settings": settings or {},
        "generated_at": ts,
    }
    with open(os.path.join(data_dir, json_name), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(os.path.join(data_dir, csv_name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i","j","label","final_label","entail","neutral","contradict","reason"])
        for e in entailments:
            p = e.get("probs", {})
            i, j = e.get("step_pair", ["",""])
            w.writerow([
                i, j,
                e.get("label",""),
                e.get("final_label",""),
                p.get("entailment",""),
                p.get("neutral",""),
                p.get("contradiction",""),
                e.get("reason","")
            ])

    rel_html = f"{model_dir}/{base_html}" if model_dir else base_html
    rel_json = f"{model_dir}/data/{json_name}" if model_dir else f"data/{json_name}"
    rel_csv  = f"{model_dir}/data/{csv_name}"  if model_dir else f"data/{csv_name}"
    return rel_html, rel_json, rel_csv
