# SICD Camera-Ready — Run Guide

Goal: regenerate all results on the 20-case corpus, harmonized prompts, fixed oscillation, and the
temperature ablation {0.4, 0.7, 0.9}. Run the **two `sicd_test.ipynb` notebooks** (the paper's two
models). The Gemma `SICD_OpenSource_Test.ipynb` is optional/supplementary.

## 0. Prerequisites (keys)
- **OpenRouter API key** — used for both Claude Haiku 4.5 and GPT-4o-mini.
- **UMLS API key** — a UTS account key (https://uts.nlm.nih.gov/uts/signup-login), used for concept
  extraction. Required; without it concept extraction returns nothing.
- (Gemma only) a **Hugging Face token** + a Colab **A100** runtime, and a real Gemma model id.

## 1. Open in Colab
These notebooks expect Colab (the setup cell scans `/content`). For each run:
1. Upload the whole folder to Colab (or mount Drive) so `sicd_cases.py`, `sicd_scorers.py`, and
   `utils/` sit next to the notebook.
2. Open the notebook (e.g. `SICD Haiku Version/sicd_test.ipynb`).

## 2. Put your keys in cell 2
Replace the `REDACTED` placeholders:
```python
os.environ['OPENROUTER_API_KEY'] = '...'   # your key
os.environ['UMLS_API_KEY']       = '...'   # your UTS key
```
Run cells 0–3. Cell 3 prints `UMLS configured: True` — confirm that before continuing.

## 3. (Recommended) Smoke test first
Before the full ablation, sanity-check the pipeline cheaply:
- In cell 5, temporarily set `TEMPS = [0.7]`.
- Optionally shrink the corpus for the test: after cell 4, run `CASES = CASES[:2]` in a scratch cell.
- Run cells 5–7. Confirm generation works, `UMLS configured: True`, and cell 7 prints a Spearman
  table with non-NaN values (and Oscillation is no longer all 0.000).
- Then restore `TEMPS = [0.4, 0.7, 0.9]` and the full `CASES` and do the real run.

## 4. Run the Claude Haiku 4.5 notebook (primary)
- File: `SICD Haiku Version/sicd_test.ipynb`. Run all cells top to bottom.
- Generation: 20 cases × 4 levels × 3 temps = **240 chains** (cached per temperature in
  `data/sicd_cache_haiku_t04|t07|t09.json` — reruns skip what's cached).
- Concept extraction is the slow part (UMLS REST). Let it finish in one sitting (extraction results
  are held in memory, not on disk — if the kernel dies mid-extraction you re-extract, though
  generation stays cached).
- Outputs (in `data/`): `sicd_ablation_haiku.csv`, `sicd_trajectories_haiku.png`,
  `sicd_sdr_ablation_haiku.png`, plus the printed per-temp Spearman tables and the per-case SDR table.

## 5. Run the GPT-4o-mini notebook (generalization)
- File: `SICD_Test/sicd_test.ipynb`. Same steps as above.
- Outputs: `data/sicd_ablation_gpt4omini.csv`, `sicd_trajectories_gpt4omini.png`,
  `sicd_sdr_ablation_gpt4omini.png`, + printed tables.

## 6. (Optional) Gemma — supplementary
- File: either `SICD_OpenSource_Test.ipynb`. Needs an **A100** + HF token.
- **First replace the model id** `google/gemma-4-31B-it` (cell 5 `MODEL` and the cell-4 warmup) with a
  real Gemma id, e.g. a current `google/gemma-*-it` release.
- Outputs: `data/sicd_ablation_gemma.csv` + figures.

## 7. What to send back
From each run's `data/` folder:
- `sicd_ablation_haiku.csv` and `sicd_ablation_gpt4omini.csv` (the core numbers for Table 1).
- The console output of **cell 7** for each notebook (per-temp Spearman tables, the SDR-by-temperature
  block, and the per-case SDR table).
- The figures: `sicd_trajectories_*.png` and `sicd_sdr_ablation_*.png`.
- (Optional, if anything looks off) the per-temp cache files `sicd_cache_*_t*.json`.

I'll then fill the `«FILL FROM RERUN»` placeholders in `CAMERA_READY_EDITS.md`, finalize Table 1 and
the abstract numbers, and tell you whether the corrected oscillation separates surrender vs. resistance.

## Troubleshooting
- `UMLS configured: False` → key not set or invalid; re-check cell 2.
- Extraction very slow / intermittent empty concepts → UMLS rate-limiting; rerun, or run one
  temperature at a time by editing `TEMPS`.
- Spearman shows `nan` for a signal → that signal was constant across chains (can happen for a weak
  signal on a small set); not an error.
- Want it cheaper/faster → reduce `TEMPS` (e.g. just `[0.7]`); you still get the headline run, just no
  ablation columns.
