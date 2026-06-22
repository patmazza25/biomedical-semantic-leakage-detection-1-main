# SICD Camera-Ready — Run Guide

Regenerate all results on the 20-case corpus with harmonized prompts, fixed oscillation, and the
temperature ablation {0.4, 0.7, 0.9}. Run the **two model notebooks** (the paper's two models).

## Folders & notebooks (renamed for clarity)
- `SICD Haiku Version/sicd_haiku.ipynb`        → Claude Haiku 4.5  (primary)
- `SICD GPT-4o-mini Version/sicd_gpt4omini.ipynb` → GPT-4o-mini  (generalization)
- `sicd_gemma.ipynb` (in either folder)        → local Gemma  (supplementary; not in the paper)

## 0. Keys you'll need
- **OpenRouter API key** (both Haiku + GPT-4o-mini).
- **UMLS/UTS API key** (concept extraction). Sign up: https://uts.nlm.nih.gov/uts/signup-login.
- (Gemma only) Hugging Face token + a Colab A100.

You no longer paste keys into the notebook — the keys cell **prompts you** (hidden input) when you run it.

## Run in Colab (recommended)
1. **Zip the folder** on your PC (e.g. right-click `SICD Haiku Version` → Send to → Compressed folder).
   Zip your **current edited** folder so Colab gets the 20 cases + fixes.
2. In Colab: **File → Upload notebook** → pick the notebook from that folder
   (`sicd_haiku.ipynb` or `sicd_gpt4omini.ipynb`).
3. Run the **first code cell** ("fetch experiment files"). It shows a **Choose Files** button →
   pick your **zip** → it extracts `sicd_cases.py`, `sicd_scorers.py`, `utils/`.
   (This cell auto-skips if the files are already present.)
4. Run the **setup cell** → it prints `Environment configured…`.
5. Run the **keys cell** → paste your `OPENROUTER_API_KEY`, then `UMLS_API_KEY` at the prompts.
   The imports cell should then print `UMLS configured: True`.
6. **Runtime → Run all.** Generation prints a live counter, e.g. `[T=0.4] 7/80  STEMI [hard_interference] -> 9 steps`.

Generation = 20 cases × 4 levels × 3 temps = **240 chains** per model (cached per temperature in
`data/sicd_cache_<model>_t04|t07|t09.json`; reruns skip cached temps). UMLS extraction is the slow
part — let it finish in one sitting.

## Run in VSCode (local) instead
Works too (no GPU needed for the two API notebooks). Open the repo folder, pick a Python 3 kernel,
`pip install requests openai python-dotenv numpy scipy matplotlib pandas`. The fetch + setup cells
auto-skip locally because the files are already on disk; just make sure the notebook's working
directory is its own folder (so `import sicd_cases` resolves).

## (Optional) Gemma
Open `sicd_gemma.ipynb`, needs an A100 + HF token, and **first replace** the placeholder model id
`google/gemma-4-31B-it` (cell 5 `MODEL` + the warmup cell) with a real Gemma id.

## What to send back (per model)
From each run's `data/` folder:
- `sicd_ablation_haiku.csv` and `sicd_ablation_gpt4omini.csv` (the Table 1 numbers).
- The **evaluation cell** console output (per-temp Spearman tables + SDR-by-temperature + per-case SDR).
- Figures: `sicd_trajectories_*.png`, `sicd_sdr_ablation_*.png`.

I'll then fill the `«FILL FROM RERUN»` placeholders in `CAMERA_READY_EDITS.md`, finalize Table 1 and
the abstract numbers, and report whether the corrected oscillation separates surrender vs. resistance.

## Troubleshooting
- `UMLS configured: False` → re-run the keys cell and enter the UMLS key.
- Generation looks stuck → watch the `N/total` counter; each chain is one API call.
- Extraction slow / occasional empty concepts → UMLS rate-limiting; rerun, or run one temperature at a
  time by editing `TEMPS` in the generation cell.
- Spearman `nan` for a signal → that signal was constant across chains; not an error.
- Cheaper/faster → set `TEMPS = [0.7]` (headline run only, no ablation columns).
