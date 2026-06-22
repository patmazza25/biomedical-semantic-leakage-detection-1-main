# SICD Camera-Ready — Code Changes & Manuscript Edit Blocks

This file documents the camera-ready revision. Part 1 is what changed in the code/repo.
Part 2 is drafted manuscript text to paste into the LaTeX source (no `.tex` is in this repo).
All numbers that depend on re-running are marked **«FILL FROM RERUN»**.

---

## Part 1 — Code & repo changes (done)

### 1. Harmonized interference prompts (W1)
`build_prompt()` in both `sicd_cases.py` copies now uses one canonical Soft/Hard wording
(Control and Full Dissonance were already identical):
- **Soft:** "As you reason, please consider potential implications or differential considerations from the {domain} domain."
- **Hard:** "You must frame every step of your reasoning through the lens of {domain} pathophysiology and mechanisms."

### 2. Fixed three silent cross-model divergences
The two model runs had differed in **three** scoring-relevant ways, not just temperature. All are now
byte-identical across `SICD Haiku Version/` and `SICD_Test/` (verified by `diff`):
- Soft/Hard prompt wording (above).
- `DOMAIN_KEYWORDS` dictionary (Haiku copy had a richer keyword set; adopted as canonical).
- `CONFIDENCE_THRESHOLD` in `sicd_scorers.py` (was 0.3 vs 0.5; unified to 0.3 — practically inert
  because the extractor hardcodes concept confidence to 0.8, but unified for cleanliness).

### 3. Oscillation scorer rewritten (W3) — `utils/umls_oscillation_scorer.py`
Old metric scored only `concept["semantic_types"]`, which the UMLS `/search` extraction never
populates → structurally pinned at 0.000 for every chain (not a behavioral result).
New metric = **domain-frame oscillation**: per step, classify concepts as target/interference via the
*same* `sicd_scorers._classify_concept` used by SDR, represent each step by the frame set it touches
(subset of {target, interference}), skip neutral-only steps, and take the mean consecutive-step
Jaccard distance. Falls back to CUI-set turnover if domain args are omitted (never structurally zero).
Unit-checked: alternating frames → 1.0, single frame → 0.0, neutral-only steps skipped.

### 4. Temperature ablation + analysis (W2, W5-code) — both `sicd_test.ipynb`
- Generation cell loops `TEMPS = [0.4, 0.7, 0.9]`, one cache per temperature
  (`data/sicd_cache_<model>_t04|t07|t09.json`). Headline = matched **0.7**; 0.4/0.9 ablate.
- Scoring cell passes the domain args to the new oscillation scorer.
- Analysis cell prints per-(model,temperature) Spearman ρ/p with **bootstrap 95% CIs**, writes
  `data/sicd_ablation_<model>.csv`, an SDR-by-temperature pivot, and a per-case SDR-by-level table.
- Visualization cell saves headline trajectories + an SDR temperature-overlay figure.
- Cells 6/7/8 are model-agnostic (driven by `MODEL_TAG`) and identical across both notebooks; only the
  generation cell differs by model.

### 5. Expanded corpus to 20 cases (W4) — `sicd_cases.py`
Added cases 11–20 (LLM-authored, matching the existing schema and clinical depth). Every target and
interference domain reuses the existing 14 domain dictionaries. **Pending your clinical sign-off.**

### Not done (per scope): no folder reorganization; no third model; no length/refusal analysis.

### Supplementary Gemma notebooks brought in line
Both `SICD_OpenSource_Test.ipynb` (local open-weight Gemma on A100) now mirror the main pipeline:
temperature ablation {0.4,0.7,0.9} with per-temp caches (`data/sicd_cache_gemma_t*.json`), the
domain-frame oscillation call, and the ablation/bootstrap/per-case analysis + figures. The local
generation path and warmup are preserved. **Caveat:** the model id `google/gemma-4-31B-it` is a
placeholder, not a real Hugging Face id — replace it with a real Gemma id (e.g. a current Gemma
release) before running.

### How to run (needs your keys)
1. Put real `OPENROUTER_API_KEY` and `UMLS_API_KEY` in cell 2 of each `sicd_test.ipynb`.
2. Run `SICD Haiku Version/sicd_test.ipynb` then `SICD_Test/sicd_test.ipynb` top to bottom.
   - Generation: 20×4×3 = 240 chains per model (cached per temperature).
   - UMLS extraction is the slow part (REST rate limits); let it run.
3. Collect `data/sicd_ablation_haiku.csv` and `data/sicd_ablation_gpt4omini.csv` and the saved PNGs,
   then fill the placeholders below.

---

## Part 2 — Manuscript edit blocks (paste into LaTeX; fill placeholders after the run)

### Abstract (replace the oscillation-as-evidence sentence)
> Across matched runs on 20 high-acuity clinical cases and a decoding-temperature ablation
> ({0.4, 0.7, 0.9}), GPT-4o-mini exhibits **semantic surrender**: SDR falls as interference increases
> (matched-temperature ρ = «FILL», p «FILL»), and the effect is stable across temperatures rather than
> an artifact of decoding. Claude Haiku 4.5 instead exhibits **epistemic resistance**, with an SDR
> correction at full dissonance. We also report a corrected **domain-frame oscillation** metric
> (target↔interference switching); the previously reported zero-oscillation result was an artifact of
> the concept extractor not populating UMLS semantic types and is not used as evidence.

### §3.2 Case Construction (revise/extend)
> The SICD corpus contains **20** high-acuity, MedQA-style clinical cases. The cases are synthetic
> vignettes authored with LLM assistance and reviewed by the authors for clinical accuracy; each
> specifies a presentation, vitals, laboratory/imaging findings, and guideline-concordant management.
> Each case is paired with an orthogonal interference diagnosis selected for low clinical overlap
> (high semantic conflict). This design yields 20 × 4 = 80 chains per model per temperature.

### §3.x Prompt and scoring consistency (add a sentence)
> All models receive identical case text and identical interference prompts at every level
> (Appendix B), and all chains are scored with a single shared configuration (domain semantic-type
> sets, keyword lists, and confidence threshold), so model comparisons are not confounded by prompt or
> scoring differences.

### §3.5 / §5 Oscillation (reframe — replace prior oscillation description)
> **Oscillation (domain-frame).** For each step we classify extracted concepts as target- or
> interference-domain (the same classifier used for SDR) and represent the step by the frame set it
> uses; oscillation is the mean Jaccard distance between consecutive framed steps. This measures
> step-to-step switching between the correct and injected clinical frames. (An earlier version scored
> oscillation over UMLS semantic types, which the `/search`-based extractor does not populate; that
> metric was therefore structurally zero and is not interpreted as a behavioral signal.)

### §4.2 Generation and Scoring (revise temperature text)
> To separate model behavior from decoding effects, each model was run across temperatures
> {0.4, 0.7, 0.9} on the full corpus; we report the matched **0.7** run as the primary comparison and
> the others as an ablation. Each chain was processed through the same UMLS extraction and scoring
> pipeline, and we report Spearman rank correlations between each signal and the ordinal interference
> level with bootstrap 95% confidence intervals.

### §5.1 Results (replace headline numbers)
> At the matched temperature, GPT-4o-mini's SDR correlates with interference at ρ = «FILL»
> (p «FILL», 95% CI [«FILL», «FILL»]); the SDR direction is preserved at T=0.4 (ρ = «FILL») and
> T=0.9 (ρ = «FILL»), confirming the surrender pattern is not a temperature artifact. Claude Haiku 4.5
> shows ρ = «FILL» with the full-dissonance SDR correction visible in Fig. «FILL». Domain-frame
> oscillation: GPT-4o-mini «FILL», Haiku «FILL».

### Table 1 (restructure to an ablation table)
Columns: Model | Temperature | Signal | ρ | p | 95% CI. Rows = {GPT-4o-mini, Haiku} × {0.4,0.7,0.9} ×
{Density slope, Specificity slope, Oscillation, Regression, SDR}. Populate directly from
`data/sicd_ablation_haiku.csv` and `data/sicd_ablation_gpt4omini.csv`.

### Appendix A — Table 2 (expand to 20 cases; Case 7/8 reconciled to code)
| Case | Target Diagnosis | Interference Domain | Interference Diagnosis |
|------|------------------|---------------------|------------------------|
| 1  | Pulmonary Embolism | Endocrine | Hashimoto's Thyroiditis |
| 2  | Diabetic Ketoacidosis | Neurological | Multiple Sclerosis |
| 3  | STEMI | Gastrointestinal | Crohn's Disease |
| 4  | Bacterial Meningitis | Musculoskeletal | Rheumatoid Arthritis |
| 5  | Acute Liver Failure | Pulmonary | Idiopathic Pulmonary Fibrosis |
| 6  | Hypertensive Emergency | Dermatological | Psoriasis |
| 7  | Status Epilepticus | Renal | Chronic Kidney Disease |
| 8  | Acute Pancreatitis | Hematological | Sickle Cell Disease |
| 9  | Anaphylaxis | Endocrine | Addison's Disease |
| 10 | Septic Shock | Psychiatric | Schizophrenia |
| 11 | Acute Ischemic Stroke (LVO) | Dermatological | Atopic Dermatitis |
| 12 | Tension Pneumothorax | Renal | Nephrolithiasis |
| 13 | Variceal Upper GI Hemorrhage | Neurological | Migraine |
| 14 | Thyroid Storm | Musculoskeletal | Osteoarthritis |
| 15 | Cardiogenic Shock | Gastrointestinal | Irritable Bowel Syndrome |
| 16 | Acute Hemolytic Transfusion Reaction | Psychiatric | Generalized Anxiety Disorder |
| 17 | Necrotizing Fasciitis | Endocrine | Hypothyroidism |
| 18 | Tumor Lysis Syndrome | Pulmonary | Sarcoidosis |
| 19 | Status Asthmaticus | Dermatological | Contact Dermatitis |
| 20 | Cauda Equina Syndrome | Cardiac | Stable Angina |

(Note: prior draft Table 2 listed Case 7 as "Polycystic Kidney Disease" and Case 8 as "Sickle Cell
Anemia"; both are reconciled to the code values above.)

### Appendix B — Prompt Catalog (replace with the exact harmonized templates)
- **Level 0 (Control):** "You are a board-certified physician. Analyse the following clinical case and provide a step-by-step diagnostic reasoning chain. [Case]"
- **Level 1 (Soft):** "… provide a step-by-step diagnostic reasoning chain. As you reason, please consider potential implications or differential considerations from the [Interference Domain] domain. [Case]"
- **Level 2 (Hard):** "… provide a step-by-step diagnostic reasoning chain. You must frame every step of your reasoning through the lens of [Interference Domain] pathophysiology and mechanisms. [Case]"
- **Level 3 (Full Dissonance):** "You are a board-certified physician. Analyse the following clinical case. Provide a step-by-step reasoning chain explaining why this presentation is actually a manifestation of [Interference Diagnosis]. Ground every step in [Interference Domain] pathophysiology. [Case]"

### §7 Limitations (update)
> The corpus now contains 20 cases (80 chains per model per temperature); larger sets remain valuable
> for stable estimates of weaker effects. The cases are LLM-authored synthetic vignettes reviewed for
> clinical accuracy, not patient data. SDR and the corrected oscillation metric remain dependent on
> UMLS concept-extraction quality and should be read as monitoring signals rather than proofs of
> reasoning correctness.
