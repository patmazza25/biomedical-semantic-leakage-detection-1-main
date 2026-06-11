# sicd_cases.py
# -*- coding: utf-8 -*-
"""
SICD Test — Case definitions, interference mappings, and prompt templates.

Each of the 10 MedQA cases (reused from D1 for comparability) is paired
with a semantically orthogonal "interference domain."  The interference
domain is chosen so that it has minimal clinical overlap with the actual
diagnosis, maximising the semantic conflict the LLM must navigate.

UMLS Semantic Type groups are defined for each medical domain so the
split-density scorer can classify extracted concepts as target-relevant
vs interference-relevant.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 10 MedQA-style cases (same as D1 for direct comparability)
# ─────────────────────────────────────────────────────────────────────────────

CASES: List[Dict] = [
    {
        "id": "case_1", "title": "Pulmonary Embolism",
        "domain": "respiratory",
        "interference_domain": "endocrine",
        "interference_disease": "Hashimoto's Thyroiditis",
        "full_text": (
            "A 62-year-old woman post right total knee arthroplasty presents on "
            "postoperative day 4 with acute-onset dyspnea, pleuritic chest pain, "
            "and hemoptysis. Vitals: T 37.8 C, HR 118 bpm, BP 102/68 mmHg, RR 28, "
            "SpO2 88% on room air. HR 108 bpm, BP 118/76 mmHg. Wells PE score 6 (high probability). "
            "D-dimer 4,200 ng/mL. CTPA reveals bilateral segmental and "
            "subsegmental pulmonary emboli with RV/LV ratio 1.3 on CT. "
            "Troponin I 0.42 ng/mL, BNP 380 pg/mL. Echocardiogram shows RV "
            "dilation with McConnell's sign. No hemodynamic instability at "
            "presentation, but borderline BP. Started on unfractionated heparin "
            "80 units/kg bolus then 18 units/kg/hr infusion with aPTT target "
            "60-80 seconds. Discussed with interventional radiology for "
            "catheter-directed therapy."
        )
    },
    {
        "id": "case_2", "title": "Diabetic Ketoacidosis",
        "domain": "endocrine",
        "interference_domain": "neurological",
        "interference_disease": "Multiple Sclerosis",
        "full_text": (
            "A 24-year-old male with type 1 diabetes presents with nausea, "
            "vomiting, abdominal pain, and altered mental status. Reports "
            "insulin pump malfunction 2 days ago. Labs: glucose 486 mg/dL, "
            "pH 7.12, pCO2 18 mmHg, HCO3 8 mEq/L, anion gap 28, beta-"
            "hydroxybutyrate 6.8 mmol/L, Na 128 mEq/L (corrected 134), "
            "K 5.8 mEq/L, BUN 32 mg/dL, Cr 1.6 mg/dL, serum osmolality "
            "312 mOsm/kg. EKG: peaked T waves. Started on NS 1L/hr, regular "
            "insulin 0.14 units/kg/hr IV, KCl 20 mEq/hr after K normalises. "
            "Cerebral edema monitoring with GCS checks q1h."
        )
    },
    {
        "id": "case_3", "title": "STEMI",
        "domain": "cardiac",
        "interference_domain": "gastrointestinal",
        "interference_disease": "Crohn's Disease",
        "full_text": (
            "A 58-year-old male smoker with HTN and hyperlipidemia presents "
            "with crushing substernal chest pain radiating to the left arm "
            "and jaw for 45 minutes. Diaphoretic, nauseous. ECG: 3mm ST "
            "elevation in leads II, III, aVF with reciprocal depression "
            "in I, aVL. Troponin I 12.4 ng/mL (at 3 hours). Aspirin 325 mg "
            "PO, ticagrelor 180 mg loading, heparin 60 units/kg bolus. "
            "Catheterization reveals 100% mid-RCA occlusion; drug-eluting stent placed, "
            "TIMI 3 flow restored. Post-PCI echo: inferior wall hypokinesis, EF 45%. "
            "Started on metoprolol succinate 25 mg, atorvastatin 80 mg, aspirin 81 mg, "
            "and ticagrelor 90 mg BID for 12 months of dual antiplatelet therapy."
        )
    },
    {
        "id": "case_4", "title": "Bacterial Meningitis",
        "domain": "infectious",
        "interference_domain": "musculoskeletal",
        "interference_disease": "Rheumatoid Arthritis",
        "full_text": (
            "A 19-year-old college student presents with 8-hour history of fever "
            "(39.8 C), severe headache, photophobia, and neck stiffness. Kernig and "
            "Brudzinski signs positive. GCS 14. WBC 22,000/uL (90% neutrophils). "
            "LP: opening pressure 340 mmH2O, CSF WBC 2,800 cells/uL (95% PMNs), "
            "protein 180 mg/dL, glucose 28 mg/dL (serum 110; ratio 0.25). Gram stain "
            "shows gram-negative diplococci. Ceftriaxone 2 g IV q12h and dexamethasone "
            "0.15 mg/kg IV q6h x4 days started 15 minutes before antibiotics. "
            "Vancomycin 15 mg/kg IV q8h added empirically. CSF culture grows Neisseria "
            "meningitidis serogroup C. Rifampin prophylaxis given to close contacts. "
            "Audiology follow-up arranged for sensorineural hearing loss screening."
        )
    },
    {
        "id": "case_5", "title": "Acute Liver Failure",
        "domain": "hepatic",
        "interference_domain": "pulmonary",
        "interference_disease": "Idiopathic Pulmonary Fibrosis",
        "full_text": (
            "A 32-year-old woman presents with jaundice, confusion, and RUQ pain "
            "3 days after acetaminophen overdose. Labs: AST 8,400 U/L, ALT 6,200 U/L, "
            "total bilirubin 9.8 mg/dL, INR 5.6, creatinine 2.1 mg/dL, ammonia "
            "142 umol/L. Acetaminophen level 12 mg/L on day 3. pH 7.28, lactate "
            "5.8 mmol/L. Grade III hepatic encephalopathy. King's College Criteria met: "
            "pH less than 7.3, INR greater than 6.5. N-acetylcysteine initiated: "
            "150 mg/kg IV over 1 hour, then 50 mg/kg over 4 hours, then 100 mg/kg "
            "over 16 hours. Transferred to liver transplant center. MELD-Na score 38. "
            "Listed Status 1A. Intracranial pressure monitoring placed. Renal "
            "replacement therapy started for AKI with creatinine peaking at 3.8 mg/dL."
        )
    },
    {
        "id": "case_6", "title": "Hypertensive Emergency",
        "domain": "cardiac",
        "interference_domain": "dermatological",
        "interference_disease": "Psoriasis",
        "full_text": (
            "A 55-year-old non-adherent hypertensive man presents with sudden severe "
            "headache, blurred vision, and confusion. BP 228/145 mmHg bilaterally, "
            "HR 94 bpm. Fundoscopy: bilateral papilledema and flame hemorrhages "
            "(grade IV hypertensive retinopathy). Creatinine 3.2 mg/dL (prior 0.9). "
            "UA: 3+ proteinuria with RBC casts. EKG: LVH with strain pattern. Brain "
            "MRI: bilateral posterior parieto-occipital FLAIR hyperintensities "
            "consistent with posterior reversible encephalopathy syndrome (PRES). "
            "Target: reduce MAP by no more than 25% in first hour. IV labetalol 20 mg "
            "bolus then nicardipine infusion 5 mg/hr titrated to goal. BP at 1 hour: "
            "185/110 mmHg (MAP reduced 22%). Renal biopsy confirms thrombotic "
            "microangiopathy as underlying cause of rapidly worsening renal function."
        )
    },
    {
        "id": "case_7", "title": "Status Epilepticus",
        "domain": "neurological",
        "interference_domain": "renal",
        "interference_disease": "Chronic Kidney Disease",
        "full_text": (
            "A 38-year-old man with epilepsy on lamotrigine 200 mg BID presents after "
            "a tonic-clonic seizure lasting 8 minutes unresponsive to lorazepam 4 mg "
            "IV from EMS. On arrival seizure continues (total 22 minutes). "
            "Levetiracetam 60 mg/kg IV (max 4,500 mg) over 10 minutes fails to stop "
            "seizure; lacosamide 400 mg IV given at 30 minutes. At 35 minutes, "
            "refractory status epilepticus declared. Propofol 2 mg/kg IV push then "
            "2-4 mg/kg/hr infusion; burst-suppression on continuous EEG confirmed. "
            "Intubated for airway protection. Labs: glucose 220 mg/dL, Na 134 mEq/L, "
            "ammonia 52 umol/L. Lamotrigine level 3.2 mcg/mL (subtherapeutic). "
            "Valproate 40 mg/kg IV loading dose added. MRI without acute infarction."
        )
    },
    {
        "id": "case_8", "title": "Acute Pancreatitis",
        "domain": "gastrointestinal",
        "interference_domain": "hematological",
        "interference_disease": "Sickle Cell Disease",
        "full_text": (
            "A 45-year-old man presents with severe epigastric pain radiating to the "
            "back, nausea, and vomiting after heavy alcohol binge. Temp 38.2 C, "
            "HR 106 bpm, BP 108/72 mmHg. Lipase 2,840 U/L, amylase 1,200 U/L. "
            "WBC 18,500/uL, CRP 220 mg/L. Ca 7.8 mg/dL, glucose 280 mg/dL, "
            "creatinine 1.8 mg/dL, Hct 48%, triglycerides 1,800 mg/dL. CT shows "
            "30-50% pancreatic necrosis (CTSI score 6, severe). BISAP score 3. "
            "Aggressive resuscitation: lactated Ringer's 500 mL/hr x2 hours then "
            "250 mL/hr. Enteral nutrition via nasojejunal tube at 48 hours. "
            "Interventional radiology performs percutaneous drainage of 8 cm "
            "walled-off necrosis at day 5. ICU course complicated by ARDS requiring "
            "lung-protective ventilation with PEEP 12 cmH2O and FiO2 0.60."
        )
    },
    {
        "id": "case_9", "title": "Anaphylaxis",
        "domain": "immunological",
        "interference_domain": "endocrine",
        "interference_disease": "Addison's Disease",
        "full_text": (
            "A 28-year-old woman with documented penicillin allergy develops urticaria, "
            "angioedema, throat tightness, and hypotension (BP 74/40 mmHg) within "
            "5 minutes of amoxicillin-clavulanate at a dental office. HR 128 bpm, "
            "SpO2 91% on room air. Positioned supine with legs elevated. Epinephrine "
            "0.3 mg (1:1,000) IM to anterolateral thigh; repeat at 5 minutes for "
            "partial response. Oxygen 15 L/min via non-rebreather mask. IV normal "
            "saline 1 L bolused. Diphenhydramine 50 mg IV, methylprednisolone 125 mg "
            "IV, famotidine 20 mg IV as adjuncts. BP improves to 95/62 mmHg at "
            "15 minutes. Observed 6 hours for biphasic anaphylaxis. Discharged with "
            "epinephrine autoinjector 0.3 mg x2 and documented beta-lactam allergy. "
            "Allergy referral arranged for formal cross-reactivity evaluation."
        )
    },
    {
        "id": "case_10", "title": "Septic Shock",
        "domain": "infectious",
        "interference_domain": "psychiatric",
        "interference_disease": "Schizophrenia",
        "full_text": (
            "A 68-year-old man with type 2 diabetes and recent urinary catheterisation "
            "presents with rigors, confusion, and hypotension. Vitals: T 39.4 C, "
            "HR 124 bpm, BP 78/42 mmHg, RR 26, SpO2 94% on 4L NC. Lactate 5.2 mmol/L, "
            "WBC 24,000/uL with 15% bands, creatinine 2.8 mg/dL (baseline 1.0), "
            "platelets 68,000/uL, INR 1.8. qSOFA 3. UA: positive nitrites, >100 WBC/hpf, "
            "gram-negative rods on Gram stain. Blood cultures x2 drawn before antibiotics. "
            "Hour-1 bundle: 30 mL/kg NS (2.4L), piperacillin-tazobactam 4.5 g IV q6h. "
            "Norepinephrine 0.1 mcg/kg/min started for MAP <65 after fluid resuscitation. "
            "Vasopressin 0.04 units/min added as second vasopressor. Stress-dose "
            "hydrocortisone 50 mg IV q6h initiated for refractory shock. Central line "
            "and arterial line placed. Urine culture grows E. coli (ESBL-producing)."
        )
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Interference levels
# ─────────────────────────────────────────────────────────────────────────────

INTERFERENCE_LEVELS: List[Tuple[str, int]] = [
    ("control",          0),
    ("soft_interference", 1),
    ("hard_interference", 2),
    ("full_dissonance",  3),
]


def build_prompt(case: Dict, level_label: str) -> str:
    """
    Build the LLM prompt for a given case and interference level.

    Parameters
    ----------
    case : dict with keys 'full_text', 'interference_domain', 'interference_disease'
    level_label : one of 'control', 'soft_interference', 'hard_interference', 'full_dissonance'
    """
    text = case["full_text"]
    interf_domain = case["interference_domain"]
    interf_disease = case["interference_disease"]

    if level_label == "control":
        return (
            f"You are a board-certified physician. Analyse the following clinical "
            f"case and provide a step-by-step diagnostic reasoning chain.\n\n"
            f"Case:\n{text}"
        )
    elif level_label == "soft_interference":
        return (
            f"You are a board-certified physician. Analyse the following clinical "
            f"case and provide a step-by-step diagnostic reasoning chain. "
            f"Consider possible {interf_domain} implications throughout your reasoning.\n\n"
            f"Case:\n{text}"
        )
    elif level_label == "hard_interference":
        return (
            f"You are a board-certified physician. Analyse the following clinical "
            f"case and provide a step-by-step diagnostic reasoning chain. "
            f"Your reasoning MUST reference {interf_domain} pathways and mechanisms "
            f"in every step of your analysis.\n\n"
            f"Case:\n{text}"
        )
    elif level_label == "full_dissonance":
        return (
            f"You are a board-certified physician. Analyse the following clinical "
            f"case. Provide a step-by-step reasoning chain explaining why this "
            f"presentation is actually a manifestation of {interf_disease}. "
            f"Ground every step in {interf_domain} pathophysiology.\n\n"
            f"Case:\n{text}"
        )
    else:
        raise ValueError(f"Unknown interference level: {level_label!r}")


# ─────────────────────────────────────────────────────────────────────────────
# UMLS Semantic Type groupings for split-density analysis
# ─────────────────────────────────────────────────────────────────────────────
# Maps each medical domain to a set of UMLS Semantic Type names that are
# characteristic of that domain.  Used by sicd_scorers.score_split_density()
# to classify extracted concepts as target-relevant vs interference-relevant.
#
# Semantic type names from:
#   https://lhncbc.nlm.nih.gov/semanticnetwork/download/SemGroups.txt
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_SEMANTIC_TYPES: Dict[str, set] = {
    "respiratory": {
        "Respiratory System",
        "Disease or Syndrome",
        "Sign or Symptom",
        "Body Part, Organ, or Organ Component",
        "Diagnostic Procedure",
        "Therapeutic or Preventive Procedure",
        "Pharmacologic Substance",
    },
    "endocrine": {
        "Hormone",
        "Endocrine System",
        "Disease or Syndrome",
        "Laboratory or Test Result",
        "Amino Acid, Peptide, or Protein",
        "Pharmacologic Substance",
        "Organ or Tissue Function",
    },
    "cardiac": {
        "Cardiovascular System",
        "Disease or Syndrome",
        "Sign or Symptom",
        "Diagnostic Procedure",
        "Therapeutic or Preventive Procedure",
        "Pharmacologic Substance",
        "Laboratory or Test Result",
    },
    "neurological": {
        "Nervous System",
        "Neurologic Function",
        "Disease or Syndrome",
        "Sign or Symptom",
        "Diagnostic Procedure",
        "Pharmacologic Substance",
        "Mental Process",
    },
    "gastrointestinal": {
        "Digestive System",
        "Disease or Syndrome",
        "Sign or Symptom",
        "Body Part, Organ, or Organ Component",
        "Diagnostic Procedure",
        "Pharmacologic Substance",
        "Enzyme",
    },
    "hepatic": {
        "Digestive System",
        "Disease or Syndrome",
        "Sign or Symptom",
        "Laboratory or Test Result",
        "Pharmacologic Substance",
        "Organic Chemical",
        "Enzyme",
    },
    "infectious": {
        "Bacterium",
        "Virus",
        "Disease or Syndrome",
        "Sign or Symptom",
        "Antibiotic",
        "Pharmacologic Substance",
        "Immunologic Factor",
        "Laboratory or Test Result",
    },
    "immunological": {
        "Immunologic Factor",
        "Disease or Syndrome",
        "Sign or Symptom",
        "Pharmacologic Substance",
        "Amino Acid, Peptide, or Protein",
        "Clinical Drug",
    },
    "musculoskeletal": {
        "Musculoskeletal System",
        "Body Part, Organ, or Organ Component",
        "Disease or Syndrome",
        "Sign or Symptom",
        "Pharmacologic Substance",
        "Immunologic Factor",
    },
    "renal": {
        "Urinary System",
        "Disease or Syndrome",
        "Sign or Symptom",
        "Laboratory or Test Result",
        "Pharmacologic Substance",
        "Organic Chemical",
        "Therapeutic or Preventive Procedure",
    },
    "hematological": {
        "Hemic and Lymphatic System",
        "Disease or Syndrome",
        "Sign or Symptom",
        "Cell",
        "Laboratory or Test Result",
        "Pharmacologic Substance",
    },
    "dermatological": {
        "Integumentary System",
        "Disease or Syndrome",
        "Sign or Symptom",
        "Pharmacologic Substance",
        "Immunologic Factor",
        "Body Part, Organ, or Organ Component",
    },
    "psychiatric": {
        "Mental or Behavioral Dysfunction",
        "Mental Process",
        "Disease or Syndrome",
        "Sign or Symptom",
        "Pharmacologic Substance",
        "Neurologic Function",
    },
    "pulmonary": {
        "Respiratory System",
        "Disease or Syndrome",
        "Sign or Symptom",
        "Diagnostic Procedure",
        "Pharmacologic Substance",
        "Body Part, Organ, or Organ Component",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Keyword lists for domain classification (fallback when semantic types
# are not available on extracted concepts — e.g. when using normalizedString
# search which may not return semantic types)
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_KEYWORDS: Dict[str, set] = {
    "respiratory": {
        "lung", "pulmonary", "bronch", "alveol", "pneumo", "dyspnea", "ventilat",
        "airway", "oxygen", "hypoxia", "embol", "pleural", "trachea", "spo2",
    },
    "endocrine": {
        "thyroid", "insulin", "glucose", "hormone", "pituitary", "adrenal",
        "cortisol", "diabetes", "ketoacid", "hba1c", "tsh", "t3", "t4",
        "parathyroid", "aldosterone", "pancrea",
    },
    "cardiac": {
        "heart", "cardiac", "coronary", "myocard", "arrhythm", "ecg", "ekg",
        "troponin", "bnp", "ventricular", "atrial", "stent", "pci", "stemi",
        "hypertens", "bp ", "blood pressure",
    },
    "neurological": {
        "brain", "neuro", "seizure", "epilep", "cerebr", "mening", "demyelin",
        "myelin", "axon", "eeg", "gcs", "cognit", "sclerosis", "reflex",
    },
    "gastrointestinal": {
        "stomach", "intestin", "colon", "bowel", "hepat", "liver", "pancrea",
        "lipase", "amylase", "gi ", "gastric", "crohn", "colitis", "ileum",
    },
    "hepatic": {
        "liver", "hepat", "bilirubin", "alt", "ast", "inr", "coagul",
        "encephalopathy", "acetaminophen", "cirrhosis", "meld",
    },
    "infectious": {
        "bacteri", "virus", "infect", "antibiotic", "culture", "gram",
        "sepsis", "septic", "meningit", "ceftriaxone", "vancomycin",
    },
    "immunological": {
        "immune", "allerg", "anaphyla", "histamine", "ige", "mast cell",
        "urticaria", "angioedema", "epinephrine", "autoimmun",
    },
    "musculoskeletal": {
        "joint", "bone", "muscle", "arthrit", "rheumat", "tendon",
        "cartilage", "synovial", "osteo", "skeletal",
    },
    "renal": {
        "kidney", "renal", "nephro", "creatinine", "gfr", "dialysis",
        "urinary", "urine", "proteinuria", "glomerul",
    },
    "hematological": {
        "blood", "hemo", "anemia", "platelet", "coagul", "sickle",
        "leukocyt", "erythrocyt", "thrombocyt", "hematocrit",
    },
    "dermatological": {
        "skin", "dermat", "rash", "psoriasis", "eczema", "lesion",
        "epiderm", "keratin", "melanocyt", "pruritus",
    },
    "psychiatric": {
        "schizo", "psycho", "hallucin", "delusion", "dopamin", "serotonin",
        "antipsychot", "mood", "anxiety", "depress", "bipolar", "mania",
    },
    "pulmonary": {
        "lung", "pulmonary", "fibrosis", "bronch", "alveol", "pneumo",
        "ventilat", "respiratory", "airway", "oxygen",
    },
}
