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
    {
        "id": "case_11", "title": "Acute Ischemic Stroke (LVO)",
        "domain": "neurological",
        "interference_domain": "dermatological",
        "interference_disease": "Atopic Dermatitis",
        "full_text": (
            "A 72-year-old man with atrial fibrillation not on anticoagulation "
            "presents with sudden-onset right-sided hemiparesis, global aphasia, "
            "and left gaze deviation; last known well 90 minutes ago. NIHSS 18. "
            "Vitals: BP 178/96 mmHg, HR 104 bpm irregularly irregular, glucose "
            "132 mg/dL, SpO2 96%. Non-contrast head CT: no hemorrhage, hyperdense "
            "left MCA sign, ASPECTS 8. CT angiography: left M1 middle cerebral "
            "artery occlusion. CT perfusion: ischemic core 18 mL, penumbra 95 mL "
            "(mismatch ratio greater than 5). IV tenecteplase 0.25 mg/kg given at "
            "110 minutes from onset. Taken for mechanical thrombectomy; TICI 2b "
            "recanalization at 200 minutes. Post-procedure NIHSS 6. Admitted to "
            "neuro-ICU with permissive hypertension, BP goal below 180/105 mmHg. "
            "Repeat CT at 24 hours: small left basal ganglia infarct without "
            "hemorrhagic transformation. Apixaban started on day 3 for "
            "cardioembolic secondary prevention. Dysphagia screen before oral intake."
        )
    },
    {
        "id": "case_12", "title": "Tension Pneumothorax",
        "domain": "pulmonary",
        "interference_domain": "renal",
        "interference_disease": "Nephrolithiasis",
        "full_text": (
            "A 25-year-old tall, thin man presents with sudden severe right-sided "
            "pleuritic chest pain and dyspnea after weightlifting. Vitals: HR 132 "
            "bpm, BP 88/54 mmHg, RR 34, SpO2 84% on room air. Exam: tracheal "
            "deviation to the left, absent breath sounds and hyperresonance over "
            "the right hemithorax, and distended neck veins, with rapidly "
            "worsening hemodynamics. Immediate needle decompression at the right "
            "second intercostal space, midclavicular line, releases a rush of air; "
            "BP improves to 112/70 mmHg and SpO2 to 94%. A chest tube is placed in "
            "the right fifth intercostal space, anterior axillary line; chest "
            "radiograph confirms right lung re-expansion with resolution of "
            "mediastinal shift. No rib fractures. CT chest shows apical blebs "
            "consistent with primary spontaneous pneumothorax. Admitted for "
            "chest-tube management on water-seal drainage; thoracic surgery "
            "consulted for pleurodesis given the high recurrence risk."
        )
    },
    {
        "id": "case_13", "title": "Variceal Upper GI Hemorrhage",
        "domain": "gastrointestinal",
        "interference_domain": "neurological",
        "interference_disease": "Migraine",
        "full_text": (
            "A 56-year-old man with alcohol-related cirrhosis presents with "
            "hematemesis and melena. Vitals: HR 118 bpm, BP 92/58 mmHg, RR 20, "
            "with orthostatic changes. Exam: pallor, spider angiomata, "
            "splenomegaly, and mild ascites. Labs: Hb 7.2 g/dL (baseline 12), "
            "platelets 78,000/uL, INR 1.6, total bilirubin 3.1 mg/dL, albumin "
            "2.8 g/dL, BUN 42 mg/dL. Resuscitated through two large-bore IVs with "
            "a restrictive transfusion target of 7-8 g/dL. Octreotide 50 mcg IV "
            "bolus then 50 mcg/hr infusion, IV ceftriaxone 1 g daily for "
            "prophylaxis, and IV pantoprazole started. Urgent EGD within 12 hours "
            "reveals large esophageal varices with active spurting; endoscopic "
            "band ligation achieves hemostasis. Child-Pugh class C, MELD 19. "
            "Admitted to the ICU; a non-selective beta-blocker for secondary "
            "prophylaxis is deferred until hemodynamically stable."
        )
    },
    {
        "id": "case_14", "title": "Thyroid Storm",
        "domain": "endocrine",
        "interference_domain": "musculoskeletal",
        "interference_disease": "Osteoarthritis",
        "full_text": (
            "A 41-year-old woman with poorly controlled Graves disease presents "
            "with fever, agitation, and palpitations two days after an upper "
            "respiratory infection. Vitals: T 40.1 C, HR 168 bpm (atrial "
            "fibrillation with rapid ventricular response), BP 158/64 mmHg, RR 26. "
            "Exam: diaphoresis, lid lag, a diffuse goiter with bruit, fine tremor, "
            "and confusion. Burch-Wartofsky score 75 (highly suggestive of thyroid "
            "storm). Labs: TSH less than 0.01 mIU/L, free T4 6.8 ng/dL, markedly "
            "elevated free T3, mild transaminitis, glucose 184 mg/dL. Management: "
            "propranolol 60-80 mg PO q4h for adrenergic control, propylthiouracil "
            "500-1000 mg load then 250 mg q4h, with iodine (SSKI) begun one hour "
            "after PTU to block hormone release, plus hydrocortisone 100 mg IV q8h "
            "and active cooling with acetaminophen. The precipitating infection is "
            "treated. Admitted to the ICU on telemetry; rate and temperature "
            "improve over 24 hours."
        )
    },
    {
        "id": "case_15", "title": "Cardiogenic Shock",
        "domain": "cardiac",
        "interference_domain": "gastrointestinal",
        "interference_disease": "Irritable Bowel Syndrome",
        "full_text": (
            "A 67-year-old man with ischemic cardiomyopathy (EF 20%) presents with "
            "progressive dyspnea, orthopnea, and cool extremities. Vitals: HR 112 "
            "bpm, BP 82/60 mmHg, RR 28, SpO2 89% on room air. Exam: elevated JVP, "
            "bilateral crackles, an S3 gallop, and cool, clammy, mottled skin. "
            "Labs: lactate 3.8 mmol/L, creatinine 1.9 mg/dL (baseline 1.1), "
            "NT-proBNP 9,400 pg/mL, and mildly elevated troponin. Bedside echo "
            "shows severely reduced LV systolic function without tamponade. "
            "Pulmonary artery catheter: cardiac index 1.6 L/min/m2 with elevated "
            "wedge pressure, consistent with cardiogenic shock (SCAI stage C). "
            "Management: IV furosemide for congestion, dobutamine for inotropic "
            "support, and norepinephrine for MAP below 65 mmHg. Cardiology "
            "evaluates for temporary mechanical circulatory support and advanced "
            "heart-failure therapies. Admitted to the CICU with arterial and "
            "central venous access."
        )
    },
    {
        "id": "case_16", "title": "Acute Hemolytic Transfusion Reaction",
        "domain": "hematological",
        "interference_domain": "psychiatric",
        "interference_disease": "Generalized Anxiety Disorder",
        "full_text": (
            "A 60-year-old woman receiving a packed red blood cell transfusion for "
            "symptomatic anemia develops fever, chills, flank pain, and burning at "
            "the infusion site 15 minutes into the transfusion. Vitals: T 39.0 C, "
            "HR 122 bpm, BP 86/50 mmHg, RR 24. The transfusion is stopped "
            "immediately and the line kept open with normal saline. Exam: dark, "
            "cola-colored urine and diffuse oozing from the IV site. Labs: falling "
            "hemoglobin, markedly elevated LDH, undetectable haptoglobin, elevated "
            "indirect bilirubin, and high plasma free hemoglobin; the direct "
            "antiglobulin (Coombs) test is positive and repeat crossmatch reveals "
            "ABO incompatibility from a clerical mismatch. DIC panel: low "
            "fibrinogen, elevated D-dimer, and prolonged PT/PTT. Management: "
            "aggressive IV crystalloid to maintain urine output above 1 mL/kg/hr, "
            "vasopressors for hypotension, and supportive care for hemolysis and "
            "DIC. The blood bank is notified and the clerical error investigated. "
            "Admitted to the ICU to monitor renal function and coagulopathy."
        )
    },
    {
        "id": "case_17", "title": "Necrotizing Fasciitis",
        "domain": "infectious",
        "interference_domain": "endocrine",
        "interference_disease": "Hypothyroidism",
        "full_text": (
            "A 58-year-old man with type 2 diabetes presents with severe left "
            "lower-extremity pain out of proportion to exam, 24 hours after a "
            "minor abrasion. Vitals: T 39.2 C, HR 126 bpm, BP 94/58 mmHg, RR 24. "
            "Exam: tense edema extending beyond a dusky erythematous patch, skin "
            "bullae, crepitus, and decreased sensation over the calf. Labs: WBC "
            "26,000/uL, glucose 320 mg/dL, Na 128 mEq/L, creatinine 1.9 mg/dL, "
            "CRP 280 mg/L, lactate 4.1 mmol/L; LRINEC score 9 (high risk). Plain "
            "film shows subcutaneous gas. Management: emergent surgical "
            "exploration with wide debridement for source control, broad-spectrum "
            "antibiotics (piperacillin-tazobactam plus vancomycin plus clindamycin "
            "for its antitoxin effect), and aggressive fluid resuscitation. Blood "
            "and tissue cultures are drawn. Admitted to the ICU with a planned "
            "second-look debridement in 24 hours."
        )
    },
    {
        "id": "case_18", "title": "Tumor Lysis Syndrome",
        "domain": "renal",
        "interference_domain": "pulmonary",
        "interference_disease": "Sarcoidosis",
        "full_text": (
            "A 34-year-old man with newly diagnosed Burkitt lymphoma develops "
            "nausea, muscle cramps, and decreased urine output 48 hours after "
            "initiating chemotherapy. Vitals: HR 104 bpm, BP 142/88 mmHg, RR 18. "
            "Labs: potassium 6.4 mEq/L, phosphate 7.8 mg/dL, calcium 6.8 mg/dL, "
            "uric acid 13 mg/dL, creatinine 3.2 mg/dL (baseline 0.9), and markedly "
            "elevated LDH, meeting Cairo-Bishop criteria for tumor lysis syndrome "
            "with acute kidney injury. EKG: peaked T waves. Management: continuous "
            "cardiac monitoring, IV calcium gluconate for membrane stabilization, "
            "insulin-dextrose and an inhaled beta-agonist for hyperkalemia, "
            "aggressive isotonic IV fluids, and rasburicase for hyperuricemia. "
            "Nephrology is consulted for possible renal replacement therapy given "
            "oliguria and refractory hyperkalemia. Admitted to the ICU."
        )
    },
    {
        "id": "case_19", "title": "Status Asthmaticus",
        "domain": "respiratory",
        "interference_domain": "dermatological",
        "interference_disease": "Contact Dermatitis",
        "full_text": (
            "A 23-year-old woman with asthma presents with severe dyspnea and "
            "wheezing unresponsive to home albuterol. Vitals: HR 128 bpm, RR 32, "
            "SpO2 88% on room air; she speaks in single words and uses accessory "
            "muscles. Exam: diffuse wheezing with a prolonged expiratory phase, "
            "transiently progressing toward a silent chest. Peak expiratory flow "
            "is below 40% of predicted. ABG: pH 7.31, pCO2 48 mmHg (a rising pCO2 "
            "is ominous in a tachypneic asthmatic), pO2 70 mmHg. Management: "
            "continuous nebulized albuterol with ipratropium, systemic "
            "corticosteroids (methylprednisolone 125 mg IV), IV magnesium sulfate "
            "2 g, and supplemental oxygen. Despite therapy, rising pCO2 and "
            "fatigue prompt a trial of non-invasive ventilation with preparation "
            "for intubation. Admitted to the ICU for impending respiratory failure."
        )
    },
    {
        "id": "case_20", "title": "Cauda Equina Syndrome",
        "domain": "musculoskeletal",
        "interference_domain": "cardiac",
        "interference_disease": "Stable Angina",
        "full_text": (
            "A 48-year-old man presents with severe low back pain radiating down "
            "both legs, progressive bilateral leg weakness, urinary retention, and "
            "saddle anesthesia evolving over 24 hours, with a history of lumbar "
            "disc disease and recent heavy lifting. Exam: decreased perianal "
            "sensation, lax anal sphincter tone, bilateral lower-extremity "
            "weakness (4-/5), and absent ankle reflexes; bladder scan shows 700 mL "
            "of retained urine with overflow incontinence. Emergent MRI of the "
            "lumbar spine reveals a large L4-L5 disc herniation with severe "
            "compression of the cauda equina. Management: urgent neurosurgical "
            "consultation for surgical decompression within 48 hours to preserve "
            "neurologic function, a Foley catheter for retention, and high-dose "
            "analgesia. Admitted for emergent operative planning."
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
            f"As you reason, please consider potential implications or differential "
            f"considerations from the {interf_domain} domain.\n\n"
            f"Case:\n{text}"
        )
    elif level_label == "hard_interference":
        return (
            f"You are a board-certified physician. Analyse the following clinical "
            f"case and provide a step-by-step diagnostic reasoning chain. "
            f"You must frame every step of your reasoning through the lens of "
            f"{interf_domain} pathophysiology and mechanisms.\n\n"
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
        "breathing", "shortness of breath", "cough",
    },
    "endocrine": {
        "thyroid", "insulin", "glucose", "hormone", "pituitary", "adrenal",
        "cortisol", "diabetes", "ketoacid", "hba1c", "tsh", "t3", "t4",
        "parathyroid", "aldosterone", "pancrea", "gland", "sugar level",
    },
    "cardiac": {
        "heart", "cardiac", "coronary", "myocard", "arrhythm", "ecg", "ekg",
        "troponin", "bnp", "ventricular", "atrial", "stent", "pci", "stemi",
        "hypertens", "bp ", "blood pressure", "chest pain", "pulse",
    },
    "neurological": {
        "brain", "neuro", "seizure", "epilep", "cerebr", "mening", "demyelin",
        "myelin", "axon", "eeg", "gcs", "cognit", "sclerosis", "reflex",
        "nerve", "paralysis", "numbness",
    },
    "gastrointestinal": {
        "stomach", "intestin", "colon", "bowel", "hepat", "liver", "pancrea",
        "lipase", "amylase", "gi ", "gastric", "crohn", "colitis", "ileum",
        "digestion", "nausea", "vomit",
    },
    "hepatic": {
        "liver", "hepat", "bilirubin", "alt", "ast", "inr", "coagul",
        "encephalopathy", "acetaminophen", "cirrhosis", "meld", "jaundice",
    },
    "infectious": {
        "bacteri", "virus", "infect", "antibiotic", "culture", "gram",
        "sepsis", "septic", "meningit", "ceftriaxone", "vancomycin",
        "fever", "chills", "contagious",
    },
    "immunological": {
        "immune", "allerg", "anaphyla", "histamine", "ige", "mast cell",
        "urticaria", "angioedema", "epinephrine", "autoimmun", "swelling",
    },
    "musculoskeletal": {
        "joint", "bone", "muscle", "arthrit", "rheumat", "tendon",
        "cartilage", "synovial", "osteo", "skeletal", "stiffness",
    },
    "renal": {
        "kidney", "renal", "nephro", "creatinine", "gfr", "dialysis",
        "urinary", "urine", "proteinuria", "glomerul", "bladder",
    },
    "hematological": {
        "blood", "hemo", "anemia", "platelet", "coagul", "sickle",
        "leukocyt", "erythrocyt", "thrombocyt", "hematocrit", "bleeding",
    },
    "dermatological": {
        "skin", "dermat", "rash", "psoriasis", "eczema", "lesion",
        "epiderm", "keratin", "melanocyt", "pruritus", "itch",
    },
    "psychiatric": {
        "schizo", "psycho", "hallucin", "delusion", "dopamin", "serotonin",
        "antipsychot", "mood", "anxiety", "depress", "bipolar", "mania",
        "behavior", "mental",
    },
    "pulmonary": {
        "lung", "pulmonary", "fibrosis", "bronch", "alveol", "pneumo",
        "ventilat", "respiratory", "airway", "oxygen", "breath",
    },
}
