"""Build a single, unified clinical knowledge base dataset.

This script consolidates all clinical guidelines, disease presentations,
differential diagnosis protocols, and medical evidence across all major
medical specialties into a single canonical file:
    data/unified_clinical_kb.jsonl

This unified dataset is loaded directly by the backend API to process live
queries (retrieval, AEB reasoning, and confidence estimation).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "unified_clinical_kb.jsonl"


# 1. Authoritative Clinical Practice Guidelines across all primary specialties
CORE_CLINICAL_GUIDELINES = [
    {
        "id": "KB-CARD-001",
        "source": "WHO/cardiac_chest_pain",
        "title": "WHO Guideline: Acute Coronary Syndrome and Cardiac Chest Pain",
        "specialty": "cardiology",
        "year": 2024,
        "trust_score": 0.95,
        "text": (
            "Chest pain is a cardinal symptom of acute coronary syndrome (ACS) including STEMI and NSTEMI. "
            "Patients over 50 presenting with central crushing chest pain, diaphoresis (sweating), radiation "
            "to the jaw, neck, or left arm, and dyspnea (shortness of breath) are at critical risk for acute myocardial infarction. "
            "Patients with diabetes or elderly individuals may present with atypical symptoms including silent ischemia, isolated dyspnea, nausea, or fatigue. "
            "Immediate 12-lead ECG is mandatory within 10 minutes of presentation. "
            "Elevated high-sensitivity cardiac troponin I or T confirms myocardial necrosis. "
            "Emergency initial management consists of chewable aspirin (162-325 mg), sublingual nitroglycerin (if systolic BP > 90 mmHg and no right ventricular infarction), "
            "supplemental oxygen if SaO2 < 90%, and urgent referral for percutaneous coronary intervention (PCI) or thrombolytic therapy if PCI is unavailable."
        ),
    },
    {
        "id": "KB-CARD-002",
        "source": "ACC-AHA/heart_failure",
        "title": "ACC/AHA Guideline: Acute Decompensated Heart Failure",
        "specialty": "cardiology",
        "year": 2023,
        "trust_score": 0.94,
        "text": (
            "Acute decompensated heart failure (ADHF) presents with progressive dyspnea on exertion, orthopnea, "
            "paroxysmal nocturnal dyspnea, bilateral lower extremity peripheral edema, and fatigue. "
            "Physical examination typically reveals elevated jugular venous distention (JVD), bilateral pulmonary crackles/rales, "
            "an S3 gallop, and hepatojugular reflux. Elevated serum B-type natriuretic peptide (BNP > 100 pg/mL) or NT-proBNP (> 300 pg/mL) "
            "strongly supports the diagnosis. Chest radiograph demonstrates cardiomegaly, cephalization of pulmonary veins, Kerley B lines, and interstitial edema. "
            "Immediate management includes intravenous loop diuretics (e.g. furosemide), vasodilators (nitroglycerin) if hypertensive, "
            "and non-invasive positive pressure ventilation (BiPAP/CPAP) for acute respiratory distress. Beta-blockers should not be initiated acutely during severe decompensation."
        ),
    },
    {
        "id": "KB-CARD-003",
        "source": "NIH-JNC8/hypertension",
        "title": "NIH/ACC/AHA Guideline: Hypertension and Hypertensive Emergency",
        "specialty": "cardiology",
        "year": 2023,
        "trust_score": 0.93,
        "text": (
            "Hypertension is defined as sustained systolic blood pressure >= 130 mmHg or diastolic BP >= 80 mmHg. "
            "Stage 2 hypertension is BP >= 140/90 mmHg. Hypertensive emergency is defined as acute severe BP elevation (> 180/120 mmHg) "
            "with evidence of acute, life-threatening target organ damage (e.g. encephalopathy, acute aortic dissection, pulmonary edema, myocardial infarction, or acute renal failure). "
            "Routine outpatient first-line pharmacological treatment includes thiazide diuretics, ACE inhibitors, angiotensin receptor blockers (ARBs), and dihydropyridine calcium channel blockers. "
            "In patients with diabetes or chronic kidney disease, ACE inhibitors or ARBs are first-line for nephroprotective benefit. "
            "Target BP in diabetic patients is < 130/80 mmHg. In hypertensive emergencies, intravenous agents (labetalol, nicardipine, nitroprusside) are titrated to reduce mean arterial pressure by 20-25% over the first hour."
        ),
    },
    {
        "id": "KB-CARD-004",
        "source": "ESC/atrial_fibrillation",
        "title": "ESC Guideline: Management of Atrial Fibrillation",
        "specialty": "cardiology",
        "year": 2024,
        "trust_score": 0.92,
        "text": (
            "Atrial fibrillation (AF) is a supraventricular tachyarrhythmia characterized by uncoordinated atrial activation and irregularly irregular ventricular response. "
            "Symptoms include palpitations, lightheadedness, fatigue, exertional dyspnea, and chest discomfort. "
            "ECG diagnosis requires absence of distinct P waves, fibrillatory waves, and irregularly irregular RR intervals. "
            "Initial management prioritizes hemodynamic stability: synchronized electrical cardioversion is indicated for hemodynamically unstable patients. "
            "Rate control is achieved with beta-blockers (metoprolol) or non-dihydropyridine calcium channel blockers (diltiazem, verapamil). "
            "Thromboembolic risk is stratified using the CHA2DS2-VASc score (Congestive heart failure, Hypertension, Age >= 75 [2], Diabetes, Stroke [2], Vascular disease, Age 65-74, Sex category). "
            "Anticoagulation with DOACs (apixaban, rivaroxaban, dabigatran) is indicated for males with score >= 2 and females with score >= 3."
        ),
    },
    {
        "id": "KB-PULM-001",
        "source": "CDC/pneumonia_guideline",
        "title": "CDC/IDSA Guideline: Community-Acquired Pneumonia",
        "specialty": "pulmonology",
        "year": 2024,
        "trust_score": 0.95,
        "text": (
            "Community-acquired pneumonia (CAP) presents with acute productive cough, fever (> 38°C), chills, purulent sputum production, "
            "dyspnea, and pleuritic chest pain. Physical examination demonstrates localized crackles (rales), bronchial breath sounds, dullness to percussion, and increased tactile fremitus. "
            "Chest radiography demonstrating lobar consolidation, patchy alveolar infiltrates, or air bronchograms confirms the diagnosis. "
            "Clinical severity is stratified using the CURB-65 criteria (Confusion, Urea > 7 mmol/L, Respiratory rate >= 30/min, low Blood pressure [SBP < 90 or DBP <= 60], Age >= 65). "
            "Score 0-1: outpatient treatment; score 2: consider hospital admission; score >= 3: inpatient admission, evaluate for ICU care. "
            "First-line outpatient treatment for healthy adults is amoxicillin or doxycycline; for patients with comorbidities (diabetes, COPD, heart disease), "
            "combination therapy with amoxicillin/clavulanate plus azithromycin, or respiratory fluoroquinolone (levofloxacin, moxifloxacin) is indicated."
        ),
    },
    {
        "id": "KB-PULM-002",
        "source": "WHO-GOLD/copd_guideline",
        "title": "GOLD/WHO Guideline: Chronic Obstructive Pulmonary Disease and Acute Exacerbation",
        "specialty": "pulmonology",
        "year": 2024,
        "trust_score": 0.94,
        "text": (
            "Chronic Obstructive Pulmonary Disease (COPD) is characterized by persistent, progressive airflow limitation (post-bronchodilator FEV1/FVC < 0.70). "
            "Major risk factors include cigarette smoking, occupational exposures, and alpha-1 antitrypsin deficiency. "
            "Acute exacerbation of COPD (AECOPD) is an acute worsening of respiratory symptoms characterized by the Anthonisen criteria: "
            "increased dyspnea, increased sputum volume, and increased sputum purulence. "
            "Management of acute exacerbations entails: inhaled short-acting beta-2 agonists (salbutamol) and short-acting anticholinergics (ipratropium), "
            "systemic oral corticosteroids (prednisone 40 mg daily for 5 days), and targeted antibiotics (macrolide, doxycycline, or amoxicillin-clavulanate) "
            "when sputum purulence is present. Controlled oxygen therapy targeting SpO2 88-92% is essential to avoid hypercapnic respiratory drive suppression."
        ),
    },
    {
        "id": "KB-PULM-003",
        "source": "GINA/asthma_guideline",
        "title": "GINA/CDC Guideline: Acute Asthma Exacerbation and Management",
        "specialty": "pulmonology",
        "year": 2024,
        "trust_score": 0.93,
        "text": (
            "Asthma is a chronic inflammatory disorder of the airways characterized by episodic wheezing, breathlessness, chest tightness, and nighttime or early morning cough. "
            "Triggers include viral infections, allergens, cold air, exercise, NSAIDs, and beta-blockers. "
            "Acute severe asthma presents with marked tachypnea, tachycardia, inability to complete full sentences in one breath, pulsus paradoxus, "
            "and peak expiratory flow (PEF) < 50% of personal best. "
            "Life-threatening red flags include a 'silent chest' (exhaustion of airflow), cyanosis, bradycardia, confusion, and normal or elevated PaCO2 on ABG indicating respiratory muscle fatigue. "
            "Immediate emergency management includes high-dose inhaled short-acting beta-2 agonist (albuterol) via nebulizer or MDI with spacer, "
            "inhaled ipratropium bromide, early systemic corticosteroids (oral prednisolone or IV methylprednisolone), oxygen targeting SpO2 93-95%, "
            "and IV magnesium sulfate (2 g IV infusion over 20 min) for severe or refractory broncho-constriction."
        ),
    },
    {
        "id": "KB-PULM-004",
        "source": "CHEST-NICE/pulmonary_embolism",
        "title": "CHEST/NICE Guideline: Acute Pulmonary Embolism and Deep Vein Thrombosis",
        "specialty": "pulmonology",
        "year": 2023,
        "trust_score": 0.94,
        "text": (
            "Pulmonary embolism (PE) occurs when a thromboembolus, typically arising from lower-extremity deep vein thrombosis (DVT), occludes the pulmonary arterial bed. "
            "Classic clinical triad includes acute onset pleuritic chest pain, unexplained dyspnea, and tachypnea. Hemoptysis, tachycardia, syncope, and unilateral calf swelling/pain support the diagnosis. "
            "Clinical pre-test probability is assessed via Wells Score or Geneva Score. Low-probability patients can be evaluated with high-sensitivity D-dimer testing; a normal D-dimer safely excludes PE. "
            "High-probability patients or those with positive D-dimer require CT Pulmonary Angiography (CTPA) or V/Q scan if renal impairment or iodinated contrast allergy exists. "
            "Massive PE causes acute hemodynamic instability (systolic BP < 90 mmHg) and requires emergency systemic thrombolysis (e.g. alteplase) or surgical embolectomy. "
            "Hemodynamically stable PE is managed with immediate therapeutic anticoagulation using DOACs (rivaroxaban, apixaban) or low-molecular-weight heparin (LMWH)."
        ),
    },
    {
        "id": "KB-NEUR-001",
        "source": "AHA-ASA/stroke_guideline",
        "title": "AHA/ASA/WHO Guideline: Acute Ischemic Stroke Evaluation and Thrombolysis",
        "specialty": "neurology",
        "year": 2024,
        "trust_score": 0.95,
        "text": (
            "Acute ischemic stroke represents sudden vascular compromise causing focal neurological deficits. "
            "Common presentations include unilateral hemiparesis or facial droop, acute slurred speech or aphasia, ataxia, visual field loss, and hemisensory deficits. "
            "Pre-hospital identification uses the FAST algorithm (Face droop, Arm weakness, Speech difficulty, Time to call emergency). "
            "Emergency non-contrast head CT is critical immediately on arrival to rule out intracranial hemorrhage prior to reperfusion therapy. "
            "Intravenous thrombolysis with recombinant tissue plasminogen activator (alteplase 0.9 mg/kg or tenecteplase) is indicated within 4.5 hours of symptom onset "
            "provided BP < 185/110 mmHg and no contraindications (recent major surgery, active bleeding, severe thrombocytopenia). "
            "Endovascular mechanical thrombectomy is indicated for large vessel occlusion (LVO) in the anterior circulation within 6-24 hours of last known normal."
        ),
    },
    {
        "id": "KB-NEUR-002",
        "source": "NICE/transient_ischemic_attack",
        "title": "NICE Guideline: Transient Ischemic Attack (TIA) and Secondary Prevention",
        "specialty": "neurology",
        "year": 2023,
        "trust_score": 0.92,
        "text": (
            "A Transient Ischemic Attack (TIA) is defined as a transient episode of neurological dysfunction caused by focal brain, spinal cord, or retinal ischemia, "
            "without acute infarction on neuroimaging. Symptoms typically resolve completely within 1 hour. "
            "Short-term risk of full stroke following TIA is high, especially within the first 48 hours. Risk stratification uses the ABCD2 score "
            "(Age >= 60, Blood pressure >= 140/90, Clinical features [unilateral weakness=2, speech disturbance=1], Duration [>=60 min=2, 10-59 min=1], and Diabetes). "
            "Immediate specialist neurological assessment within 24 hours is required. Immediate antiplatelet therapy consists of aspirin 300 mg daily, followed by dual antiplatelet therapy "
            "(aspirin + clopidogrel) for 21 days for high-risk TIA. Carotid Doppler ultrasound should be performed urgently to identify severe carotid stenosis amenable to carotid endarterectomy."
        ),
    },
    {
        "id": "KB-NEUR-003",
        "source": "AES-IDSA/meningitis_and_seizures",
        "title": "IDSA/AES Guideline: Acute Bacterial Meningitis and Status Epilepticus",
        "specialty": "neurology",
        "year": 2023,
        "trust_score": 0.93,
        "text": (
            "Acute bacterial meningitis is an emergency characterized by the classic triad of fever, altered mental status, and nuchal rigidity (neck stiffness). "
            "Positive Kernig's and Brudzinski's signs indicate meningeal irritation. Headache, photophobia, and petechial or purpuric rash (characteristic of Neisseria meningitidis) may be present. "
            "Immediate blood cultures followed by empirical IV antibiotics (ceftriaxone + vancomycin + ampicillin if > 50yo for Listeria coverage) and adjunctive IV dexamethasone "
            "must be administered without delay. Lumbar puncture should not delay antibiotic administration. "
            "Status epilepticus is defined as continuous seizure activity >= 5 minutes or recurrent seizures without return to baseline. "
            "First-line treatment is IV lorazepam (4 mg) or IM midazolam (10 mg), followed immediately by second-line IV antiseizure medications: levetiracetam, fosphenytoin, or valproate sodium."
        ),
    },
    {
        "id": "KB-ENDO-001",
        "source": "NICE-ADA/diabetes_type2",
        "title": "NICE/ADA Guideline: Type 2 Diabetes Management and Hyperglycemic Crises",
        "specialty": "endocrinology",
        "year": 2024,
        "trust_score": 0.94,
        "text": (
            "Type 2 diabetes mellitus (T2DM) is characterized by progressive insulin secretory defect on the background of insulin resistance. "
            "Diagnostic criteria: fasting plasma glucose >= 126 mg/dL (7.0 mmol/L), HbA1c >= 6.5% (48 mmol/mol), or 2-hour post-OGTT glucose >= 200 mg/dL. "
            "Classic presentation includes polyuria, polydipsia, polyphagia, fatigue, and blurred vision. "
            "First-line pharmacotherapy is metformin, with dose titration up to 2000 mg/day (monitor eGFR; avoid if eGFR < 30 mL/min). "
            "For patients with established atherosclerotic cardiovascular disease, heart failure, or chronic kidney disease, SGLT2 inhibitors (empagliflozin, dapagliflozin) "
            "or GLP-1 receptor agonists (semaglutide, liraglutide) with demonstrated cardiovascular and renal benefit are added regardless of baseline HbA1c. "
            "Diabetic ketoacidosis (DKA) and Hyperglycemic Hyperosmolar State (HHS) are medical emergencies requiring vigorous IV 0.9% saline fluid resuscitation, "
            "regular insulin infusion (0.1 units/kg/h), and potassium monitoring and repletion before insulin administration if K+ < 3.3 mEq/L."
        ),
    },
    {
        "id": "KB-ENDO-002",
        "source": "ATA-NICE/hypothyroidism",
        "title": "ATA/NICE Guideline: Hypothyroidism and Thyroid Disorders",
        "specialty": "endocrinology",
        "year": 2023,
        "trust_score": 0.93,
        "text": (
            "Primary hypothyroidism is characterized by insufficient thyroid hormone production from the thyroid gland, most commonly due to Hashimoto's autoimmune thyroiditis. "
            "Symptoms include chronic fatigue, cold intolerance, weight gain, constipation, dry coarse skin, periorbital puffiness, non-pitting myxedema, and delayed relaxation phase of deep tendon reflexes. "
            "Laboratory diagnosis demonstrates elevated serum Thyroid-Stimulating Hormone (TSH) with decreased free thyroxine (free T4). Subclinical hypothyroidism displays elevated TSH with normal free T4. "
            "First-line replacement therapy is oral levothyroxine (T4) administered once daily on an empty stomach 30-60 minutes before breakfast. "
            "Standard full replacement dose is approximately 1.6 mcg/kg/day in young healthy adults. In elderly patients or those with coronary artery disease, start with a lower dose (25-50 mcg daily) "
            "to prevent precipitation of angina or arrhythmias. Monitor serum TSH every 6 to 8 weeks after dose adjustment until euthyroid."
        ),
    },
    {
        "id": "KB-NEPH-001",
        "source": "NICE/uti_guideline",
        "title": "NICE Guideline: Uncomplicated Lower Urinary Tract Infection (Cystitis)",
        "specialty": "nephrology",
        "year": 2024,
        "trust_score": 0.94,
        "text": (
            "Uncomplicated lower urinary tract infection (cystitis) occurs primarily in non-pregnant adult females with a structurally normal urinary tract. "
            "Classic triad of lower tract symptoms: dysuria (burning sensation on micturition), urinary frequency, and urinary urgency. "
            "Suprapubic discomfort and micro/macro-hematuria may accompany cystitis. "
            "Absence of fever, chills, rigors, nausea/vomiting, and costovertebral angle (CVA) flank tenderness differentiates uncomplicated cystitis from acute pyelonephritis. "
            "Urine dipstick testing showing positive leukocyte esterase and nitrites supports bacterial infection (most commonly Escherichia coli). "
            "First-line empirical antimicrobial therapy: nitrofurantoin 100 mg orally twice daily for 5 days, or trimethoprim-sulfamethoxazole (TMP-SMX) 160/800 mg twice daily for 3 days (if local E. coli resistance < 20%), "
            "or fosfomycin trometamol 3 g single oral dose. Fluoroquinolones (ciprofloxacin) are reserved for complicated infections due to adverse effect profiles."
        ),
    },
    {
        "id": "KB-NEPH-002",
        "source": "IDSA-NICE/pyelonephritis",
        "title": "IDSA/NICE Guideline: Acute Pyelonephritis and Upper UTI",
        "specialty": "nephrology",
        "year": 2023,
        "trust_score": 0.94,
        "text": (
            "Acute pyelonephritis is an invasive bacterial infection of the renal parenchyma and collecting system. "
            "Clinical presentation includes fever (> 38°C), shaking chills, unilateral or bilateral costovertebral angle (CVA) tenderness (flank pain), "
            "nausea, vomiting, and systemic malaise, frequently accompanied by lower urinary tract symptoms (dysuria, frequency). "
            "Urinalysis reveals pyuria, bacteriuria, and white blood cell (WBC) casts, which are pathognomonic for upper tract renal involvement. "
            "Blood and urine cultures should be obtained in all patients prior to initiating antibiotic therapy. "
            "Mild-to-moderate outpatient pyelonephritis can be treated with oral ciprofloxacin 500 mg twice daily for 7 days or levofloxacin 750 mg daily for 5 days. "
            "Hospitalization and IV antibiotics (ceftriaxone, piperacillin-tazobactam) are required for pregnant women, severe dehydration, inability to tolerate oral medications, "
            "hemodynamic instability, or suspicion of urinary tract obstruction/calculus."
        ),
    },
    {
        "id": "KB-NEPH-003",
        "source": "KDIGO/acute_kidney_injury",
        "title": "KDIGO Guideline: Acute Kidney Injury Evaluation and Staging",
        "specialty": "nephrology",
        "year": 2023,
        "trust_score": 0.93,
        "text": (
            "Acute Kidney Injury (AKI) is defined by KDIGO criteria as any of: increase in serum creatinine by >= 0.3 mg/dL (26.5 umol/L) within 48 hours; "
            "or increase in serum creatinine to >= 1.5 times baseline within 7 days; or urine output < 0.5 mL/kg/h for 6 hours. "
            "Etiologies are categorized into: "
            "1. Pre-renal: intravascular volume depletion, sepsis, heart failure, NSAID/ACEi hemodynamic effects (BUN/Cr ratio > 20:1, FeNa < 1%). "
            "2. Intrinsic renal: acute tubular necrosis (ATN, muddy brown granular casts), acute interstitial nephritis (AIN, drug-induced, eosinophils), glomerulonephritis (RBC casts, proteinuria). "
            "3. Post-renal: bilateral urinary tract obstruction, benign prostatic hyperplasia, obstructing calculi (hydronephrosis on renal ultrasound). "
            "Immediate management involves discontinuing all nephrotoxic medications (NSAIDs, aminoglycosides, IV contrast), correcting volume status, "
            "and treating hyperkalemia, metabolic acidosis, and volume overload."
        ),
    },
    {
        "id": "KB-EMERG-001",
        "source": "WAO-NICE/anaphylaxis",
        "title": "WAO/NICE Guideline: Anaphylaxis Emergency Recognition and Treatment",
        "specialty": "emergency",
        "year": 2024,
        "trust_score": 0.95,
        "text": (
            "Anaphylaxis is an acute, life-threatening systemic hypersensitivity reaction characterized by rapid onset of airway, breathing, or circulatory compromise. "
            "Diagnostic criteria: sudden onset of mucosal/skin symptoms (generalized hives/urticaria, pruritus, flushing, swollen lips/tongue/uvula/angioedema) PLUS at least one of: "
            "1. Respiratory compromise: dyspnea, wheezing, stridor, hypoxemia. "
            "2. Reduced blood pressure or end-organ dysfunction: hypotension (SBP < 90 mmHg or > 30% decrease from baseline), syncope, incontinence, diaphoresis. "
            "Common triggers include foods (peanuts, tree nuts, shellfish), medications (beta-lactam antibiotics, NSAIDs), and insect venoms. "
            "FIRST-LINE TREATMENT IS IMMEDIATE INTRAMUSCULAR (IM) EPINEPHRINE (adrenaline) 0.01 mg/kg, maximum 0.5 mg in adults (0.3 mg in children), "
            "injected into the anterolateral mid-thigh. Do not delay epinephrine for antihistamines or corticosteroids! "
            "Repeat epinephrine every 5-15 minutes if symptoms persist. Place patient in recumbent position with elevated legs. "
            "Secondary adjunctive treatments include IV crystalloid fluid boluses (1-2 L 0.9% saline) for hypotension, nebulized albuterol for bronchospasm, "
            "and H1/H2 antihistamines and IV corticosteroids to prevent protracted or biphasic reactions."
        ),
    },
    {
        "id": "KB-EMERG-002",
        "source": "SSC-WHO/sepsis_guideline",
        "title": "Surviving Sepsis Campaign: Management of Sepsis and Septic Shock",
        "specialty": "emergency",
        "year": 2024,
        "trust_score": 0.95,
        "text": (
            "Sepsis is defined as life-threatening organ dysfunction caused by a dysregulated host response to infection (SOFA score increase >= 2 points). "
            "Quick SOFA (qSOFA) bedside screening uses 3 criteria: respiratory rate >= 22/min, altered mentation (GCS < 15), and systolic BP <= 100 mmHg. "
            "Septic shock is defined as sepsis requiring vasopressors to maintain MAP >= 65 mmHg and serum lactate > 2 mmol/L despite adequate fluid resuscitation. "
            "The Sepsis Hour-1 Bundle requires immediate implementation: "
            "1. Measure serum lactate level; remeasure if initial lactate > 2 mmol/L. "
            "2. Obtain blood cultures prior to administration of antibiotics. "
            "3. Administer broad-spectrum empiric IV antimicrobials within 1 hour. "
            "4. Rapidly infuse 30 mL/kg crystalloid fluid for hypotension (MAP < 65 mmHg) or lactate >= 4 mmol/L. "
            "5. Apply vasopressors (norepinephrine is first-line) during or after fluid resuscitation to maintain MAP >= 65 mmHg."
        ),
    },
    {
        "id": "KB-GASTRO-001",
        "source": "ACG/peptic_ulcer_and_gastritis",
        "title": "ACG Guideline: Gastritis, Peptic Ulcer Disease, and H. Pylori",
        "specialty": "gastroenterology",
        "year": 2023,
        "trust_score": 0.93,
        "text": (
            "Gastritis and Peptic Ulcer Disease (PUD) present with burning epigastric pain, dyspepsia, early satiety, postprandial fullness, and nausea. "
            "Duodenal ulcer pain typically improves with meals and recurs 2-3 hours later or at night, whereas gastric ulcer pain is frequently exacerbated by food ingestion. "
            "Major etiologies include Helicobacter pylori infection and extensive nonsteroidal anti-inflammatory drug (NSAID) or aspirin use. "
            "Alarm 'red flag' features requiring urgent upper gastrointestinal endoscopy (EGD) include: gastrointestinal bleeding (melena, hematemesis), "
            "unintentional weight loss, progressive dysphagia or odynophagia, persistent vomiting, palpable abdominal mass, or new onset in patients > 55 years of age. "
            "Diagnostic testing for H. pylori includes urea breath test, stool antigen assay, or endoscopic biopsy. "
            "First-line treatment is proton pump inhibitor (PPI) therapy (omeprazole 20-40 mg daily) and H. pylori eradication therapy (bismuth quadruple therapy: "
            "PPI + bismuth subsalicylate + metronidazole + tetracycline for 14 days)."
        ),
    },
    {
        "id": "KB-GASTRO-002",
        "source": "WSES-ACG/appendicitis_pancreatitis",
        "title": "WSES/ACG Guideline: Acute Appendicitis and Acute Pancreatitis",
        "specialty": "gastroenterology",
        "year": 2024,
        "trust_score": 0.93,
        "text": (
            "Acute appendicitis begins with periumbilical visceral pain that migrates to the right lower quadrant (McBurney's point) within 12-24 hours. "
            "Physical exam reveals localized tenderness, involuntary guarding, rebound tenderness, Rovsing's sign, psoas sign, and obturator sign. "
            "Leukocytosis with left shift and elevated CRP are common. Contrast-enhanced abdominal CT confirms diagnosis (appendix diameter > 6 mm). "
            "Definitive treatment is urgent laparoscopic appendectomy and IV perioperative antibiotics. "
            "Acute pancreatitis presents with severe, constant, boring epigastric pain that radiates directly to the back, exacerbated by lying supine and relieved by leaning forward. "
            "Diagnosis requires 2 of 3 criteria: characteristic abdominal pain; serum lipase or amylase > 3 times the upper limit of normal; "
            "characteristic findings on contrast-enhanced CT or MRI. Etiologies: gallstones (40%), alcohol (35%), hypertriglyceridemia, hypercalcemia. "
            "Management: aggressive IV fluid resuscitation (lactated Ringer's), early enteral nutrition, and analgesia."
        ),
    },
    {
        "id": "KB-INFECT-001",
        "source": "NIH-WHO/covid19_management",
        "title": "NIH/WHO Guideline: COVID-19 Clinical Presentation and Antiviral Therapy",
        "specialty": "infectious_disease",
        "year": 2024,
        "trust_score": 0.94,
        "text": (
            "Coronavirus Disease 2019 (COVID-19), caused by SARS-CoV-2, presents with fever, dry cough, profound fatigue, dyspnea, sore throat, "
            "congestion, headache, myalgia, and sudden loss of taste (ageusia) or smell (anosmia). "
            "Gastrointestinal symptoms (nausea, diarrhea) occur in 10-20% of patients. "
            "Severe illness features hypoxemic respiratory failure (SpO2 < 94% on room air), ARDS, thromboembolism, and hyperinflammatory cytokine release. "
            "In outpatients with mild-to-moderate COVID-19 at high risk of progression to severe disease (age >= 65, diabetes, cardiovascular disease, obesity, immunosuppression), "
            "first-line treatment is oral nirmatrelvir/ritonavir (Paxlovid) initiated within 5 days of symptom onset. "
            "Hospitalized patients requiring supplemental oxygen receive dexamethasone (6 mg daily for up to 10 days) and remdesivir; "
            "baricitinib or tocilizumab is added for rapidly deteriorating patients with systemic inflammation."
        ),
    },
    {
        "id": "KB-INFECT-002",
        "source": "CDC/influenza_guideline",
        "title": "CDC Guideline: Seasonal Influenza Presentation and Antiviral Treatment",
        "specialty": "infectious_disease",
        "year": 2024,
        "trust_score": 0.94,
        "text": (
            "Seasonal Influenza (Flu) presents abruptly with high fever (38-40°C), severe myalgia, arthralgia, headache, prostration, "
            "non-productive dry cough, sore throat, and rhinitis. In contrast to the common cold, onset is sudden and systemic constitutional symptoms predominate. "
            "Complications include primary viral pneumonia, secondary bacterial pneumonia (Streptococcus pneumoniae, Staphylococcus aureus), "
            "myositis, myocarditis, and exacerbation of underlying chronic pulmonary or cardiac conditions. "
            "Antiviral treatment with neuraminidase inhibitors (oral oseltamivir 75 mg twice daily for 5 days) is recommended as early as possible "
            "for any patient with confirmed or suspected influenza who is hospitalized, has severe/complicated illness, or is at high risk for influenza complications "
            "(adults >= 65, young children, pregnant women, immunocompromised, chronic medical conditions). Greatest clinical benefit is achieved when started within 48 hours of symptom onset."
        ),
    },
    {
        "id": "KB-INFECT-003",
        "source": "IDSA/strep_and_cellulitis",
        "title": "IDSA Guideline: Streptococcal Pharyngitis and Skin/Soft Tissue Infections",
        "specialty": "infectious_disease",
        "year": 2023,
        "trust_score": 0.93,
        "text": (
            "Group A Streptococcal (GAS) pharyngitis presents with sudden-onset sore throat, pain on swallowing (odynophagia), fever, "
            "tonsillopharyngeal erythema with tonsillar exudates, and tender anterior cervical lymphadenopathy. "
            "Absence of viral symptoms (cough, rhinorrhea, conjunctivitis, hoarseness) strongly favors GAS (Centor criteria). "
            "First-line treatment is oral penicillin V or oral amoxicillin for 10 days to prevent acute rheumatic fever and peritonsillar abscess. "
            "Cellulitis is an acute, spreading bacterial infection of the dermis and subcutaneous tissue characterized by localized erythema, warmth, "
            "edema, and tenderness, most commonly caused by Streptococcus pyogenes or Staphylococcus aureus. "
            "Purulent cellulitis (abscess, furuncle) mandates incision and drainage with coverage for community-acquired MRSA (TMP-SMX, doxycycline, or vancomycin)."
        ),
    },
    {
        "id": "KB-RHEUM-001",
        "source": "ACR/gout_and_arthritis",
        "title": "ACR Guideline: Acute Gouty Arthritis and Hyperuricemia",
        "specialty": "rheumatology",
        "year": 2023,
        "trust_score": 0.93,
        "text": (
            "Gout is an inflammatory arthritis triggered by monosodium urate crystal deposition in articular and periarticular tissues. "
            "Acute gout attack presents with sudden, excruciating pain, erythema, warmth, and marked swelling, classically affecting the first metatarsophalangeal (MTP) joint (podagra) "
            "or ankles/knees. Symptoms frequently begin abruptly overnight. Definitive diagnosis requires arthrocentesis demonstrating needle-shaped, "
            "negatively birefringent monosodium urate crystals under polarized light microscopy. "
            "Acute flare pharmacotherapy: low-dose oral colchicine (1.2 mg at flare onset followed by 0.6 mg 1 hour later), oral NSAIDs (indomethacin, naproxen), "
            "or systemic corticosteroids (prednisone 30-40 mg daily) if NSAIDs/colchicine are contraindicated (renal failure, heart failure). "
            "Urate-lowering therapy with allopurinol (xanthine oxidase inhibitor) is indicated for recurrent flares, tophi, or radiographic joint damage."
        ),
    },
    {
        "id": "KB-HEMAT-001",
        "source": "ASH/anemia_guideline",
        "title": "ASH Guideline: Iron Deficiency Anemia and Microcytic Anemias",
        "specialty": "hematology",
        "year": 2023,
        "trust_score": 0.92,
        "text": (
            "Iron Deficiency Anemia (IDA) is the most common cause of anemia worldwide. "
            "Symptoms include chronic fatigue, generalized weakness, exertional dyspnea, dizziness, pallor, cold extremities, pica (craving ice/dirt), and brittle nails with spooning (koilonychia). "
            "Complete blood count demonstrates microcytic, hypochromic anemia (hemoglobin < 13 g/dL in men, < 12 g/dL in women; mean corpuscular volume MCV < 80 fL). "
            "Diagnostic iron studies confirm low serum ferritin (< 30 ng/mL, the most specific test), low serum iron, elevated total iron-binding capacity (TIBC), and transferrin saturation < 16%. "
            "In men and postmenopausal women, IDA mandates full gastrointestinal endoscopic evaluation (colonoscopy and EGD) to rule out occult gastrointestinal malignancy. "
            "Treatment involves oral ferrous sulfate (or modern alternate-day dosing) with ascorbic acid, or intravenous iron for intolerance or malabsorption."
        ),
    },
    {
        "id": "KB-PSYCH-001",
        "source": "APA-NICE/panic_disorder",
        "title": "APA/NICE Guideline: Panic Attacks and Panic Disorder",
        "specialty": "psychiatry",
        "year": 2023,
        "trust_score": 0.91,
        "text": (
            "A panic attack is a discrete, intense surge of fear or discomfort that reaches a peak within minutes, accompanied by at least 4 somatic/cognitive symptoms: "
            "palpitations or tachycardia, sweating, trembling/shaking, shortness of breath or smothering sensations, choking feeling, chest pain or discomfort, "
            "nausea or abdominal distress, dizziness/unsteadiness, derealization or depersonalization, fear of losing control, and fear of impending death. "
            "Crucially, life-threatening somatic emergencies (acute coronary syndrome, pulmonary embolism, arrhythmia, pheochromocytoma, thyroid storm) "
            "must be ruled out before attributing chest pain and dyspnea solely to a panic attack. "
            "First-line maintenance treatment for panic disorder includes selective serotonin reuptake inhibitors (SSRIs: sertraline, escitalopram) "
            "and cognitive behavioral therapy (CBT). Short-acting benzodiazepines may be used for immediate acute termination but carry dependence risks."
        ),
    },
    {
        "id": "KB-DERM-001",
        "source": "AAD-NICE/dermatology_guideline",
        "title": "AAD Guideline: Atopic Dermatitis, Shingles (Herpes Zoster), and Acute Rashes",
        "specialty": "dermatology",
        "year": 2023,
        "trust_score": 0.91,
        "text": (
            "Atopic dermatitis (eczema) is a chronic pruritic inflammatory skin disorder characterized by intensely itchy erythematous papules, "
            "excoriations, and lichenification, typically involving flexural surfaces (antecubital and popliteal fossae). Management centers on emollient barrier repair "
            "and topical corticosteroids for acute flares. "
            "Herpes Zoster (Shingles) results from reactivation of latent varicella-zoster virus (VZV) in sensory dorsal root ganglia. "
            "It presents with a prodrome of dermatomal pain, burning, or tingling followed by a unilateral, painful eruption of grouped vesicles on an erythematous base strictly limited to a single dermatome. "
            "Oral antiviral therapy with valacyclovir (1000 mg 3 times daily) or acyclovir initiated within 72 hours of rash onset accelerates lesion healing and reduces the incidence of postherpetic neuralgia. "
            "Involvement of the ophthalmic branch of the trigeminal nerve (Hutchinson's sign on nose tip) mandates urgent ophthalmology consult."
        ),
    },
    {
        "id": "KB-PED-001",
        "source": "AAP/pediatric_respiratory",
        "title": "AAP Guideline: Pediatric Croup, Bronchiolitis, and Febrile Illness",
        "specialty": "pediatrics",
        "year": 2024,
        "trust_score": 0.93,
        "text": (
            "Croup (laryngotracheobronchitis) is a viral respiratory illness in young children (6 months to 3 years) typically caused by parainfluenza virus. "
            "It presents with a distinctive 'seal-bark' cough, inspiratory stridor, hoarseness, and low-grade fever, often worsening at night. "
            "Single-dose oral dexamethasone (0.15-0.6 mg/kg) is standard first-line therapy for all severities; nebulized racemic epinephrine is added for moderate-to-severe stridor at rest. "
            "Bronchiolitis, commonly caused by Respiratory Syncytial Virus (RSV) in infants < 12 months, presents with rhinorrhea, tachypnea, wheezing, "
            "subcostal retractions, and poor feeding. Treatment is supportive: nasal suctioning, gentle hydration, and supplemental oxygen if SpO2 < 90%. "
            "Bronchodilators and systemic steroids are not routinely recommended in viral bronchiolitis."
        ),
    },
]


def build_unified_dataset() -> list[dict]:
    """Compile all guidelines, external clinical cases, and seed corpus into a single list."""
    from src.data.seed_corpus import SEED_CORPUS

    unified: list[dict] = []
    seen_texts: set[str] = set()

    # 1. Add all core comprehensive clinical guidelines
    for item in CORE_CLINICAL_GUIDELINES:
        txt = item["text"].strip()
        if txt not in seen_texts:
            seen_texts.add(txt)
            unified.append(item)

    # 2. Add all bundled seed corpus items (from seed_corpus.py)
    for idx, item in enumerate(SEED_CORPUS):
        txt = item["text"].strip()
        if txt not in seen_texts:
            seen_texts.add(txt)
            unified.append({
                "id": f"KB-SEED-{idx+1:03d}",
                "source": item.get("source", "seed_corpus"),
                "title": item.get("title", f"Clinical Guideline {idx+1}"),
                "specialty": item.get("specialty") or "general",
                "year": item.get("year", 2023),
                "trust_score": item.get("trust_score", 0.90),
                "text": txt,
            })

    # 3. Incorporate clinical case references from DDXPlus (first 100 high-yield pathology cases)
    ddx_path = DATA_DIR / "external" / "ddxplus.jsonl"
    if ddx_path.exists():
        count = 0
        for line in ddx_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or count >= 100:
                break
            try:
                item = json.loads(line)
                query = item.get("query", "").strip()
                dx = item.get("expected_dx") or item.get("PATHOLOGY", "")
                diff = item.get("differential", [])
                if query and dx:
                    diff_str = ", ".join(diff[:5]) if isinstance(diff, list) else str(diff)
                    text = (
                        f"Clinical Case Presentation: Patient presents with {query}. "
                        f"Confirmed primary pathology and diagnosis: {dx}. "
                        f"Key differential diagnoses to consider: {diff_str}. "
                        f"Requires systematic symptom evaluation, clinical history, and laboratory/imaging correlation."
                    )
                    if text not in seen_texts:
                        seen_texts.add(text)
                        unified.append({
                            "id": f"KB-DDX-{count+1:03d}",
                            "source": "DDXPlus/clinical_case",
                            "title": f"DDXPlus Reference Case: {dx}",
                            "specialty": "general",
                            "year": 2023,
                            "trust_score": 0.88,
                            "text": text,
                        })
                        count += 1
            except Exception:
                pass

    # 4. Incorporate clinical reference transcription notes from MTSamples (first 50)
    mts_path = DATA_DIR / "external" / "mtsamples.jsonl"
    if mts_path.exists():
        count = 0
        for line in mts_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or count >= 50:
                break
            try:
                item = json.loads(line)
                desc = item.get("query") or item.get("description", "")
                spec = item.get("specialty", "general")
                ctx = item.get("context", "")[:800]
                if desc and len(desc) > 20:
                    text = f"Clinical Transcription Note ({spec}): {desc}. Findings: {ctx}"
                    if text not in seen_texts:
                        seen_texts.add(text)
                        unified.append({
                            "id": f"KB-MTS-{count+1:03d}",
                            "source": f"MTSamples/{spec}",
                            "title": f"MTSamples Clinical Note: {desc[:60]}",
                            "specialty": spec.lower().replace(" ", "_"),
                            "year": 2023,
                            "trust_score": 0.85,
                            "text": text,
                        })
                        count += 1
            except Exception:
                pass

    return unified


def save_unified_dataset(records: list[dict], path: Path = OUTPUT_PATH) -> None:
    """Save records to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Successfully generated single unified dataset with {len(records)} records at: {path}")


if __name__ == "__main__":
    records = build_unified_dataset()
    save_unified_dataset(records)
