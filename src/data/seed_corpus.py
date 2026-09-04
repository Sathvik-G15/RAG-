"""Seed medical corpus for testing without downloading datasets."""

from __future__ import annotations

SEED_CORPUS = [
    {
        "source": "WHO/cardiac_chest_pain",
        "title": "WHO Guideline: Management of Cardiac Chest Pain",
        "year": 2024,
        "trust_score": 0.95,
        "text": (
            "Chest pain is a cardinal symptom of acute coronary syndrome (ACS). "
            "Patients over 60 with chest pain, diaphoresis, and a history of diabetes "
            "or hypertension are at high risk for myocardial infarction. "
            "Immediate 12-lead ECG is recommended within 10 minutes of presentation. "
            "Elevated cardiac troponin confirms myocardial injury. "
            "Management includes aspirin 162-325 mg chewed, nitroglycerin if blood pressure "
            "permits, and urgent percutaneous coronary intervention for STEMI. "
            "Cardiac biomarkers should be measured at baseline and 3-6 hours later. "
            "Diabetic patients may present with atypical chest pain or silent ischemia."
        ),
    },
    {
        "source": "CDC/pneumonia_guideline",
        "title": "CDC Guideline: Community-Acquired Pneumonia",
        "year": 2024,
        "trust_score": 0.95,
        "text": (
            "Community-acquired pneumonia (CAP) presents with cough, fever, sputum production, "
            "dyspnea, and pleuritic chest pain. Physical exam may reveal crackles, "
            "bronchial breath sounds, and tactile fremitus. Chest X-ray showing lobar "
            "consolidation confirms diagnosis. The CURB-65 score (Confusion, Urea, "
            "Respiratory rate, Blood pressure, Age >= 65) stratifies severity. "
            "First-line outpatient treatment for healthy adults is amoxicillin or a "
            "macrolide. Hospitalized patients receive ceftriaxone plus azithromycin. "
            "Elderly patients (>65) with comorbidities such as diabetes have worse outcomes "
            "and warrant hospital admission. Vaccination with pneumococcal conjugate "
            "vaccine is recommended for adults 65 and older."
        ),
    },
    {
        "source": "NICE/diabetes_type2",
        "title": "NICE Guideline: Type 2 Diabetes Management",
        "year": 2023,
        "trust_score": 0.93,
        "text": (
            "Type 2 diabetes mellitus is characterized by insulin resistance and "
            "relative insulin deficiency. Diagnostic criteria include fasting plasma "
            "glucose >= 126 mg/dL, HbA1c >= 6.5%, or random glucose >= 200 mg/dL with "
            "symptoms. First-line pharmacotherapy is metformin for most patients. "
            "SGLT2 inhibitors or GLP-1 receptor agonists are preferred when cardiovascular "
            "disease or high risk exists. Blood pressure target is < 130/80 mmHg. "
            "Statins are recommended for patients 40-75 with diabetes. "
            "Annual screening for retinopathy, nephropathy, and neuropathy is mandatory. "
            "Metformin should be dose-adjusted in renal impairment (eGFR < 45)."
        ),
    },
    {
        "source": "NIH/hypertension_jnc8",
        "title": "NIH Guideline: Hypertension Management",
        "year": 2023,
        "trust_score": 0.93,
        "text": (
            "Hypertension is defined as sustained blood pressure >= 130/80 mmHg per "
            "current ACC/AHA guidelines. Stage 2 hypertension is >= 140/90 mmHg. "
            "First-line antihypertensives include thiazide diuretics, ACE inhibitors, "
            "angiotensin receptor blockers, and calcium channel blockers. "
            "For patients with diabetes and hypertension, ACE inhibitors or ARBs are "
            "preferred due to renal protective effects. Target blood pressure in "
            "diabetics is < 130/80 mmHg. Combination therapy is often required for "
            "achieving target. Lifestyle modifications: DASH diet, sodium restriction "
            "< 1500 mg/day, aerobic exercise 150 min/week, weight loss, alcohol moderation."
        ),
    },
    {
        "source": "WHO/copd_guideline",
        "title": "WHO Guideline: COPD Management",
        "year": 2024,
        "trust_score": 0.95,
        "text": (
            "Chronic obstructive pulmonary disease (COPD) is characterized by persistent "
            "respiratory symptoms and airflow limitation due to airway/alveolar abnormalities. "
            "Diagnosis requires spirometry showing FEV1/FVC < 0.70 post-bronchodilator. "
            "Symptoms include chronic cough, sputum, and progressive dyspnea. "
            "Smoking is the leading risk factor. GOLD classification stages A-D guide therapy. "
            "Initial therapy for Group D includes LAMA/LABA combination. "
            "Inhaled corticosteroids are added for exacerbations. "
            "Acute exacerbations present with increased dyspnea, sputum volume, or purulence. "
            "Treatment of exacerbations: short-acting bronchodilators, systemic corticosteroids "
            "(prednisone 40 mg for 5 days), and antibiotics if increased sputum purulence."
        ),
    },
    {
        "source": "CDC/asthma_guideline",
        "title": "CDC Guideline: Asthma Management",
        "year": 2024,
        "trust_score": 0.95,
        "text": (
            "Asthma is a chronic inflammatory airway disease with reversible airflow "
            "obstruction. Diagnosis rests on history of wheeze, dyspnea, chest tightness, "
            "and cough, plus reversible airflow limitation on spirometry (FEV1 improvement "
            ">= 12% and 200 mL post-bronchodilator). Stepwise treatment: Step 1 short-acting "
            "beta-agonist (SABA) PRN. Step 2 adds low-dose inhaled corticosteroid (ICS). "
            "Step 3 ICS plus LABA. Step 4 medium-dose ICS plus LABA. Step 5-high-dose ICS/LABA "
            "with or without biologic (omalizumab, mepolizumab). Asthma exacerbations: oxygen, "
            "SABA via nebulization, systemic corticosteroids, and consider magnesium sulfate "
            "in severe cases. Avoid triggers and ensure inhaler technique adherence."
        ),
    },
    {
        "source": "NICE/uti_guideline",
        "title": "NICE Guideline: Urinary Tract Infection",
        "year": 2023,
        "trust_score": 0.93,
        "text": (
            "Uncomplicated urinary tract infection (UTI) typically presents with dysuria, "
            "frequency, urgency, and suprapubic discomfort. Complicated UTI features fever, "
            "flank pain, or signs of pyelonephritis. Diagnosis is confirmed by positive "
            "urinalysis (nitrites, leukocyte esterase) and urine culture with >= 10^3 CFU/mL. "
            "First-line treatment for uncomplicated cystitis in non-pregnant women: "
            "nitrofurantoin 100 mg BD for 5 days, or trimethoprim 200 mg BD for 3 days. "
            "Pregnant women require 7-day course, avoiding trimethoprim in first trimester. "
            "Pyelonephritis: ceftriaxone IV or oral fluoroquinolone. Recurrent UTIs warrant "
            "investigation for structural or functional abnormalities."
        ),
    },
    {
        "source": "WHO/sepsis_guideline",
        "title": "WHO Guideline: Sepsis Management",
        "year": 2024,
        "trust_score": 0.95,
        "text": (
            "Sepsis is life-threatening organ dysfunction caused by dysregulated host "
            "response to infection. The SOFA score (Sequential Organ Failure Assessment) "
            "identifies organ dysfunction. Septic shock is sepsis with fluids unresponsive "
            "hypotension requiring vasopressors and elevated lactate > 2 mmol/L. "
            "Management follows Hour-1 bundle: measure lactate, obtain blood cultures, "
            "administer broad-spectrum antibiotics within 1 hour, begin crystalloid 30 mL/kg, "
            "and apply vasopressors if hypotensive. The qSOFA screen (altered mentation, "
            "RR >= 22, SBP <= 100) identifies high-risk patients. Lactate clearance and "
            "ongoing organ support in ICU reduce mortality. Early recognition is critical."
        ),
    },
    {
        "source": "PubMed/covid19_review",
        "title": "COVID-19: Diagnosis and Management Review",
        "year": 2023,
        "trust_score": 0.80,
        "text": (
            "COVID-19, caused by SARS-CoV-2, presents with fever, cough, dyspnea, fatigue, "
            "myalgia, and loss of taste/smell. Severe disease features hypoxia and bilateral "
            "infiltrates on chest imaging. Diagnosis confirmed by RT-PCR or rapid antigen testing. "
            "Risk factors for severe disease: age > 65, diabetes, hypertension, obesity, "
            "immunocompromised state. Mild disease is managed symptomatically. Moderate-severe "
            "disease requires antiviral therapy (nirmatrelvir-ritonavir in outpatients within 5 days), "
            "dexamethasone 6 mg for those requiring oxygen, and prophylactic anticoagulation. "
            "Long COVID characterized by persistent symptoms > 12 weeks warrants follow-up."
        ),
    },
    {
        "source": "NICE/hypothyroidism",
        "title": "NICE Guideline: Hypothyroidism",
        "year": 2023,
        "trust_score": 0.93,
        "text": (
            "Primary hypothyroidism is defined by elevated TSH and low free T4. Symptoms "
            "are insidious: fatigue, weight gain, cold intolerance, constipation, dry skin, "
            "bradycardia, and cognitive slowing. Most common cause in iodine-sufficient areas "
            "is Hashimoto's autoimmune thyroiditis, characterized by positive anti-TPO antibodies. "
            "Treatment is levothyroxine, dosed by weight (1.6 mcg/kg/day). Starting dose in "
            "younger healthy adults is full replacement; in elderly or cardiac disease, start "
            "low (25 mcg) and titrate every 4-6 weeks based on TSH. Target TSH is 0.4-4.0 mIU/L. "
            "Subclinical hypothyroidism (elevated TSH, normal T4) warrants treatment if "
            "TSH > 10 or symptomatic."
        ),
    },
    {
        "source": "WHO/stroke_guideline",
        "title": "WHO Guideline: Acute Stroke Management",
        "year": 2024,
        "trust_score": 0.95,
        "text": (
            "Acute stroke is sudden neurological deficit from cerebral ischemia or hemorrhage. "
            "The FAST acronym (Face, Arm, Speech, Time) aids recognition. Non-contrast CT "
            "head distinguishes ischemic from hemorrhagic stroke. Ischemic stroke treated "
            "with IV thrombolysis (alteplase) within 4.5 hours of symptom onset if no "
            "contraindications; mechanical thrombectomy for large vessel occlusion within "
            "6-24 hours. Blood pressure management: permissive hypertension (<= 220/120) "
            "for ischemic stroke; target SBP < 140 for hemorrhagic stroke. Secondary prevention "
            "includes antiplatelet therapy (aspirin 81 mg), statin, blood pressure control, "
            "and glycemic management in diabetics."
        ),
    },
    {
        "source": "PubMed/atk_practice",
        "title": "Primer on Acute Kidney Injury",
        "year": 2023,
        "trust_score": 0.80,
        "text": (
            "Acute kidney injury (AKI) is abrupt decrease in renal function. KDIGO criteria: "
            "rise in creatinine >= 0.3 mg/dL within 48h, or >= 1.5x baseline, or urine output "
            "< 0.5 mL/kg/h for 6 hours. Causes are pre-renal (hypovolemia, heart failure), "
            "intrinsic (acute tubular necrosis, glomerulonephritis), or post-renal (obstruction). "
            "Management: identify and treat underlying cause, restore intravascular volume, "
            "avoid nephrotoxins (NSAIDs, ACE inhibitors, contrast). Indications for dialysis "
            "(AEIOU): Acidosis, Electrolyte disturbances, Ingestion, Overload, Uremia. "
            "Patients with diabetic nephropathy are at higher risk of contrast-induced AKI;"
            "pre-hydration and low-contrast protocols are recommended."
        ),
    },
    {
        "source": "CDC/influenza",
        "title": "CDC Guideline: Seasonal Influenza",
        "year": 2024,
        "trust_score": 0.95,
        "text": (
            "Influenza is an acute viral respiratory illness caused by influenza A and B. "
            "Symptoms include abrupt onset fever, chills, myalgia, headache, sore throat, "
            "and cough. Diagnosis via rapid antigen testing or RT-PCR. Antiviral "
            "treatment (oseltamivir 75 mg BD for 5 days) started within 48 hours of "
            "symptom onset reduces severity and duration. Hospitalization is required "
            "for dyspnea, hypoxia, or dehydration. High-risk groups (age > 65, pregnancy, "
            "chronic lung/heart disease, diabetes) warrant empiric antiviral treatment "
            "regardless of symptom duration. Annual influenza vaccination is the most "
            "effective prevention. Elderly patients may have atypical presentations without fever."
        ),
    },
    {
        "source": "NICE/anaphylaxis",
        "title": "NICE Guideline: Anaphylaxis",
        "year": 2023,
        "trust_score": 0.93,
        "text": (
            "Anaphylaxis is a severe systemic allergic reaction with rapid onset and airway/"
            "breathing/circulation compromise. Common triggers: foods (peanuts, shellfish), "
            "medications (penicillin, NSAIDs), insect stings. Clinical features: urticaria, "
            "angioedema, bronchospasm, hypotension, hypoxia. Immediate treatment is "
            "intramuscular epinephrine 0.5 mg (adult), repeated every 5 minutes if "
            "unresponsive. Adjuncts: oxygen, IV fluids, H1/H2 antihistamines, corticosteroids. "
            "Patients on beta-blockers may be refractory to epinephrine; consider glucagon. "
            "Disposition: observation 6-12 hours for biphasic reaction risk. Confirm known "
            "allergies in patient history and avoid triggers."
        ),
    },
    {
        "source": "PubMed/gastritis",
        "title": "Review: Acute Gastritis and Peptic Ulcer Disease",
        "year": 2023,
        "trust_score": 0.80,
        "text": (
            "Acute gastritis presents with epigastric pain, nausea, and vomiting. Common "
            "causes: NSAID use, alcohol, Helicobacter pylori infection, and stress. "
            "Peptic ulcer disease (PUD) features epigastric pain relieved by food or antacids "
            "in duodenal ulcers, or worsened by food in gastric ulcers. Diagnosis: "
            "esophagogastroduodenoscopy (EGD), H. pylori testing (urea breath test, stool antigen). "
            "Treatment: proton pump inhibitor (omeprazole 20-40 mg) for 4-8 weeks. "
            "H. pylori eradication: triple therapy (PPI + amoxicillin + clarithromycin) for 14 days. "
            "Stop NSAIDs; consider misoprostol for NSAID users. Seek urgent care if hematemesis, "
            "melena, or severe pain (possible perforation)."
        ),
    },
    {
        "source": "CDC/cardiac_biomarkers",
        "title": "CDC: Cardiac Biomarkers in Acute Coronary Syndrome",
        "year": 2024,
        "trust_score": 0.95,
        "text": (
            "In patients presenting with chest pain suspicious for acute coronary syndrome, "
            "high-sensitivity cardiac troponin is the biomarker of choice. Obtain a 12-lead ECG "
            "within 10 minutes. Serial troponin measurements at 0 and 3 hours (or 1-2 hours with "
            "high-sensitivity assays) enable rapid rule-out or rule-in of myocardial infarction. "
            "A rise and/or fall in troponin with at least one value above the 99th percentile "
            "upper reference limit indicates myocardial injury. Ischemia is suggested by symptoms "
            "or ECG changes. Coronary angiography is indicated for non-ST elevation ACS with "
            "high-risk features such as dynamic ECG changes, elevated troponin, or hemodynamic "
            "instability. Patients with diabetes often have silent or atypical ischemia."
        ),
    },
    {
        "source": "WHO/pneumonia_hospital",
        "title": "WHO: Hospital Management of Severe Pneumonia",
        "year": 2024,
        "trust_score": 0.95,
        "text": (
            "Severe community-acquired pneumonia requires hospital admission and empiric "
            "antibiotics. Empiric coverage for hospitalized patients includes a beta-lactam "
            "plus a macrolide (e.g., ceftriaxone 1 g IV daily plus azithromycin 500 mg IV daily). "
            "For patients with risk factors for resistant organisms, broader coverage is needed. "
            "Supportive care includes oxygen for hypoxia, intravenous fluids, and antipyretics. "
            "Assess CURB-65: confusion, urea > 7 mmol/L, respiratory rate >= 30, blood pressure "
            "low, and age >= 65. A score of 3 or more suggests severe disease and ICU admission "
            "should be considered. Monitor for treatment failure if fever persists beyond 72 hours."
        ),
    },
    {
        "source": "PubMed/uti_recurrent",
        "title": "Recurrent and Complicated Urinary Tract Infections",
        "year": 2023,
        "trust_score": 0.80,
        "text": (
            "Urinary tract infection is diagnosed by symptoms of dysuria, frequency, urgency, "
            "and a positive urine culture. Uncomplicated cystitis in young women is usually "
            "caused by Escherichia coli. First-line short-course therapy includes nitrofurantoin "
            "or trimethoprim-sulfamethoxazole. Pyelonephritis, indicated by fever and flank pain, "
            "requires 7-14 days of treatment and typically a fluoroquinolone or ceftriaxone. "
            "Complicated UTI risk factors include pregnancy, diabetes, immunosuppression, and "
            "urinary obstruction. In men, UTI suggests possible prostatitis and warrants longer "
            "therapy. Recurrent cystitis in women can be managed with post-coital prophylaxis "
            "or continuous low-dose antibiotics. Hydration and voiding after intercourse reduce "
            "recurrence risk."
        ),
    },
    {
        "source": "PubMed/copd_exacerbation",
        "title": "Management of COPD Exacerbations",
        "year": 2023,
        "trust_score": 0.80,
        "text": (
            "An acute exacerbation of COPD is an acute worsening of respiratory symptoms "
            "that requires additional therapy. Cardinal symptoms are increased dyspnea, "
            "increased sputum volume, and increased sputum purulence. Treatment includes "
            "short-acting bronchodilators (inhaled beta-agonists and anticholinergics), "
            "systemic corticosteroids such as prednisone 40 mg daily for five days, and "
            "antibiotics when sputum is purulent. Oxygen therapy titrated to 88-92% "
            "saturation prevents hypercapnia. Noninvasive ventilation (NIV) reduces "
            "intubation in acute hypercapnic respiratory failure. Hospitalization is "
            "warranted for severe dyspnea, altered mental status, or hypoxia. Smoking "
            "cessation and vaccination reduce future exacerbations."
        ),
    },
    {
        "source": "NICE/anaphylaxis_management",
        "title": "NICE: Anaphylaxis Emergency Treatment",
        "year": 2023,
        "trust_score": 0.93,
        "text": (
            "Anaphylaxis is an acute, life-threatening systemic reaction. First-line treatment "
            "is intramuscular epinephrine 0.5 mg in adults and 0.01 mg/kg in children, "
            "repeated every 5 minutes if no improvement. Place the patient supine with legs "
            "elevated unless airway compromise. Administer high-flow oxygen and rapid IV fluid "
            "bolus for hypotension. Antihistamines (H1 and H2) and corticosteroids are "
            "adjunctive, not life-saving. Beta-blocked patients may be epinephrine-resistant; "
            "consider glucagon. Observe all patients for at least 6 hours due to risk of "
            "biphasic reactions. Discharge with an epinephrine auto-injector and an "
            "allergist referral."
        ),
    },
    {
        "source": "PubMed/hypothyroid_update",
        "title": "Update: Levothyroxine Dosing in Hypothyroidism",
        "year": 2023,
        "trust_score": 0.80,
        "text": (
            "Primary hypothyroidism is confirmed by a persistently elevated TSH with a low "
            "free thyroxine (T4). Typical symptoms include fatigue, weight gain, cold "
            "intolerance, constipation, and cognitive slowing. Treatment is levothyroxine, "
            "approximately 1.6 mcg/kg/day. In healthy adults under 60, start at the full "
            "estimated replacement dose. In patients over 60 or with cardiac disease, start "
            "at 25-50 mcg and titrate slowly to avoid precipitating arrhythmia. Recheck TSH "
            "in 6-8 weeks and adjust by 12.5-25 mcg increments. In pregnancy, levothyroxine "
            "requirements increase by about 30%. Subclinical hypothyroidism (TSH 4-10 with "
            "normal T4) is treated when TSH is above 10 or when symptoms or pregnancy are present."
        ),
    },
    {
        "source": "WHO/stroke_thrombolysis",
        "title": "WHO: IV Thrombolysis for Acute Ischemic Stroke",
        "year": 2024,
        "trust_score": 0.95,
        "text": (
            "Acute ischemic stroke is treated with intravenous alteplase within 4.5 hours of "
            "last-known-well, provided no contraindications. Rapid recognition via the FAST "
            "screen (face, arm, speech, time) and immediate brain imaging are essential. "
            "Mechanical thrombectomy is offered for large vessel occlusion up to 24 hours. "
            "Blood pressure is managed to less than 185/110 mmHg before thrombolysis. "
            "Exclude hemorrhagic stroke with non-contrast CT before treatment. Antiplatelets "
            "should be avoided for 24 hours after alteplase. Early dysphagia screening "
            "reduces aspiration pneumonia. Secondary prevention includes antiplatelets, "
            "statins, blood pressure control, and anticoagulation for atrial fibrillation."
        ),
    },
    {
        "source": "NICE/pyelonephritis",
        "title": "NICE: Pyelonephritis Management",
        "year": 2023,
        "trust_score": 0.93,
        "text": (
            "Pyelonephritis is an infection of the renal pelvis and kidney, typically "
            "presenting with fever, chills, flank pain, nausea, and dysuria. Urinalysis "
            "shows pyuria and often nitrites. A urine culture confirms the organism; "
            "blood cultures are reserved for severe disease. Outpatient treatment for "
            "uncomplicated pyelonephritis is oral ciprofloxacin 500 mg twice daily or "
            "levofloxacin 750 mg daily for 7 days. Pregnant patients and those who are "
            "septic require hospitalization with IV ceftriaxone. Assess for complicating "
            "factors such as obstruction or stones when fever persists beyond 72 hours. "
            "Analgesia and hydration are supportive. Diabetes increases risk of "
            "complicated infection and bacteremia."
        ),
    },
]


SEED_PATIENT_QUERIES = [
    {
        "query": "55-year-old diabetic male presents with central chest pain, sweating, and shortness of breath.",
        "expected_dx": "acute_coronary_syndrome",
        "complexity": "complex",
        "correct_k": 15,
    },
    {
        "query": "70-year-old male with productive cough, fever 39C, and crackles on auscultation.",
        "expected_dx": "community_acquired_pneumonia",
        "complexity": "medium",
        "correct_k": 7,
    },
    {
        "query": "45-year-old woman with dysuria and urinary frequency without flank pain.",
        "expected_dx": "uncomplicated_uti",
        "complexity": "simple",
        "correct_k": 3,
    },
    {
        "query": "60-year-old smoker with worsening dyspnea and wheeze; baseline COPD.",
        "expected_dx": "copd_exacerbation",
        "complexity": "medium",
        "correct_k": 7,
    },
    {
        "query": "30-year-old woman with urticaria, lip swelling, and hypotension after receiving penicillin.",
        "expected_dx": "anaphylaxis",
        "complexity": "complex",
        "correct_k": 10,
    },
    {
        "query": "32-year-old woman with fatigue, weight gain, cold intolerance, and TSH of 8.5.",
        "expected_dx": "hypothyroidism",
        "complexity": "medium",
        "correct_k": 5,
    },
    {
        "query": "65-year-old man with sudden right-sided weakness and slurred speech.",
        "expected_dx": "acute_ischemic_stroke",
        "complexity": "complex",
        "correct_k": 10,
    },
    {
        "query": "50-year-old diabetic with fever, flank pain, and dysuria.",
        "expected_dx": "pyelonephritis",
        "complexity": "medium",
        "correct_k": 7,
    },
]


def get_seed_corpus():
    return SEED_CORPUS


def get_seed_queries():
    return SEED_PATIENT_QUERIES
