# Big Data Analytics for Breast Cancer Prognosis: A Machine Learning-Based Approach Using SEER Data
### MSc Computing Research Project - Code, Analytics & Dissertation Deliverable

---

## ?? Project Overview
This repository contains the complete, publication-grade machine learning and big data analytics codebase for the research project:
**"Big Data Analytics for Breast Cancer Prognosis: A Machine Learning-Based Approach Using SEER Data"**, developed in strict compliance with the **MSc Computing Research Project Handbook (May 2026)** guidelines.

### Research Question:
> *"To what extent can machine learning techniques applied to large-scale breast cancer clinical data predict patient outcomes?"*

### Dataset Specification:
- **Source**: Surveillance, Epidemiology, and End Results (SEER) Program (`Query_5_years.xlsx`)
- **Patient Cohort**: **35,349 records** (Alive = 23,404 [66.2%], Dead = 11,945 [33.8%])
- **Attributes**: 19 clinical, demographic, histopathological, and treatment features

---

## ?? Project Architecture & Files

```
breast_cancer_seer_prognosis/
+-- seer_breast_cancer_prognosis.ipynb   # Primary Jupyter Notebook deliverable
+-- app.py                              # Interactive Streamlit Web App (Clinical CDSS Demo)
+-- run_pipeline.py                     # Master automated execution script
+-- build_seer_notebook.py              # Generator script for Jupyter Notebook
+-- README.md                           # Project documentation & execution guide
+-- dissertation_report.docx            # 3,570-word Dissertation Report (Word Document)
+-- dissertation_report.md              # 3,570-word Dissertation Report (Markdown Version)
+-- data/
¦   +-- raw/
¦   ¦   +-- Query_5_years.xlsx          # Raw SEER dataset (35,349 patient records)
¦   +-- processed/
¦       +-- seer_clean.csv              # Cleaned, encoded, leakage-free dataset
+-- src/
¦   +-- data_loader.py                  # Ingestion, data hygiene & 80/20 stratified split
¦   +-- eda.py                          # Exploratory data analysis & distribution plotting
¦   +-- feature_engineering.py          # One-Hot Encoding, Z-score scaling & PCA projections
¦   +-- model_trainer.py                # 8 ML classifiers + 10-Fold Stratified CV & Grid Search
¦   +-- evaluator.py                    # Metric benchmarking, confusion matrices & ROC curves
¦   +-- explainability.py               # Feature importance ranking & clinical biomarkers
+-- artifacts/
    +-- models/                         # Saved pre-trained model binaries (.pkl)
    +-- plots/                          # Publication-grade PNG figures for thesis & slides
    +-- reports/                        # Metric CSV tables for Appendix G
```

---

## ?? How to Run the Deliverables

### 1. Running the Jupyter Notebook (`.ipynb`)
Open VS Code, Jupyter Lab, or Antigravity and select:
`seer_breast_cancer_prognosis.ipynb`

Run all cells sequentially. The notebook features:
- **Markdown Narrative**: Research context, mathematical equations, and academic interpretations preceding every code block.
- **Line-by-Line Code Comments**: Explaining preprocessing, cross-validation, and visualization steps.

### 2. Launching the Interactive Clinical Web App (`app.py`)
Open your terminal in the project directory and run:

```powershell
python -m streamlit run app.py
```

This launches an interactive Clinical Decision Support System at `http://localhost:8501`:
- **Tab 1: SEER Dataset & EDA**: Cohort statistics, target distribution, age density curves, lymph node boxplots.
- **Tab 2: Model Benchmarks**: 10-Fold CV tables, holdout test metrics, confusion matrix grids, ROC/PR curves.
- **Tab 3: Patient 5-Year Survival Predictor**: Interactive sidebar controls for age, tumor size, positive nodes, hormone receptor status (ER, PR, HER2), and treatments to calculate real-time 5-year mortality risk scores.
- **Tab 4: Prognostic Biomarkers**: Top 15 SEER clinical biomarker feature importances.

### 3. Running the Master Pipeline (`run_pipeline.py`)
To re-run data ingestion, 10-fold CV, holdout test evaluation, figure generation, and CSV table exports in a single command:

```powershell
python run_pipeline.py
```

---

## ?? Performance Benchmarks (35,349 Patients)
- **10-Fold Stratified Cross-Validation**: Tuned XGBoost, Gradient Boosting, and Random Forest achieved **ROC-AUC > 0.935** on 28,279 training patients.
- **7,070 Unseen Test Patients**: Tuned XGBoost achieved **88.59% Accuracy**, **74.88% Sensitivity**, **95.58% Specificity**, and **0.9359 ROC-AUC**.
- **Top Prognostic Biomarkers**: `Diagnosis_year` (staging advances), `Regional nodes positive`, `Radiation sequence with surgery`, `PR Status Positive`, `Age at diagnosis`, `CS tumor size`, `ER Status`.

---
*Created for MSc Computing Research Project Dissertation (Student: Alan Paulose, Supervisor: Joshua Thompson).*
