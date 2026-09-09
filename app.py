"""
Streamlit Clinical Decision Support System (CDSS) App
for SEER Breast Cancer Prognosis & 5-Year Survival Prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="SEER Breast Cancer Prognosis CDSS",
    page_icon="🏥",
    layout="wide"
)

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'artifacts', 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'artifacts', 'plots')
REPORTS_DIR = os.path.join(BASE_DIR, 'artifacts', 'reports')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'seer_clean.csv')

@st.cache_data
def load_processed_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

@st.cache_resource
def load_models_and_preprocessor():
    prep_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
    preprocessor = joblib.load(prep_path) if os.path.exists(prep_path) else None
    
    models = {}
    model_files = {
        'Tuned XGBoost': 'tuned_xgboost.pkl',
        'Gradient Boosting': 'gradient_boosting.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl',
        'Logistic Regression': 'logistic_regression.pkl',
        'MLP Neural Network': 'mlp_neural_network.pkl'
    }
    
    for name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
            
    return models, preprocessor

df_clean = load_processed_data()
models, preprocessor = load_models_and_preprocessor()

# Header
st.title("🏥 Big Data Analytics for Breast Cancer Prognosis")
st.markdown("""
**Clinical Decision Support System (CDSS) Prototype using SEER Clinical Data**  
Developed for MSc Computing Research Project Dissertation.  
Predicting **5-Year Patient Survival Outcome (Alive vs High Mortality Risk)** based on 35,349 SEER cancer registry records.
""")

# Sidebar Controls
st.sidebar.header("⚙️ Configuration & Clinical Predictor")
selected_model_name = st.sidebar.selectbox("Select ML Model for Prediction:", list(models.keys()) if models else ["None"])

st.sidebar.subheader("Patient Clinical & Diagnostic Inputs")
input_age = st.sidebar.slider("Age at Diagnosis (Years):", 18, 95, 56, step=1)
input_nodes_pos = st.sidebar.slider("Regional Nodes Positive:", 0, 50, 2, step=1)
input_nodes_exam = st.sidebar.slider("Regional Nodes Examined:", 0, 60, 15, step=1)
input_tumor_size = st.sidebar.slider("CS Tumor Size (mm):", 1, 150, 25, step=1)
input_total_tumors = st.sidebar.slider("Total Tumor Count:", 1, 5, 1, step=1)

input_er = st.sidebar.selectbox("ER Status:", ["Positive", "Negative", "Unknown", "Borderline"])
input_pr = st.sidebar.selectbox("PR Status:", ["Positive", "Negative", "Unknown", "Borderline"])
input_her2 = st.sidebar.selectbox("HER2 Status:", ["Positive", "Negative", "Unknown", "Borderline"])

input_chemo = st.sidebar.selectbox("Chemotherapy Recode:", ["Yes", "No/Unknown"])
input_rad = st.sidebar.selectbox("Radiation Recode:", ["Beam radiation", "None/Unknown", "Radioactive implants (includes brachytherapy) (1988+)", "Radioisotopes (1988+)"])
input_race = st.sidebar.selectbox("Race:", ["White", "Black", "Unknown"])
input_sex = st.sidebar.selectbox("Sex:", ["Female", "Male"])

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 SEER Dataset & EDA", 
    "📈 Model Benchmarks & Metrics", 
    "🩺 Patient 5-Year Survival Predictor", 
    "🧬 Prognostic Biomarkers"
])

# TAB 1: EDA
with tab1:
    st.header("SEER Breast Cancer Registry Dataset (35,349 Patients)")
    if df_clean is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Patient Records", f"{len(df_clean):,}")
        c2.metric("5-Year Survival: Alive (Class 0)", f"{(df_clean['target']==0).sum():,} (66.2%)")
        c3.metric("5-Year Survival: Dead (Class 1)", f"{(df_clean['target']==1).sum():,} (33.8%)")
        
        st.subheader("Cleaned SEER Data Preview")
        st.dataframe(df_clean.head(10), use_container_width=True)
        
        st.subheader("Exploratory Data Analysis Visualizations")
        col_a, col_b, col_c = st.columns(3)
        p1 = os.path.join(PLOTS_DIR, 'target_distribution.png')
        p2 = os.path.join(PLOTS_DIR, 'age_survival_kde.png')
        p3 = os.path.join(PLOTS_DIR, 'nodes_boxplot.png')
        if os.path.exists(p1):
            col_a.image(p1, caption="5-Year Survival Outcome Distribution", use_container_width=True)
        if os.path.exists(p2):
            col_b.image(p2, caption="Age at Diagnosis vs Survival KDE", use_container_width=True)
        if os.path.exists(p3):
            col_c.image(p3, caption="Positive Lymph Nodes vs Survival Boxplot", use_container_width=True)

# TAB 2: BENCHMARKS
with tab2:
    st.header("10-Fold Stratified Cross-Validation & Test Set Metrics")
    cv_csv = os.path.join(REPORTS_DIR, 'cv_10fold_results.csv')
    test_csv = os.path.join(REPORTS_DIR, 'test_set_evaluation.csv')
    
    if os.path.exists(cv_csv):
        st.subheader("10-Fold Stratified CV Benchmark Table (28,279 Patients)")
        st.dataframe(pd.read_csv(cv_csv), use_container_width=True)
        
    if os.path.exists(test_csv):
        st.subheader("Unseen Holdout Test Set Evaluation Table (7,070 Patients)")
        st.dataframe(pd.read_csv(test_csv), use_container_width=True)
        
    st.subheader("Evaluation Visualizations")
    col_x, col_y = st.columns(2)
    roc_plot = os.path.join(PLOTS_DIR, 'roc_pr_curves.png')
    cm_plot = os.path.join(PLOTS_DIR, 'confusion_matrices.png')
    if os.path.exists(roc_plot):
        col_x.image(roc_plot, caption="ROC & Precision-Recall Curves", use_container_width=True)
    if os.path.exists(cm_plot):
        col_y.image(cm_plot, caption="Confusion Matrices Across Classifiers", use_container_width=True)

# TAB 3: PREDICTOR
with tab3:
    st.header("Real-Time Patient 5-Year Survival Risk Prediction")
    st.markdown("Adjust patient demographics, tumor characteristics, and treatment inputs in the sidebar to compute live 5-year prognosis risk.")
    
    if selected_model_name in models and preprocessor is not None:
        model = models[selected_model_name]
        
        sample_dict = {
            'Age at diagnosis': input_age,
            'Regional nodes positive (1988+)': input_nodes_pos,
            'Total number of in situ/malignant tumors for patient': input_total_tumors,
            'Radiation recode': input_rad,
            'Chemotherapy recode': input_chemo,
            'Radiation sequence with surgery': 'Radiation after surgery',
            'ER Status Recode Breast Cancer (1990+)': input_er,
            'PR Status Recode Breast Cancer (1990+)': input_pr,
            'Derived HER2 Recode (2010+)': input_her2,
            'Regional nodes examined (1988+)': input_nodes_exam,
            'Race recode': input_race,
            'Sex': input_sex,
            'Diagnosis_year': 2012,
            'CS_tumor_size_mm': input_tumor_size
        }
        
        sample_df = pd.DataFrame([sample_dict])
        processed_vec = preprocessor.transform(sample_df)
        
        prob_dead = model.predict_proba(processed_vec)[0, 1]
        prob_alive = 1.0 - prob_dead
        pred_class = model.predict(processed_vec)[0]
        
        st.subheader("Diagnostic Prognosis Output")
        r1, r2 = st.columns(2)
        
        with r1:
            if pred_class == 1 or prob_dead > 0.5:
                st.error("?? 5-Year Prognosis: HIGH MORTALITY RISK")
                st.metric("5-Year Mortality Risk Score", f"{prob_dead * 100:.1f}%")
                st.metric("Estimated 5-Year Survival Probability", f"{prob_alive * 100:.1f}%")
                st.warning("Clinical Recommendation: Intensive multi-modal treatment monitoring and oncology follow-up recommended.")
            else:
                st.success("? 5-Year Prognosis: FAVORABLE SURVIVAL OUTCOME")
                st.metric("Estimated 5-Year Survival Probability", f"{prob_alive * 100:.1f}%")
                st.metric("5-Year Mortality Risk Score", f"{prob_dead * 100:.1f}%")
                st.info("Clinical Recommendation: Routine post-treatment surveillance; high baseline survival likelihood.")
                
        with r2:
            st.progress(float(prob_dead))
            st.caption(f"5-Year Mortality Risk Gauge computed using {selected_model_name}")

# TAB 4: BIOMARKERS
with tab4:
    st.header("SEER Clinical Prognostic Biomarkers")
    st.markdown("Relative feature importance extracted from optimized tree ensemble models.")
    imp_plot = os.path.join(PLOTS_DIR, 'feature_importance.png')
    imp_csv = os.path.join(REPORTS_DIR, 'biomarker_importance_ranking.csv')
    
    if os.path.exists(imp_plot):
        st.image(imp_plot, caption="Top 15 Prognostic Biomarkers for 5-Year Survival", use_container_width=True)
    if os.path.exists(imp_csv):
        st.dataframe(pd.read_csv(imp_csv), use_container_width=True)
