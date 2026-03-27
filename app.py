import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


@st.cache_resource
def load_models():
    models = {}
    for name in ['RF', 'XGBoost', 'GBM']:
        path = os.path.join(MODELS_DIR, f'{name}_model.pkl')
        if os.path.exists(path):
            models[name] = joblib.load(path)
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
    return models, scaler, feature_names


st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

st.title("Medical Insurance Cost Prediction")
st.caption("Predicting US medical insurance charges using ensemble ML models with explainable AI (SHAP & ICE)")

models, scaler, feature_names = load_models()

tab1, tab2 = st.tabs(["Predict Cost", "Model Dashboard"])

# ===================== TAB 1: PREDICTION =====================
with tab1:
    col_input, col_result = st.columns([1, 1.2])

    with col_input:
        st.subheader("Patient Details")

        age = st.slider("Age", min_value=18, max_value=64, value=30)
        sex = st.selectbox("Sex", ["Male", "Female"])

        c1, c2 = st.columns(2)
        with c1:
            height_ft = st.number_input("Height (ft)", min_value=4, max_value=7, value=5)
        with c2:
            height_in = st.number_input("Height (in)", min_value=0, max_value=11, value=9)
        weight_lbs = st.slider("Weight (lbs)", min_value=90, max_value=330, value=155)

        # Auto-calculate BMI: weight (lbs) / [height (in)]² × 703
        total_height_inches = height_ft * 12 + height_in
        bmi = (weight_lbs / (total_height_inches ** 2)) * 703
        st.info(f"Calculated BMI: **{bmi:.1f}**")

        children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
        smoker = st.selectbox("Smoker", ["No", "Yes"])
        region = st.selectbox("Region", ["Southwest", "Southeast", "Northwest", "Northeast"])

        model_choice = st.selectbox("Select Model", list(models.keys()))

        predict_btn = st.button("Predict Cost", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            # Build input matching the training feature order
            input_data = pd.DataFrame([{
                'age': age,
                'sex': 1 if sex == "Male" else 0,
                'bmi': bmi,
                'children': children,
                'smoker': 1 if smoker == "Yes" else 0,
                'region_northwest': 1 if region == "Northwest" else 0,
                'region_southeast': 1 if region == "Southeast" else 0,
                'region_southwest': 1 if region == "Southwest" else 0,
            }])

            # Ensure column order matches training
            input_data = input_data[feature_names]

            # Scale
            input_scaled = pd.DataFrame(
                scaler.transform(input_data),
                columns=feature_names
            )

            # Predict
            model = models[model_choice]
            prediction = model.predict(input_scaled)[0]

            # Display results
            st.subheader("Prediction Result")
            m1, m2 = st.columns(2)
            m1.metric("Predicted Annual Cost", f"${prediction:,.2f}")
            m2.metric("BMI", f"{bmi:.1f}")

            # SHAP explanation
            st.subheader("Why this cost?")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_scaled)

            fig, ax = plt.subplots(figsize=(8, 5))
            shap.waterfall_plot(shap_values[0], show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Fill in patient details and click **Predict Cost** to see results.")

# ===================== TAB 2: DASHBOARD =====================
with tab2:
    st.subheader("Model Performance Comparison")

    perf_path = os.path.join(RESULTS_DIR, 'tables', 'model_performance.csv')
    impr_path = os.path.join(RESULTS_DIR, 'tables', 'improvement_table.csv')

    if os.path.exists(perf_path):
        perf_df = pd.read_csv(perf_path)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Run the notebook first to generate performance tables.")

    if os.path.exists(impr_path):
        st.subheader("Improvement After Hyperparameter Tuning")
        impr_df = pd.read_csv(impr_path)
        st.dataframe(impr_df, use_container_width=True, hide_index=True)

    figures_dir = os.path.join(RESULTS_DIR, 'figures')
    if os.path.isdir(figures_dir) and os.listdir(figures_dir):
        st.subheader("Visualizations")

        figure_groups = {
            "Correlation Heatmap": ["correlation_heatmap.png"],
            "Features vs Charges": ["features_vs_charges.png"],
            "Learning Curves": ["learning_curves.png"],
            "Residual Plots": ["residual_plots.png"],
            "Prediction Error": ["prediction_error.png"],
            "SHAP Summary": [f"shap_summary_{m}.png" for m in ["RF", "XGBoost", "GBM"]],
            "SHAP Importance": [f"shap_importance_{m}.png" for m in ["RF", "XGBoost", "GBM"]],
            "ICE Plots": [f"ice_plots_{m}.png" for m in ["RF", "XGBoost", "GBM"]],
        }

        for group_name, filenames in figure_groups.items():
            existing = [f for f in filenames if os.path.exists(os.path.join(figures_dir, f))]
            if existing:
                with st.expander(group_name, expanded=False):
                    for fname in existing:
                        st.image(os.path.join(figures_dir, fname), use_container_width=True)
    else:
        st.info("Run the notebook to generate figures, then they will appear here.")
