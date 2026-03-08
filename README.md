# Medical Insurance Cost Prediction Using ML with Explainable AI

## Paper
"Machine Learning for an Explainable Cost Prediction of Medical Insurance"
- Authors: Ugochukwu Orji & Elochukwu Ukwandu (2023)
- Published: Machine Learning with Applications, ScienceDirect
- Link: https://arxiv.org/abs/2311.14139

## Overview
This project applies the paper's methodology to predict US medical insurance
charges using three ensemble ML models (XGBoost, GBM, Random Forest)
and explains predictions using SHAP and ICE plots.

## Dataset
- Source: Kaggle — Medical Cost Personal Dataset
- Records: 1,338 rows, 6 features + 1 target (charges in USD)
- Features: age, sex, bmi, children, smoker, region
- Link: https://www.kaggle.com/datasets/mirichoi0218/insurance

## How to Run

### Jupyter Notebook
1. `pip install -r requirements.txt`
2. Download dataset and place CSV in `data/` folder
3. Open `notebooks/insurance_cost_prediction.ipynb`
4. Run all cells

### Streamlit App
1. Train and save models: `python src/train_and_save.py`
2. Launch the app: `streamlit run app.py`
3. Open the browser at the URL shown in the terminal

## Results
See `results/` folder for generated figures and tables.

## Key Findings
- Smoker status is the dominant cost driver across all models
- XGBoost and GBM achieved best R² (~82%) on test data
- SHAP analysis confirms age, BMI, and smoking as top determinants
- ICE plots reveal individual-level heterogeneity in feature effects
