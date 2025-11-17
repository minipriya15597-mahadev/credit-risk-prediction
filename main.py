# Interpretable ML – SHAP Credit Risk Prediction


import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from xgboost import XGBClassifier

# STEP 1: Load Dataset


# Example: Replace with your file name
df = pd.read_csv("credit_data.csv")

print("Initial Data Shape:", df.shape)
print(df.head())

# STEP 2: Basic Preprocessing

# 1. Handle missing values
df.fillna(df.median(), inplace=True)

# 2. Convert categorical columns to category codes
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].astype('category').cat.codes


# STEP 3: Feature Engineering


# Example domain-specific features (modify for your dataset)
df["credit_utilization"] = df["loan_amnt"] / (df["annual_inc"] + 1)
df["income_to_loan_ratio"] = df["annual_inc"] / (df["loan_amnt"] + 1)
df["dti_scaled"] = df["dti"] / 100


# STEP 4: Train/Test Split


X = df.drop("loan_status", axis=1)   # target column name may differ
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Standardizing numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# STEP 5: Model Training (XGBoost works well with SHAP)


model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)

model.fit(X_train_scaled, y_train)

# STEP 6: Model Performance Metrics


y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred))

print("AUC Score:", roc_auc_score(y_test, y_prob))
print("F1 Score:", f1_score(y_test, y_pred))

# STEP 7: SHAP Interpretability


print("\nGenerating SHAP values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_scaled)

# --- Global Interpretation (Feature Importance) ---
print("\nShowing SHAP Summary Plot...")
shap.summary_plot(shap_values, X_train, show=True)

# --- Dependence Plot (Feature interaction) ---
shap.dependence_plot("credit_utilization", shap_values, X_train)

# --- Local Explanation for a single prediction ---
sample_idx = 0
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_idx],
    X_train.iloc[sample_idx, :],
    matplotlib=True
)

# Save summary plot
plt.savefig("shap_summary_plot.png")
