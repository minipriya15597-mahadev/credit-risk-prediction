"""
Readable, step-by-step credit risk pipeline with SHAP explanations.

This script is intentionally organized in a very “human-friendly” way:
- Each major task gets its own well-named helper with docstrings + comments.
- Control flow inside `main()` mirrors the Cultus project checklist.
- Outputs (metrics, SHAP artefacts, narrative report) are written to `artifacts/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# --------------------------------------------------------------------------- #
# Project-level constants
# --------------------------------------------------------------------------- #
DATA_PATH = Path("credit_data.csv")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


@dataclass
class ModelResult:
    """Container that keeps metrics and estimator together."""

    name: str
    auc: float
    f1: float
    report: str
    estimator: object


# --------------------------------------------------------------------------- #
# Data preparation helpers
# --------------------------------------------------------------------------- #
def load_dataset(path: Path) -> pd.DataFrame:
    """Read the CSV that Cultus provided."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path.resolve()}.\n"
            "Download the official credit dataset and place it here."
        )
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create domain-specific ratios so the model reflects underwriting logic."""
    engineered: List[str] = []

    if {"loan_amnt", "annual_inc"}.issubset(df.columns):
        df["credit_utilization"] = df["loan_amnt"] / (df["annual_inc"] + 1)
        df["income_to_loan_ratio"] = df["annual_inc"] / (df["loan_amnt"] + 1)
        df["loan_to_income_pct"] = df["credit_utilization"] * 100
        df["income_buffer"] = df["annual_inc"] - df["loan_amnt"]
        engineered += [
            "credit_utilization",
            "income_to_loan_ratio",
            "loan_to_income_pct",
            "income_buffer",
        ]

    if "dti" in df.columns:
        df["dti_scaled"] = df["dti"] / 100
        engineered.append("dti_scaled")

    if {"fico_range_low", "fico_range_high"}.issubset(df.columns):
        df["fico_mean"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
        engineered.append("fico_mean")

    return df, engineered


def preprocess(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Clean missing values, encode categoricals, and split out the target."""
    cleaned = df.copy()

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in cleaned.columns if c not in numeric_cols]

    for col in numeric_cols:
        cleaned.loc[:, col] = cleaned[col].fillna(cleaned[col].median())

    for col in categorical_cols:
        cleaned.loc[:, col] = cleaned[col].fillna(cleaned[col].mode().iloc[0])
        cleaned.loc[:, col] = cleaned[col].astype("category").cat.codes

    cleaned, engineered = engineer_features(cleaned)

    if target_col not in cleaned.columns:
        raise KeyError(f"Missing target column '{target_col}' in dataset.")

    if cleaned[target_col].dtype == object:
        cleaned[target_col] = cleaned[target_col].astype("category").cat.codes

    y = cleaned[target_col].astype(int)
    X = cleaned.drop(columns=[target_col])
    return X, y, engineered


# --------------------------------------------------------------------------- #
# Modelling helpers
# --------------------------------------------------------------------------- #
def candidate_models(random_state: int = 42) -> Dict[str, object]:
    """Four reasonable baselines covering linear + tree ensembles."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1500, solver="liblinear", class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=random_state,
        ),
    }


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test) -> ModelResult:
    """Train a model and compute the AUC/F1 scores + classification report."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    return ModelResult(name=name, auc=auc, f1=f1, report=report, estimator=model)


def run_model_tournament(
    models: Dict[str, object], X_train, X_test, y_train, y_test
) -> Tuple[List[ModelResult], ModelResult]:
    """Compare all candidates and choose the strongest according to AUC then F1."""
    results = [evaluate_model(name, mdl, X_train, X_test, y_train, y_test) for name, mdl in models.items()]
    results.sort(key=lambda r: (r.auc, r.f1), reverse=True)

    # Persist a CSV that is easy to include in the final submission
    metrics_df = pd.DataFrame(
        [{"model": r.name, "auc": r.auc, "f1": r.f1, "precision_recall": r.report} for r in results]
    )
    metrics_path = ARTIFACT_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved model comparison metrics to {metrics_path.resolve()}")

    return results, results[0]


# --------------------------------------------------------------------------- #
# SHAP + storytelling helpers
# --------------------------------------------------------------------------- #
def _ensure_2d(array_like) -> np.ndarray:
    arr = np.array(array_like)
    if arr.ndim == 3:  # sometimes (n_samples, n_features, 2) for tree boosters
        arr = arr[:, :, -1]
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def build_shap_runner(model_name: str, model, background: pd.DataFrame):
    if model_name in {"random_forest", "gradient_boosting", "xgboost"}:
        return shap.TreeExplainer(model)
    if model_name == "logistic_regression":
        return shap.LinearExplainer(model, background)
    return shap.KernelExplainer(model.predict_proba, background)


def generate_shap_outputs(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Dict[str, object]:
    """Create global plots plus per-row SHAP values for later storytelling."""
    background = X_train.sample(min(len(X_train), 500), random_state=42)
    explainer = build_shap_runner(model_name, model, background)

    shap_train = _ensure_2d(explainer.shap_values(X_train))
    shap_test = _ensure_2d(explainer.shap_values(X_test))
    ranking = pd.Series(np.abs(shap_train).mean(axis=0), index=X_train.columns).sort_values(ascending=False)

    summary_path = ARTIFACT_DIR / "shap_summary.png"
    shap.summary_plot(shap_train, X_train, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200)
    plt.close()

    top_feature = ranking.index[0]
    dependence_path = ARTIFACT_DIR / f"shap_dependence_{top_feature}.png"
    shap.dependence_plot(top_feature, shap_train, X_train, show=False)
    plt.tight_layout()
    plt.savefig(dependence_path, dpi=200)
    plt.close()

    return {
        "explainer": explainer,
        "shap_train": shap_train,
        "shap_test": shap_test,
        "ranking": ranking,
        "summary_plot": summary_path,
        "dependence_plot": dependence_path,
    }


def narrate_local_cases(shap_values: np.ndarray, data: pd.DataFrame, probs: np.ndarray, label: str) -> List[str]:
    """Produce five bullet-sized stories for high-risk and low-risk applicants."""
    if len(data) == 0:
        return []

    order = np.argsort(probs)
    if label == "high-risk":
        order = order[::-1]
    selected = order[: min(5, len(order))]

    notes: List[str] = []
    for idx in selected:
        feature_impacts = sorted(
            zip(data.columns, shap_values[idx], data.iloc[idx]), key=lambda row: abs(row[1]), reverse=True
        )[:3]
        sentence = "; ".join(
            f"{feat}={val:.2f} {'raises' if shap_val > 0 else 'lowers'} risk by {abs(shap_val):.3f}"
            for feat, shap_val, val in feature_impacts
        )
        notes.append(f"{label.title()} case (score {probs[idx]:.2f}): {sentence}")
    return notes


def simple_feature_importance(model, feature_names: List[str]) -> pd.Series:
    """Fallback importance metric for the executive summary."""
    if hasattr(model, "feature_importances_"):
        raw = np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        raw = np.abs(np.array(model.coef_)).ravel()
    else:
        raw = np.zeros(len(feature_names))
    return pd.Series(raw, index=feature_names).sort_values(ascending=False)


def write_report(
    df_shape: Tuple[int, int],
    engineered: List[str],
    results: List[ModelResult],
    best: ModelResult,
    shap_bundle: Dict[str, object],
    model_importance: pd.Series,
    high_stories: List[str],
    low_stories: List[str],
) -> Path:
    """Create the narrative text deliverable requested by Cultus."""
    report_lines = [
        "Cultus Credit Risk Project Report",
        "=" * 34,
        f"Dataset shape: {df_shape[0]} rows x {df_shape[1]} columns",
        f"Engineered features: {', '.join(engineered) if engineered else 'None'}",
        "",
        "Model comparison (sorted by AUC, then F1):",
    ]
    for r in results:
        report_lines.append(f"- {r.name}: AUC={r.auc:.3f}, F1={r.f1:.3f}")

    report_lines += [
        "",
        f"Best model: {best.name}",
        "Classification report:",
        best.report,
        "",
        "Top SHAP drivers:",
        shap_bundle["ranking"].head(5).to_string(float_format=lambda x: f"{x:.4f}"),
        "",
        "Local explanations – high risk:",
    ]
    report_lines += [f"- {line}" for line in (high_stories or ["Not enough samples"])]

    report_lines += ["", "Local explanations – low risk:"]
    report_lines += [f"- {line}" for line in (low_stories or ["Not enough samples"])]

    exec_summary = []
    for idx, feat in enumerate(shap_bundle["ranking"].head(3).index, start=1):
        direction = "higher" if model_importance.get(feat, 0) >= 0 else "higher"
        exec_summary.append(f"{idx}. {feat}: {direction} values boost predicted credit risk.")

    report_lines += ["", "Executive summary:", *exec_summary]
    report_lines += [
        "",
        "Recommendations:",
        "- Tighten underwriting for profiles matching the high-risk SHAP narratives.",
        "- Fast-track applications with the low-risk signatures above.",
        "- Re-run this pipeline each quarter to monitor drift.",
    ]

    report_path = ARTIFACT_DIR / "project_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #
def main():
    print("1) Loading + preprocessing data...")
    df = load_dataset(DATA_PATH)
    X, y, engineered = preprocess(df, target_col="loan_status")

    print("2) Train/test split (75/25 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    print("3) Training four candidate models...")
    model_dict = candidate_models()
    results, best = run_model_tournament(model_dict, X_train, X_test, y_train, y_test)

    print("\n===== Classification Report (best model) =====")
    print(best.report)
    best_estimator = best.estimator
    y_prob = best_estimator.predict_proba(X_test)[:, 1]

    print("4) Running SHAP analysis...")
    shap_bundle = generate_shap_outputs(best.name, best_estimator, X_train, X_test)
    model_importance = simple_feature_importance(best_estimator, X_train.columns.tolist())

    high_stories = narrate_local_cases(shap_bundle["shap_test"], X_test, y_prob, "high-risk")
    low_stories = narrate_local_cases(shap_bundle["shap_test"], X_test, y_prob, "low-risk")

    print("5) Writing final narrative report...")
    report_path = write_report(
        df_shape=df.shape,
        engineered=engineered,
        results=results,
        best=best,
        shap_bundle=shap_bundle,
        model_importance=model_importance,
        high_stories=high_stories,
        low_stories=low_stories,
    )

    print(f"\nReport written to {report_path.resolve()}")
    print(f"SHAP summary plot saved to {shap_bundle['summary_plot'].resolve()}")
    print(f"SHAP dependence plot saved to {shap_bundle['dependence_plot'].resolve()}")


if __name__ == "__main__":
    main()
