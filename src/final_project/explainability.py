"""Model interpretability module using SHAP TreeExplainer with feature grouping."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from final_project.data import load_features_and_target

KNOWN_PARENTS = [
    "Contract",
    "Internet Service",
    "Payment Method",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Multiple Lines",
    "Phone Service",
    "Gender",
    "Partner",
    "Dependents",
    "Paperless Billing",
]


@dataclass(frozen=True)
class ExplanationResult:
    grouped_importance: pd.DataFrame
    generated_figures: list[Path]
    samples_explained: int


@dataclass(frozen=True)
class CustomerExplanation:
    churn_probability: float
    base_value: float
    risk_drivers: list[tuple[str, float]]  # Parent features pushing toward churn (+)
    protective_factors: list[tuple[str, float]]  # Parent features protecting against churn (-)
    raw_shap_values: dict[str, float]


def group_shap_by_parent_features(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Aggregate one-hot dummy variable SHAP values into parent categorical features.

    Computes the sample-wise sum of absolute SHAP contributions across dummy columns,
    then averages across samples to produce an executive ranking of parent features.
    """
    if shap_values.ndim != 2:
        raise ValueError(f"Expected 2D shap_values array, got shape {shap_values.shape}")
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"shap_values columns ({shap_values.shape[1]}) does not match feature_names length ({len(feature_names)})"
        )

    parent_map: dict[str, list[int]] = {}
    for idx, col in enumerate(feature_names):
        parent: str | None = None
        for kp in KNOWN_PARENTS:
            if col == kp or col.startswith(f"{kp}_"):
                parent = kp
                break
        if parent is None:
            if "_" in col:
                parent = col.split("_")[0]
            else:
                parent = col
        parent_map.setdefault(parent, []).append(idx)

    rows = []
    for parent, indices in parent_map.items():
        # Sample-wise sum of absolute SHAP values across child dummy features, then mean across samples
        parent_sample_contrib = np.sum(np.abs(shap_values[:, indices]), axis=1)
        mean_abs_val = float(np.mean(parent_sample_contrib))
        rows.append({
            "parent_feature": parent,
            "mean_abs_shap": mean_abs_val,
            "num_subfeatures": len(indices),
        })

    grouped_df = pd.DataFrame(rows).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    return grouped_df


def extract_tree_estimator(model: Any) -> Any:
    """Extract underlying tree estimator from pipelines or wrapped objects if needed."""
    if hasattr(model, "named_steps"):
        if "classifier" in model.named_steps:
            return model.named_steps["classifier"]
        return model.steps[-1][1]
    return model


def run_explanation(
    data_dir: Path | str = "data/processed",
    model_path: Path | str = "models/model.joblib",
    output_dir: Path | str = "reports/figures",
    max_samples: int = 150,
    random_state: int = 42,
) -> ExplanationResult:
    """Compute exact TreeExplainer SHAP values and export diagnostic visualizations."""
    data_dir = Path(data_dir)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    model = joblib.load(model_path)
    estimator = extract_tree_estimator(model)

    X, y = load_features_and_target(data_dir=data_dir)

    # Use holdout test split for unbiased explainability
    if len(X) > 20:
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=random_state
        )
    else:
        X_test, y_test = X, y

    # Sample for tractable SHAP compute while preserving class diversity
    sample_size = min(len(X_test), max_samples)
    if sample_size < len(X_test):
        X_sample, _, y_sample, _ = train_test_split(
            X_test,
            y_test,
            train_size=sample_size,
            stratify=y_test if y_test.nunique() > 1 else None,
            random_state=random_state,
        )
    else:
        X_sample, y_sample = X_test, y_test

    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(X_sample)

    # Handle multi-class / binary 3D SHAP explanation arrays
    if explanation.values.ndim == 3:
        exp_class1 = explanation[:, :, 1]
    else:
        exp_class1 = explanation

    shap_matrix = exp_class1.values
    feature_names = list(X_sample.columns)

    grouped_df = group_shap_by_parent_features(shap_matrix, feature_names)

    generated_figures: list[Path] = []

    # 1. Grouped Mean Absolute SHAP horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    top_grouped = grouped_df.head(10).iloc[::-1]
    bars = ax.barh(
        top_grouped["parent_feature"],
        top_grouped["mean_abs_shap"],
        color="#2563eb",
        edgecolor="#1d4ed8",
        height=0.6,
    )
    ax.set_title("Global Feature Importance (Grouped by Parent Category)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Model Churn Output)", fontsize=11)
    ax.set_ylabel("Parent Feature", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Add numeric labels to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="semibold",
        )

    plt.tight_layout()
    grouped_fig_path = output_dir / "shap_summary_grouped.png"
    plt.savefig(grouped_fig_path)
    plt.close("all")
    generated_figures.append(grouped_fig_path)

    # 2. Detailed Beeswarm Plot across all individual features
    plt.figure(figsize=(10, 8), dpi=300)
    shap.plots.beeswarm(exp_class1, max_display=15, show=False)
    plt.title("Detailed SHAP Feature Impact (Beeswarm Summary)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    detailed_fig_path = output_dir / "shap_summary_detailed.png"
    plt.savefig(detailed_fig_path, bbox_inches="tight")
    plt.close("all")
    generated_figures.append(detailed_fig_path)

    # 3. Dependence Plot: Monthly Charges
    if "Monthly Charges" in feature_names:
        plt.figure(figsize=(8, 6), dpi=300)
        shap.plots.scatter(exp_class1[:, "Monthly Charges"], show=False)
        plt.title("SHAP Dependence Plot: Monthly Charges", fontsize=12, fontweight="bold")
        plt.tight_layout()
        dep_mc_path = output_dir / "shap_dependence_monthly_charges.png"
        plt.savefig(dep_mc_path, bbox_inches="tight")
        plt.close("all")
        generated_figures.append(dep_mc_path)

    # 4. Dependence Plot: Tenure Months
    if "Tenure Months" in feature_names:
        plt.figure(figsize=(8, 6), dpi=300)
        shap.plots.scatter(exp_class1[:, "Tenure Months"], show=False)
        plt.title("SHAP Dependence Plot: Tenure Months", fontsize=12, fontweight="bold")
        plt.tight_layout()
        dep_tenure_path = output_dir / "shap_dependence_tenure.png"
        plt.savefig(dep_tenure_path, bbox_inches="tight")
        plt.close("all")
        generated_figures.append(dep_tenure_path)

    # 5. Local Waterfall Plot: Sample Churner (True positive or high risk)
    churner_indices = np.where(y_sample.to_numpy() == 1)[0]
    churner_idx = int(churner_indices[0]) if len(churner_indices) > 0 else 0

    plt.figure(figsize=(9, 6), dpi=300)
    shap.plots.waterfall(exp_class1[churner_idx], max_display=10, show=False)
    plt.title("Individual Risk Factors: Churned Customer", fontsize=12, fontweight="bold")
    plt.tight_layout()
    churner_fig_path = output_dir / "shap_waterfall_churner.png"
    plt.savefig(churner_fig_path, bbox_inches="tight")
    plt.close("all")
    generated_figures.append(churner_fig_path)

    # 6. Local Waterfall Plot: Sample Retained Customer (True negative or low risk)
    retained_indices = np.where(y_sample.to_numpy() == 0)[0]
    retained_idx = int(retained_indices[0]) if len(retained_indices) > 0 else 0

    plt.figure(figsize=(9, 6), dpi=300)
    shap.plots.waterfall(exp_class1[retained_idx], max_display=10, show=False)
    plt.title("Individual Risk Factors: Retained Customer", fontsize=12, fontweight="bold")
    plt.tight_layout()
    retained_fig_path = output_dir / "shap_waterfall_retained.png"
    plt.savefig(retained_fig_path, bbox_inches="tight")
    plt.close("all")
    generated_figures.append(retained_fig_path)

    return ExplanationResult(
        grouped_importance=grouped_df,
        generated_figures=generated_figures,
        samples_explained=len(X_sample),
    )


def explain_customer_instance(
    model: Any,
    customer_features: pd.DataFrame | pd.Series | np.ndarray,
    feature_names: list[str] | None = None,
) -> CustomerExplanation:
    """Compute local TreeExplainer attribution with signed parent feature aggregation for a single customer."""
    # Normalize customer_features into 1-row DataFrame
    if isinstance(customer_features, pd.DataFrame):
        if len(customer_features) != 1:
            raise ValueError(f"Expected single customer row in customer_features, got {len(customer_features)} rows")
        df = customer_features.copy()
        if feature_names is None:
            feature_names = list(df.columns)
        else:
            df = df[feature_names]
    elif isinstance(customer_features, pd.Series):
        if feature_names is None:
            feature_names = list(customer_features.index)
        df = pd.DataFrame([customer_features.values], columns=feature_names)
    elif isinstance(customer_features, np.ndarray):
        arr = customer_features
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.shape[0] != 1:
            raise ValueError(f"Expected single customer row in numpy array, got shape {arr.shape}")
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=feature_names)
    else:
        raise TypeError(f"Unsupported customer_features type: {type(customer_features)}")

    estimator = extract_tree_estimator(model)

    # Compute probability prediction
    if hasattr(model, "predict_proba"):
        churn_prob = float(model.predict_proba(df)[:, 1][0])
    elif hasattr(estimator, "predict_proba"):
        churn_prob = float(estimator.predict_proba(df.to_numpy())[:, 1][0])
    else:
        churn_prob = 0.0

    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(df)

    # Extract single instance values and base_value for positive churn class
    if explanation.values.ndim == 3:
        shap_vals = explanation.values[0, :, 1]
        base_val = (
            float(explanation.base_values[0, 1])
            if hasattr(explanation.base_values, "ndim") and explanation.base_values.ndim > 1
            else float(explanation.base_values[0])
        )
    else:
        shap_vals = explanation.values[0, :]
        base_val = (
            float(explanation.base_values[0])
            if hasattr(explanation.base_values, "__len__")
            else float(explanation.base_values)
        )

    raw_shap_values = {
        feat: float(shap_vals[i]) for i, feat in enumerate(feature_names)
    }

    # Map features to parents
    parent_map: dict[str, list[int]] = {}
    for idx, col in enumerate(feature_names):
        parent: str | None = None
        for kp in KNOWN_PARENTS:
            if col == kp or col.startswith(f"{kp}_"):
                parent = kp
                break
        if parent is None:
            if "_" in col:
                parent = col.split("_")[0]
            else:
                parent = col
        parent_map.setdefault(parent, []).append(idx)

    risk_drivers: list[tuple[str, float]] = []
    protective_factors: list[tuple[str, float]] = []

    for parent, indices in parent_map.items():
        net_signed_shap = float(np.sum(shap_vals[indices]))
        if net_signed_shap > 0:
            risk_drivers.append((parent, round(net_signed_shap, 4)))
        elif net_signed_shap < 0:
            protective_factors.append((parent, round(net_signed_shap, 4)))

    # Sort risk drivers descending (highest positive push towards churn first)
    risk_drivers.sort(key=lambda x: x[1], reverse=True)
    # Sort protective factors ascending (most negative push away from churn first)
    protective_factors.sort(key=lambda x: x[1])

    return CustomerExplanation(
        churn_probability=churn_prob,
        base_value=base_val,
        risk_drivers=risk_drivers,
        protective_factors=protective_factors,
        raw_shap_values=raw_shap_values,
    )

