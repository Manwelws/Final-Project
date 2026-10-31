"""Adapter routines converting raw customer business attributes into model feature vectors."""

from typing import Any
import pandas as pd
from final_project.serving.schemas import (
    CustomerProfileRaw,
    CustomerFeaturesVector,
    CustomerPredictionRequest,
)

MODEL_FEATURE_NAMES: list[str] = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Paperless Billing",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
    "Internet Service_DSL",
    "Internet Service_Fiber optic",
    "Internet Service_No",
    "Contract_Month-to-month",
    "Contract_One year",
    "Contract_Two year",
    "Payment Method_Bank transfer (automatic)",
    "Payment Method_Credit card (automatic)",
    "Payment Method_Electronic check",
    "Payment Method_Mailed check",
]


def _to_binary(val: Any) -> int:
    """Map string Yes/No, bool, or int into binary 1 or 0."""
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, float)):
        return 1 if val > 0 else 0
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("yes", "y", "true", "1", "male", "m"):
            return 1
        return 0
    return 0


def encode_raw_profile(profile: CustomerProfileRaw | dict[str, Any]) -> pd.DataFrame:
    """Transform a single raw customer profile into a 1x27 numeric DataFrame."""
    if isinstance(profile, CustomerProfileRaw):
        data = profile.model_dump(by_alias=False)
    else:
        # If passed as dict, instantiate schema to apply defaults and validation
        validated = CustomerProfileRaw.model_validate(profile)
        data = validated.model_dump(by_alias=False)

    gender_val = 1 if str(data.get("gender", "")).lower() in ("male", "m", "1") else 0

    internet_svc = str(data.get("internet_service", "No")).strip().lower()
    contract_val = str(data.get("contract", "Month-to-month")).strip().lower()
    payment_val = str(data.get("payment_method", "Electronic check")).strip().lower()

    row: dict[str, float] = {
        "Gender": float(gender_val),
        "Senior Citizen": float(_to_binary(data.get("senior_citizen", 0))),
        "Partner": float(_to_binary(data.get("partner", 0))),
        "Dependents": float(_to_binary(data.get("dependents", 0))),
        "Tenure Months": float(data.get("tenure_months", 0.0)),
        "Phone Service": float(_to_binary(data.get("phone_service", 0))),
        "Multiple Lines": float(_to_binary(data.get("multiple_lines", 0))),
        "Online Security": float(_to_binary(data.get("online_security", 0))),
        "Online Backup": float(_to_binary(data.get("online_backup", 0))),
        "Device Protection": float(_to_binary(data.get("device_protection", 0))),
        "Tech Support": float(_to_binary(data.get("tech_support", 0))),
        "Streaming TV": float(_to_binary(data.get("streaming_tv", 0))),
        "Streaming Movies": float(_to_binary(data.get("streaming_movies", 0))),
        "Paperless Billing": float(_to_binary(data.get("paperless_billing", 0))),
        "Monthly Charges": float(data.get("monthly_charges", 0.0)),
        "Total Charges": float(data.get("total_charges", 0.0)),
        "CLTV": float(data.get("cltv", 3000.0)),
        # One-hot Internet Service
        "Internet Service_DSL": 1.0 if internet_svc == "dsl" else 0.0,
        "Internet Service_Fiber optic": 1.0 if "fiber" in internet_svc else 0.0,
        "Internet Service_No": 1.0 if internet_svc in ("no", "none") else 0.0,
        # One-hot Contract
        "Contract_Month-to-month": 1.0 if "month" in contract_val else 0.0,
        "Contract_One year": 1.0 if contract_val == "one year" else 0.0,
        "Contract_Two year": 1.0 if contract_val == "two year" else 0.0,
        # One-hot Payment Method
        "Payment Method_Bank transfer (automatic)": 1.0 if "bank" in payment_val else 0.0,
        "Payment Method_Credit card (automatic)": 1.0 if "credit" in payment_val else 0.0,
        "Payment Method_Electronic check": 1.0 if "electronic" in payment_val else 0.0,
        "Payment Method_Mailed check": 1.0 if "mailed" in payment_val else 0.0,
    }

    df = pd.DataFrame([row], columns=MODEL_FEATURE_NAMES)
    return df


def encode_batch_profiles(
    items: list[CustomerProfileRaw | CustomerFeaturesVector | dict[str, Any]],
) -> pd.DataFrame:
    """Transform a list of raw profiles or feature vectors into an Nx27 DataFrame."""
    frames = [resolve_to_dataframe(item) for item in items]
    if not frames:
        return pd.DataFrame(columns=MODEL_FEATURE_NAMES)
    return pd.concat(frames, ignore_index=True)


def resolve_to_dataframe(item: Any) -> pd.DataFrame:
    """Normalize CustomerPredictionRequest, Profile, Vector, or dict into an aligned DataFrame."""
    if isinstance(item, CustomerPredictionRequest):
        if item.vector is not None:
            return resolve_to_dataframe(item.vector)
        if item.profile is not None:
            return encode_raw_profile(item.profile)
        raise ValueError("CustomerPredictionRequest contains neither profile nor vector.")

    if isinstance(item, CustomerFeaturesVector):
        feat_dict = item.features
        row = {col: float(feat_dict.get(col, 0.0)) for col in MODEL_FEATURE_NAMES}
        return pd.DataFrame([row], columns=MODEL_FEATURE_NAMES)

    if isinstance(item, CustomerProfileRaw):
        return encode_raw_profile(item)

    if isinstance(item, dict):
        if "features" in item and isinstance(item["features"], dict):
            feat_dict = item["features"]
            row = {col: float(feat_dict.get(col, 0.0)) for col in MODEL_FEATURE_NAMES}
            return pd.DataFrame([row], columns=MODEL_FEATURE_NAMES)
        # Check if already contains majority of the 27 dummy features
        if "Internet Service_DSL" in item or "Contract_Month-to-month" in item:
            row = {col: float(item.get(col, 0.0)) for col in MODEL_FEATURE_NAMES}
            return pd.DataFrame([row], columns=MODEL_FEATURE_NAMES)
        return encode_raw_profile(item)

    raise TypeError(f"Cannot resolve object of type {type(item)} to customer DataFrame.")
