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
    """Map string Yes/No, bool, or int into binary 1 or 0, rejecting unknown values."""
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, float)):
        if val in (0, 0.0):
            return 0
        if val in (1, 1.0):
            return 1
        raise ValueError(f"Invalid binary numeric value: {val!r}")
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("yes", "y", "true", "1"):
            return 1
        if v in ("no", "n", "false", "0"):
            return 0
        raise ValueError(f"Unknown binary categorical value: {val!r}")
    raise ValueError(f"Unsupported binary categorical type: {type(val)}")


def encode_raw_profile(profile: CustomerProfileRaw | dict[str, Any]) -> pd.DataFrame:
    """Transform a single raw customer profile into a 1x27 numeric DataFrame."""
    if isinstance(profile, CustomerProfileRaw):
        data = profile.model_dump(by_alias=False)
    else:
        # If passed as dict, instantiate schema to apply validation
        validated = CustomerProfileRaw.model_validate(profile)
        data = validated.model_dump(by_alias=False)

    gender_str = str(data["gender"]).strip().lower()
    if gender_str in ("male", "m", "1"):
        gender_val = 1.0
    elif gender_str in ("female", "f", "0"):
        gender_val = 0.0
    else:
        raise ValueError(f"Unknown gender value: {data['gender']!r}")

    internet_svc = str(data["internet_service"]).strip().lower()
    if internet_svc not in ("dsl", "fiber optic", "no"):
        raise ValueError(f"Unknown internet service value: {data['internet_service']!r}")

    contract_val = str(data["contract"]).strip().lower()
    if contract_val not in ("month-to-month", "one year", "two year"):
        raise ValueError(f"Unknown contract value: {data['contract']!r}")

    payment_val = str(data["payment_method"]).strip().lower()
    valid_payments = {
        "bank transfer (automatic)",
        "credit card (automatic)",
        "electronic check",
        "mailed check",
    }
    if payment_val not in valid_payments:
        raise ValueError(f"Unknown payment method value: {data['payment_method']!r}")

    row: dict[str, float] = {
        "Gender": gender_val,
        "Senior Citizen": float(_to_binary(data["senior_citizen"])),
        "Partner": float(_to_binary(data["partner"])),
        "Dependents": float(_to_binary(data["dependents"])),
        "Tenure Months": float(data["tenure_months"]),
        "Phone Service": float(_to_binary(data["phone_service"])),
        "Multiple Lines": float(_to_binary(data["multiple_lines"])),
        "Online Security": float(_to_binary(data["online_security"])),
        "Online Backup": float(_to_binary(data["online_backup"])),
        "Device Protection": float(_to_binary(data["device_protection"])),
        "Tech Support": float(_to_binary(data["tech_support"])),
        "Streaming TV": float(_to_binary(data["streaming_tv"])),
        "Streaming Movies": float(_to_binary(data["streaming_movies"])),
        "Paperless Billing": float(_to_binary(data["paperless_billing"])),
        "Monthly Charges": float(data["monthly_charges"]),
        "Total Charges": float(data["total_charges"]),
        "CLTV": float(data["cltv"]),
        # One-hot Internet Service
        "Internet Service_DSL": 1.0 if internet_svc == "dsl" else 0.0,
        "Internet Service_Fiber optic": 1.0 if internet_svc == "fiber optic" else 0.0,
        "Internet Service_No": 1.0 if internet_svc == "no" else 0.0,
        # One-hot Contract
        "Contract_Month-to-month": 1.0 if contract_val == "month-to-month" else 0.0,
        "Contract_One year": 1.0 if contract_val == "one year" else 0.0,
        "Contract_Two year": 1.0 if contract_val == "two year" else 0.0,
        # One-hot Payment Method
        "Payment Method_Bank transfer (automatic)": 1.0 if payment_val == "bank transfer (automatic)" else 0.0,
        "Payment Method_Credit card (automatic)": 1.0 if payment_val == "credit card (automatic)" else 0.0,
        "Payment Method_Electronic check": 1.0 if payment_val == "electronic check" else 0.0,
        "Payment Method_Mailed check": 1.0 if payment_val == "mailed check" else 0.0,
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
        missing = [col for col in MODEL_FEATURE_NAMES if col not in feat_dict]
        if missing:
            raise ValueError(f"Missing required model features: {missing}")
        extra = [col for col in feat_dict if col not in MODEL_FEATURE_NAMES]
        if extra:
            raise ValueError(f"Unknown features not in model contract: {extra}")
        row = {col: float(feat_dict[col]) for col in MODEL_FEATURE_NAMES}
        return pd.DataFrame([row], columns=MODEL_FEATURE_NAMES)

    if isinstance(item, CustomerProfileRaw):
        return encode_raw_profile(item)

    if isinstance(item, dict):
        if "features" in item and isinstance(item["features"], dict):
            feat_dict = item["features"]
            missing = [col for col in MODEL_FEATURE_NAMES if col not in feat_dict]
            if missing:
                raise ValueError(f"Missing required model features: {missing}")
            extra = [col for col in feat_dict if col not in MODEL_FEATURE_NAMES]
            if extra:
                raise ValueError(f"Unknown features not in model contract: {extra}")
            row = {col: float(feat_dict[col]) for col in MODEL_FEATURE_NAMES}
            return pd.DataFrame([row], columns=MODEL_FEATURE_NAMES)
        # Check if already contains majority of the 27 dummy features
        if "Internet Service_DSL" in item or "Contract_Month-to-month" in item:
            missing = [col for col in MODEL_FEATURE_NAMES if col not in item]
            if missing:
                raise ValueError(f"Missing required model features: {missing}")
            extra = [col for col in item if col not in MODEL_FEATURE_NAMES and col not in ("customer_id", "CustomerID")]
            if extra:
                raise ValueError(f"Unknown features not in model contract: {extra}")
            row = {col: float(item[col]) for col in MODEL_FEATURE_NAMES}
            return pd.DataFrame([row], columns=MODEL_FEATURE_NAMES)
        return encode_raw_profile(item)

    raise TypeError(f"Cannot resolve object of type {type(item)} to customer DataFrame.")
