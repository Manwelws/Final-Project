import pytest
import pandas as pd
from pydantic import ValidationError

from final_project.serving.schemas import (
    CustomerProfileRaw,
    CustomerFeaturesVector,
    CustomerPredictionRequest,
    SinglePredictionResponse,
    HealthResponse,
    CustomerExplainResponse,
)
from final_project.serving.adapter import (
    encode_raw_profile,
    encode_batch_profiles,
    resolve_to_dataframe,
    MODEL_FEATURE_NAMES,
)


def test_customer_profile_raw_defaults_and_encoding():
    profile = CustomerProfileRaw(
        gender="Female",
        senior_citizen="Yes",
        tenure_months=24.0,
        monthly_charges=85.5,
        total_charges=2052.0,
        contract="One year",
        internet_service="Fiber optic",
        payment_method="Credit card (automatic)",
    )

    df = encode_raw_profile(profile)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == MODEL_FEATURE_NAMES
    assert len(df) == 1

    # Check categorical mappings
    assert df["Gender"].iloc[0] == 0.0  # Female -> 0
    assert df["Senior Citizen"].iloc[0] == 1.0  # Yes -> 1
    assert df["Tenure Months"].iloc[0] == 24.0
    assert df["Internet Service_Fiber optic"].iloc[0] == 1.0
    assert df["Internet Service_DSL"].iloc[0] == 0.0
    assert df["Contract_One year"].iloc[0] == 1.0
    assert df["Contract_Month-to-month"].iloc[0] == 0.0
    assert df["Payment Method_Credit card (automatic)"].iloc[0] == 1.0


def test_customer_profile_alias_parsing():
    raw_data = {
        "Gender": "Male",
        "Senior Citizen": "No",
        "Tenure Months": 12,
        "Monthly Charges": 60.0,
        "Total Charges": 720.0,
        "Contract": "Month-to-month",
        "Internet Service": "DSL",
        "Payment Method": "Electronic check",
    }
    profile = CustomerProfileRaw.model_validate(raw_data)
    df = encode_raw_profile(profile)

    assert df["Gender"].iloc[0] == 1.0  # Male -> 1
    assert df["Contract_Month-to-month"].iloc[0] == 1.0
    assert df["Internet Service_DSL"].iloc[0] == 1.0
    assert df["Payment Method_Electronic check"].iloc[0] == 1.0


def test_customer_features_vector_resolution():
    features_dict = {col: 0.5 for col in MODEL_FEATURE_NAMES}
    features_dict["Tenure Months"] = 36.0
    vec = CustomerFeaturesVector(features=features_dict)

    df = resolve_to_dataframe(vec)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == MODEL_FEATURE_NAMES
    assert df["Tenure Months"].iloc[0] == 36.0


def test_customer_prediction_request_polymorphism():
    # 1. Plain dictionary treated as profile
    req1 = CustomerPredictionRequest.model_validate({"contract": "Two year"})
    df1 = resolve_to_dataframe(req1)
    assert df1["Contract_Two year"].iloc[0] == 1.0

    # 2. Features vector dictionary
    feat_map = {col: 1.0 for col in MODEL_FEATURE_NAMES}
    req2 = CustomerPredictionRequest.model_validate({"features": feat_map})
    df2 = resolve_to_dataframe(req2)
    assert df2["Gender"].iloc[0] == 1.0


def test_encode_batch_profiles_mixed():
    p1 = CustomerProfileRaw(contract="Month-to-month")
    p2 = CustomerProfileRaw(contract="Two year")
    p3 = CustomerFeaturesVector(features={col: 0.0 for col in MODEL_FEATURE_NAMES})

    batch_df = encode_batch_profiles([p1, p2, p3])
    assert len(batch_df) == 3
    assert batch_df["Contract_Month-to-month"].iloc[0] == 1.0
    assert batch_df["Contract_Two year"].iloc[1] == 1.0
    assert batch_df["Contract_Month-to-month"].iloc[2] == 0.0


def test_validation_error_on_negative_tenure():
    with pytest.raises(ValidationError):
        CustomerProfileRaw(tenure_months=-5.0)
