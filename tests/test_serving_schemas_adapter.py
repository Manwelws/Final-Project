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
    BatchPredictionRequest,
)
from final_project.serving.adapter import (
    encode_raw_profile,
    encode_batch_profiles,
    resolve_to_dataframe,
    _to_binary,
    MODEL_FEATURE_NAMES,
)


def make_valid_raw_profile_dict(**overrides) -> dict:
    base = {
        "Gender": "Female",
        "Senior Citizen": "Yes",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": 24.0,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "One year",
        "Paperless Billing": "Yes",
        "Payment Method": "Credit card (automatic)",
        "Monthly Charges": 85.5,
        "Total Charges": 2052.0,
        "CLTV": 3200.0,
    }
    base.update(overrides)
    return base


def test_customer_profile_raw_validation_and_encoding():
    raw_dict = make_valid_raw_profile_dict()
    profile = CustomerProfileRaw.model_validate(raw_dict)

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
    raw_data = make_valid_raw_profile_dict(
        Gender="Male",
        Contract="Month-to-month",
        **{"Internet Service": "DSL", "Payment Method": "Electronic check"}
    )
    profile = CustomerProfileRaw.model_validate(raw_data)
    df = encode_raw_profile(profile)

    assert df["Gender"].iloc[0] == 1.0  # Male -> 1
    assert df["Contract_Month-to-month"].iloc[0] == 1.0
    assert df["Internet Service_DSL"].iloc[0] == 1.0
    assert df["Payment Method_Electronic check"].iloc[0] == 1.0


def test_empty_dict_rejected():
    with pytest.raises((ValidationError, ValueError)):
        CustomerProfileRaw.model_validate({})

    with pytest.raises((ValidationError, ValueError)):
        CustomerPredictionRequest.model_validate({})


def test_extra_fields_forbidden():
    raw_data = make_valid_raw_profile_dict(UnknownAttribute="unexpected_value")
    with pytest.raises(ValidationError):
        CustomerProfileRaw.model_validate(raw_data)


def test_unknown_categoricals_rejected():
    with pytest.raises(ValidationError):
        CustomerProfileRaw.model_validate(make_valid_raw_profile_dict(Partner="maybe"))

    with pytest.raises(ValidationError):
        CustomerProfileRaw.model_validate(make_valid_raw_profile_dict(**{"Internet Service": "satellite"}))

    with pytest.raises(ValueError):
        _to_binary("unsupported_string")

    with pytest.raises(ValueError):
        _to_binary(99)


def test_customer_features_vector_resolution():
    features_dict = {col: 0.5 for col in MODEL_FEATURE_NAMES}
    features_dict["Tenure Months"] = 36.0
    vec = CustomerFeaturesVector(features=features_dict)

    df = resolve_to_dataframe(vec)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == MODEL_FEATURE_NAMES
    assert df["Tenure Months"].iloc[0] == 36.0


def test_customer_features_vector_missing_or_extra_forbidden():
    incomplete = {col: 0.5 for col in MODEL_FEATURE_NAMES[:10]}
    with pytest.raises(ValidationError):
        CustomerFeaturesVector(features=incomplete)

    extra = {col: 0.5 for col in MODEL_FEATURE_NAMES}
    extra["SpuriousFeature"] = 1.0
    with pytest.raises(ValidationError):
        CustomerFeaturesVector(features=extra)


def test_customer_prediction_request_polymorphism():
    # 1. Full raw profile dict
    raw_profile = make_valid_raw_profile_dict(Contract="Two year")
    req1 = CustomerPredictionRequest.model_validate(raw_profile)
    df1 = resolve_to_dataframe(req1)
    assert df1["Contract_Two year"].iloc[0] == 1.0

    # 2. Features vector dictionary
    feat_map = {col: 1.0 for col in MODEL_FEATURE_NAMES}
    req2 = CustomerPredictionRequest.model_validate({"features": feat_map})
    df2 = resolve_to_dataframe(req2)
    assert df2["Gender"].iloc[0] == 1.0


def test_encode_batch_profiles_mixed():
    p1 = CustomerProfileRaw.model_validate(make_valid_raw_profile_dict(Contract="Month-to-month"))
    p2 = CustomerProfileRaw.model_validate(make_valid_raw_profile_dict(Contract="Two year"))
    p3 = CustomerFeaturesVector(features={col: 0.0 for col in MODEL_FEATURE_NAMES})

    batch_df = encode_batch_profiles([p1, p2, p3])
    assert len(batch_df) == 3
    assert batch_df["Contract_Month-to-month"].iloc[0] == 1.0
    assert batch_df["Contract_Two year"].iloc[1] == 1.0
    assert batch_df["Contract_Month-to-month"].iloc[2] == 0.0


def test_batch_prediction_request_bare_list_support():
    p1 = make_valid_raw_profile_dict()
    p2 = make_valid_raw_profile_dict()

    # Bare list
    req_list = BatchPredictionRequest.model_validate([p1, p2])
    assert len(req_list.items) == 2

    # Dict alias
    req_dict = BatchPredictionRequest.model_validate({"items": [p1, p2]})
    assert len(req_dict.items) == 2


def test_validation_error_on_negative_tenure():
    with pytest.raises(ValidationError):
        CustomerProfileRaw.model_validate(make_valid_raw_profile_dict(**{"Tenure Months": -5.0}))
