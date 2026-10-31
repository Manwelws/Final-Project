import pytest
from fastapi.testclient import TestClient
from final_project.serving.api import app
from final_project.serving.adapter import MODEL_FEATURE_NAMES, encode_raw_profile
from final_project.serving.schemas import CustomerProfileRaw


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint_contract(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_algorithm" in data
    assert data["optimal_threshold"] > 0.0
    assert data["optimal_threshold"] == 0.12
    assert data["feature_count"] == 27
    assert "model_timestamp" in data


def test_predict_raw_profile_matches_preprocessed_vector(client):
    raw_payload = {
        "Gender": "Female",
        "Senior Citizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure Months": 12.0,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 85.5,
        "Total Charges": 1026.0,
        "CLTV": 3200.0,
    }

    # Predict via raw business profile
    resp_raw = client.post("/predict", json=raw_payload)
    assert resp_raw.status_code == 200
    data_raw = resp_raw.json()

    # Pre-encode into exact 27-feature vector and predict
    profile = CustomerProfileRaw.model_validate(raw_payload)
    encoded_df = encode_raw_profile(profile)
    vector_features = {col: float(encoded_df[col].iloc[0]) for col in MODEL_FEATURE_NAMES}
    resp_vec = client.post("/predict", json={"features": vector_features})
    assert resp_vec.status_code == 200
    data_vec = resp_vec.json()

    # Scores and decisions must match exactly
    assert data_raw["churn_probability"] == data_vec["churn_probability"]
    assert data_raw["churn_prediction"] == data_vec["churn_prediction"]
    assert data_raw["risk_tier"] == data_vec["risk_tier"]
    assert data_raw["threshold_applied"] == 0.12


def test_predict_threshold_calibration_and_risk_tiers(client):
    # High risk profile: month-to-month, high charges, low tenure, fiber optic, no security
    high_risk = {
        "Contract": "Month-to-month",
        "Internet Service": "Fiber optic",
        "Tenure Months": 1.0,
        "Monthly Charges": 105.0,
        "Payment Method": "Electronic check",
    }
    resp = client.post("/predict", json=high_risk)
    assert resp.status_code == 200
    data = resp.json()
    assert data["threshold_applied"] == 0.12
    assert data["risk_tier"] in ("Medium", "High")
    assert data["churn_prediction"] == 1

    # Low risk profile: two year contract, high tenure, low charges
    low_risk = {
        "Contract": "Two year",
        "Internet Service": "No",
        "Tenure Months": 70.0,
        "Monthly Charges": 20.0,
        "Total Charges": 1400.0,
    }
    resp_low = client.post("/predict", json=low_risk)
    assert resp_low.status_code == 200
    data_low = resp_low.json()
    assert data_low["churn_probability"] < 0.12
    assert data_low["churn_prediction"] == 0
    assert data_low["risk_tier"] == "Low"


def test_predict_batch_vectorized_scoring(client):
    batch_payload = {
        "items": [
            {
                "customer_id": "cust-101",
                "Contract": "Month-to-month",
                "Monthly Charges": 95.0,
            },
            {
                "customer_id": "cust-102",
                "Contract": "Two year",
                "Tenure Months": 65.0,
                "Monthly Charges": 22.0,
            },
            {
                "customer_id": "cust-103",
                "features": {col: 0.1 for col in MODEL_FEATURE_NAMES},
            },
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["customer_id"] == "cust-101"
    assert data["predictions"][1]["customer_id"] == "cust-102"
    assert data["predictions"][2]["customer_id"] == "cust-103"
    assert data["threshold_applied"] == 0.12


def test_explain_signed_attributions_and_triage(client):
    payload = {
        "Contract": "Month-to-month",
        "Internet Service": "Fiber optic",
        "Tenure Months": 2.0,
        "Monthly Charges": 98.0,
        "Total Charges": 196.0,
        "Payment Method": "Electronic check",
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["base_value"], float)
    assert len(data["risk_drivers"]) > 0
    assert len(data["protective_factors"]) > 0

    # Risk drivers push probability up (positive)
    for driver in data["risk_drivers"]:
        assert driver["shap_impact"] > 0
        assert isinstance(driver["parent_feature"], str)

    # Protective factors push probability down (negative)
    for factor in data["protective_factors"]:
        assert factor["shap_impact"] < 0
        assert isinstance(factor["parent_feature"], str)

    assert data["top_risk_driver"] is not None
    assert data["top_protective_factor"] is not None


def test_validation_errors_and_boundary_handling(client):
    # 1. Negative tenure violates ge=0.0
    resp_neg = client.post("/predict", json={"Tenure Months": -5.0})
    assert resp_neg.status_code == 422
    assert "tenure_months" in resp_neg.text or "Tenure Months" in resp_neg.text

    # 2. Negative monthly charges violates ge=0.0
    resp_charges = client.post("/predict", json={"Monthly Charges": -100.0})
    assert resp_charges.status_code == 422

    # 3. Empty batch items list violates min_length=1
    resp_empty_batch = client.post("/predict/batch", json={"items": []})
    assert resp_empty_batch.status_code == 422
