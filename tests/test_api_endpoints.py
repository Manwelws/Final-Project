import pytest
from fastapi.testclient import TestClient
from final_project.serving.api import app
from final_project.serving.adapter import MODEL_FEATURE_NAMES


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_algorithm" in data
    assert data["optimal_threshold"] > 0.0
    assert data["feature_count"] == 27


def test_predict_single_raw_profile(client):
    payload = {
        "Contract": "Month-to-month",
        "Internet Service": "Fiber optic",
        "Tenure Months": 2.0,
        "Monthly Charges": 89.5,
        "Total Charges": 179.0,
        "Payment Method": "Electronic check",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in (0, 1)
    assert data["risk_tier"] in ("Low", "Medium", "High")
    assert data["threshold_applied"] == 0.12


def test_predict_single_feature_vector(client):
    features = {col: 0.0 for col in MODEL_FEATURE_NAMES}
    features["Monthly Charges"] = 100.0
    features["Tenure Months"] = 1.0
    features["Contract_Month-to-month"] = 1.0

    payload = {"features": features}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_batch_endpoint(client):
    payload = {
        "items": [
            {
                "customer_id": "cust-001",
                "Contract": "Month-to-month",
                "Monthly Charges": 90.0,
            },
            {
                "customer_id": "cust-002",
                "Contract": "Two year",
                "Tenure Months": 60.0,
                "Monthly Charges": 25.0,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["customer_id"] == "cust-001"
    assert data["predictions"][1]["customer_id"] == "cust-002"


def test_predict_invalid_payload_triggers_422(client):
    # Negative tenure violates ge=0.0
    payload = {"Tenure Months": -10.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_explain_endpoint_raw_profile(client):
    payload = {
        "Contract": "Month-to-month",
        "Internet Service": "Fiber optic",
        "Tenure Months": 1.0,
        "Monthly Charges": 95.0,
        "Total Charges": 95.0,
        "Payment Method": "Electronic check",
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert "base_value" in data
    assert isinstance(data["risk_drivers"], list)
    assert isinstance(data["protective_factors"], list)

    for driver in data["risk_drivers"]:
        assert "parent_feature" in driver
        assert driver["shap_impact"] > 0

    for factor in data["protective_factors"]:
        assert "parent_feature" in factor
        assert factor["shap_impact"] < 0


def test_explain_endpoint_feature_vector(client):
    features = {col: 0.0 for col in MODEL_FEATURE_NAMES}
    features["Monthly Charges"] = 90.0
    features["Tenure Months"] = 2.0
    features["Contract_Month-to-month"] = 1.0

    payload = {"features": features}
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert len(data["risk_drivers"]) + len(data["protective_factors"]) > 0

