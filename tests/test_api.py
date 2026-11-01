import pytest
from fastapi.testclient import TestClient
from final_project.serving.api import app
from final_project.serving.adapter import MODEL_FEATURE_NAMES, encode_raw_profile
from final_project.serving.schemas import CustomerProfileRaw
from final_project.registry import ModelBundle


def make_full_raw_profile(**overrides) -> dict:
    base = {
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
    base.update(overrides)
    return base


@pytest.fixture
def client(synthetic_model_bundle_dir, monkeypatch):
    monkeypatch.setenv("CHURN_MODELS_DIR", str(synthetic_model_bundle_dir))
    bundle = ModelBundle.load(model_dir=synthetic_model_bundle_dir)
    app.state.bundle = bundle
    app.state.explainer = None
    with TestClient(app) as test_client:
        app.state.bundle = bundle
        app.state.explainer = None
        yield test_client
    app.state.bundle = None
    app.state.explainer = None


def test_health_endpoint_contract(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "algorithm" in data
    assert "model_algorithm" in data
    assert data["algorithm"] == data["model_algorithm"]
    assert 0.0 < data["optimal_threshold"] < 1.0
    assert data["feature_count"] > 0
    assert "model_timestamp" in data


def test_health_liveness_when_bundle_unloaded(client):
    app.state.bundle = None
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["algorithm"] == "unloaded"


def test_ready_endpoint_contract(client):
    # Ready when bundle loaded
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "algorithm" in data

    # 503 when bundle is unloaded
    app.state.bundle = None
    resp_unready = client.get("/ready")
    assert resp_unready.status_code == 503


def test_predict_raw_profile_matches_preprocessed_vector(client):
    raw_payload = make_full_raw_profile()

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
    assert data_raw["threshold_applied"] == data_vec["threshold_applied"]
    assert 0.0 < data_raw["threshold_applied"] < 1.0


def test_predict_threshold_calibration_and_risk_tiers(client):
    # High risk profile: month-to-month, high charges, low tenure, fiber optic, no security
    high_risk = make_full_raw_profile(
        Contract="Month-to-month",
        **{
            "Internet Service": "Fiber optic",
            "Tenure Months": 1.0,
            "Monthly Charges": 105.0,
            "Total Charges": 105.0,
            "Payment Method": "Electronic check",
            "Online Security": "No",
            "Tech Support": "No",
        }
    )
    resp = client.post("/predict", json=high_risk)
    assert resp.status_code == 200
    data = resp.json()
    assert data["threshold_applied"] == 0.12
    assert data["risk_tier"] in ("Medium", "High")
    assert data["churn_prediction"] == 1

    # Low risk profile: two year contract, high tenure, low charges
    low_risk = make_full_raw_profile(
        Contract="Two year",
        **{
            "Internet Service": "No",
            "Tenure Months": 70.0,
            "Monthly Charges": 20.0,
            "Total Charges": 1400.0,
            "Payment Method": "Credit card (automatic)",
            "Online Security": "No",
            "Tech Support": "No",
        }
    )
    resp_low = client.post("/predict", json=low_risk)
    assert resp_low.status_code == 200
    data_low = resp_low.json()
    assert data_low["churn_probability"] < 0.12
    assert data_low["churn_prediction"] == 0
    assert data_low["risk_tier"] == "Low"


def test_predict_batch_vectorized_scoring_dict_and_bare_list(client):
    item1 = make_full_raw_profile(
        CustomerID="cust-101",
        Contract="Month-to-month",
        **{"Monthly Charges": 95.0, "Total Charges": 950.0}
    )
    item2 = make_full_raw_profile(
        CustomerID="cust-102",
        Contract="Two year",
        **{"Tenure Months": 65.0, "Monthly Charges": 22.0, "Total Charges": 1430.0}
    )
    item3 = {
        "customer_id": "cust-103",
        "features": {col: 0.1 for col in MODEL_FEATURE_NAMES},
    }

    # 1. Alias shape {"items": [...]}
    batch_payload = {"items": [item1, item2, item3]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["customer_id"] == "cust-101"
    assert data["predictions"][1]["customer_id"] == "cust-102"
    assert data["predictions"][2]["customer_id"] == "cust-103"
    assert data["threshold_applied"] == 0.12

    # 2. Bare array shape [...]
    resp_bare = client.post("/predict/batch", json=[item1, item2, item3])
    assert resp_bare.status_code == 200
    data_bare = resp_bare.json()
    assert data_bare["count"] == 3
    assert data_bare["predictions"][0]["churn_probability"] == data["predictions"][0]["churn_probability"]


def test_explain_signed_attributions_and_cached_explainer(client):
    payload = make_full_raw_profile(
        Contract="Month-to-month",
        **{
            "Internet Service": "Fiber optic",
            "Tenure Months": 2.0,
            "Monthly Charges": 98.0,
            "Total Charges": 196.0,
            "Payment Method": "Electronic check",
        }
    )
    # First call initializes cached explainer
    assert getattr(app.state, "explainer", None) is None
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["base_value"], float)
    assert len(data["risk_drivers"]) > 0
    assert len(data["protective_factors"]) > 0
    assert data.get("attribution_method") in ("exact", "approximate")

    # Explainer should now be cached on app state
    cached_explainer = getattr(app.state, "explainer", None)
    assert cached_explainer is not None

    # Risk drivers push probability up (positive)
    for driver in data["risk_drivers"]:
        assert driver["shap_impact"] > 0
        assert driver["impact"] > 0
        assert isinstance(driver["parent_feature"], str)
        assert isinstance(driver["feature"], str)

    # Protective factors push probability down (negative)
    for factor in data["protective_factors"]:
        assert factor["shap_impact"] < 0
        assert factor["impact"] < 0
        assert isinstance(factor["parent_feature"], str)
        assert isinstance(factor["feature"], str)

    assert data["top_risk_driver"] is not None
    assert data["top_protective_factor"] is not None

    # Second call should reuse the same cached explainer instance
    resp2 = client.post("/explain", json=payload)
    assert resp2.status_code == 200
    assert getattr(app.state, "explainer", None) is cached_explainer


def test_validation_errors_and_boundary_handling(client):
    # 1. Empty dict returns 422
    resp_empty = client.post("/predict", json={})
    assert resp_empty.status_code == 422

    # 2. Unknown keys return 422 (extra="forbid")
    resp_extra = client.post("/predict", json=make_full_raw_profile(ExtraAttribute=999))
    assert resp_extra.status_code == 422

    # 3. Negative tenure violates ge=0.0
    resp_neg = client.post("/predict", json=make_full_raw_profile(**{"Tenure Months": -5.0}))
    assert resp_neg.status_code == 422

    # 4. Negative monthly charges violates ge=0.0
    resp_charges = client.post("/predict", json=make_full_raw_profile(**{"Monthly Charges": -100.0}))
    assert resp_charges.status_code == 422

    # 5. Empty batch items list violates min_length=1
    resp_empty_batch = client.post("/predict/batch", json={"items": []})
    assert resp_empty_batch.status_code == 422

    resp_empty_list = client.post("/predict/batch", json=[])
    assert resp_empty_list.status_code == 422

    # 6. Missing feature in feature vector returns 422
    incomplete_features = {col: 0.1 for col in MODEL_FEATURE_NAMES[:15]}
    resp_missing = client.post("/predict", json={"features": incomplete_features})
    assert resp_missing.status_code == 422
