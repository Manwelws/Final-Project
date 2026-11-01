"""FastAPI application for model serving, prediction, and health monitoring."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import os
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from final_project.registry import ModelBundle
from final_project.explainability import explain_customer_instance
from final_project.serving.schemas import (
    CustomerPredictionRequest,
    SinglePredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchPredictionItem,
    CustomerExplainResponse,
    DriverContribution,
    HealthResponse,
)
from final_project.serving.adapter import (
    resolve_to_dataframe,
    encode_batch_profiles,
    MODEL_FEATURE_NAMES,
)

MODELS_DIR = Path(os.environ.get("CHURN_MODELS_DIR", "models"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact and metadata into application state on startup."""
    if getattr(app.state, "bundle", None) is None:
        models_dir = Path(os.environ.get("CHURN_MODELS_DIR", "models"))
        try:
            app.state.bundle = ModelBundle.load(model_dir=models_dir)
        except Exception as exc:
            app.state.bundle = None
            app.state.load_error = str(exc)
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production REST microservice for customer churn scoring and retention triage.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_model_bundle() -> ModelBundle:
    """Retrieve or lazily initialize the ModelBundle."""
    bundle = getattr(app.state, "bundle", None)
    if bundle is not None:
        return bundle
    models_dir = Path(os.environ.get("CHURN_MODELS_DIR", "models"))
    try:
        bundle = ModelBundle.load(model_dir=models_dir)
        app.state.bundle = bundle
        return bundle
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifact unavailable: {exc}",
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Check service liveness and model metadata",
)
def health_check() -> HealthResponse:
    """Return service status, active model algorithm, optimal decision cutoff, and metadata."""
    bundle = get_model_bundle()
    meta = bundle.metadata
    return HealthResponse(
        status="healthy",
        model_algorithm=meta.algorithm or bundle.model.__class__.__name__,
        optimal_threshold=meta.optimal_threshold if meta.optimal_threshold > 0 else 0.50,
        holdout_roc_auc=meta.holdout_roc_auc,
        holdout_pr_auc=meta.holdout_pr_auc,
        model_timestamp=meta.timestamp,
        feature_count=len(meta.features) if meta.features else 27,
    )


@app.post(
    "/predict",
    response_model=SinglePredictionResponse,
    tags=["Inference"],
    summary="Score churn risk for a single customer",
)
def predict_churn(request: CustomerPredictionRequest) -> SinglePredictionResponse:
    """Predict churn probability and operational risk tier for a single customer.

    Accepts either raw customer profile fields (CRM attributes) or preprocessed 27 numeric feature vectors.
    Decision threshold applied: Cost-optimal threshold (0.12).
    """
    bundle = get_model_bundle()
    try:
        df = resolve_to_dataframe(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to resolve customer input to feature vector: {exc}",
        )

    risk_pred = bundle.predict_risk(df)

    prob = float(risk_pred.probabilities[0])
    pred = int(risk_pred.predictions[0])
    tier = risk_pred.risk_tiers[0]
    threshold = float(risk_pred.threshold_applied)

    return SinglePredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        risk_tier=tier,
        threshold_applied=threshold,
        model_algorithm=bundle.metadata.algorithm or bundle.model.__class__.__name__,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Bulk churn risk scoring for multiple customers",
)
def predict_churn_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score churn probability and operational risk tiers for a list of up to 1,000 customers in bulk."""
    bundle = get_model_bundle()
    try:
        batch_df = encode_batch_profiles(request.items)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to resolve batch input to feature matrix: {exc}",
        )

    risk_pred = bundle.predict_risk(batch_df)

    items: list[BatchPredictionItem] = []
    for i, (prob, pred, tier) in enumerate(
        zip(risk_pred.probabilities, risk_pred.predictions, risk_pred.risk_tiers)
    ):
        raw_item = request.items[i]
        cust_id = None
        if isinstance(raw_item, dict):
            cust_id = raw_item.get("customer_id") or raw_item.get("CustomerID")
        elif hasattr(raw_item, "customer_id"):
            cust_id = getattr(raw_item, "customer_id", None)

        items.append(
            BatchPredictionItem(
                customer_id=str(cust_id) if cust_id is not None else None,
                churn_probability=round(float(prob), 4),
                churn_prediction=int(pred),
                risk_tier=tier,
            )
        )

    return BatchPredictionResponse(
        predictions=items,
        count=len(items),
        threshold_applied=float(risk_pred.threshold_applied),
        model_algorithm=bundle.metadata.algorithm or bundle.model.__class__.__name__,
    )


@app.post(
    "/explain",
    response_model=CustomerExplainResponse,
    tags=["Explainability"],
    summary="Real-time SHAP explanation for frontline customer triage",
)
def explain_customer(request: CustomerPredictionRequest) -> CustomerExplainResponse:
    """Compute local TreeExplainer Shapley values and signed parent feature attributions for a customer.

    Returns:
        * churn_probability: Model predicted risk probability.
        * base_value: Expected value baseline.
        * risk_drivers: Factors increasing churn probability (+) sorted by magnitude.
        * protective_factors: Factors protecting against churn (-) sorted by magnitude.
        * top_risk_driver & top_protective_factor: Primary drivers for retention outreach.
    """
    bundle = get_model_bundle()
    try:
        df = resolve_to_dataframe(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to resolve customer input to feature vector: {exc}",
        )

    try:
        explanation = explain_customer_instance(
            model=bundle.model,
            customer_features=df,
            feature_names=MODEL_FEATURE_NAMES,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing SHAP explanation: {exc}",
        )

    risk_drivers = [
        DriverContribution(parent_feature=p, shap_impact=val)
        for p, val in explanation.risk_drivers
    ]
    protective_factors = [
        DriverContribution(parent_feature=p, shap_impact=val)
        for p, val in explanation.protective_factors
    ]

    top_risk = risk_drivers[0].parent_feature if risk_drivers else None
    top_prot = protective_factors[0].parent_feature if protective_factors else None

    return CustomerExplainResponse(
        churn_probability=round(explanation.churn_probability, 4),
        base_value=round(explanation.base_value, 4),
        risk_drivers=risk_drivers,
        protective_factors=protective_factors,
        top_risk_driver=top_risk,
        top_protective_factor=top_prot,
    )

