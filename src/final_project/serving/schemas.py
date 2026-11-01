"""Pydantic v2 schemas for the prediction, explainability, and health endpoints."""

from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator


GenderType = Literal["Male", "Female", "male", "female", 0, 1]
BinaryChoice = Literal["Yes", "No", "yes", "no", 0, 1, True, False]
InternetServiceType = Literal["DSL", "Fiber optic", "No", "dsl", "fiber optic", "no"]
ContractType = Literal[
    "Month-to-month", "One year", "Two year",
    "month-to-month", "one year", "two year"
]
PaymentMethodType = Literal[
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
    "bank transfer (automatic)",
    "credit card (automatic)",
    "electronic check",
    "mailed check",
]
RiskTierType = Literal["Low", "Medium", "High"]


class CustomerProfileRaw(BaseModel):
    """Raw business customer profile accepted by the prediction service."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    customer_id: str | int | None = Field(default=None, alias="CustomerID")
    gender: GenderType = Field(..., alias="Gender")
    senior_citizen: BinaryChoice = Field(..., alias="Senior Citizen")
    partner: BinaryChoice = Field(..., alias="Partner")
    dependents: BinaryChoice = Field(..., alias="Dependents")
    tenure_months: float = Field(..., ge=0.0, alias="Tenure Months")
    phone_service: BinaryChoice = Field(..., alias="Phone Service")
    multiple_lines: BinaryChoice = Field(..., alias="Multiple Lines")
    internet_service: InternetServiceType = Field(..., alias="Internet Service")
    online_security: BinaryChoice = Field(..., alias="Online Security")
    online_backup: BinaryChoice = Field(..., alias="Online Backup")
    device_protection: BinaryChoice = Field(..., alias="Device Protection")
    tech_support: BinaryChoice = Field(..., alias="Tech Support")
    streaming_tv: BinaryChoice = Field(..., alias="Streaming TV")
    streaming_movies: BinaryChoice = Field(..., alias="Streaming Movies")
    contract: ContractType = Field(..., alias="Contract")
    paperless_billing: BinaryChoice = Field(..., alias="Paperless Billing")
    payment_method: PaymentMethodType = Field(..., alias="Payment Method")
    monthly_charges: float = Field(..., ge=0.0, alias="Monthly Charges")
    total_charges: float = Field(..., ge=0.0, alias="Total Charges")
    cltv: float = Field(..., ge=0.0, alias="CLTV")


class CustomerFeaturesVector(BaseModel):
    """Direct dictionary of the 27 numeric model features."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    customer_id: str | int | None = Field(default=None, alias="CustomerID")
    features: dict[str, float] = Field(
        ...,
        description="Key-value mapping of all 27 preprocessed numeric model features."
    )

    @model_validator(mode="after")
    def validate_features_complete(self) -> "CustomerFeaturesVector":
        from final_project.serving.adapter import MODEL_FEATURE_NAMES

        missing = [col for col in MODEL_FEATURE_NAMES if col not in self.features]
        if missing:
            raise ValueError(f"Missing required model features: {missing}")
        extra = [col for col in self.features if col not in MODEL_FEATURE_NAMES]
        if extra:
            raise ValueError(f"Unknown features not in model contract: {extra}")
        return self


class CustomerPredictionRequest(BaseModel):
    """Flexible wrapper accepting either a raw profile or preprocessed feature vector."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    customer_id: str | int | None = Field(default=None, alias="CustomerID")
    profile: CustomerProfileRaw | None = None
    vector: CustomerFeaturesVector | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_one_present(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data:
                raise ValueError("Empty request payload is not allowed.")
            cid = data.get("customer_id") or data.get("CustomerID")
            if "profile" in data or "vector" in data:
                return data
            if "features" in data and isinstance(data["features"], dict):
                return {
                    "customer_id": cid,
                    "vector": {"customer_id": cid, "features": data["features"]},
                }
            if "Internet Service_DSL" in data or "Contract_Month-to-month" in data:
                feat_dict = {k: v for k, v in data.items() if k not in ("customer_id", "CustomerID")}
                return {
                    "customer_id": cid,
                    "vector": {"customer_id": cid, "features": feat_dict},
                }
            # If plain fields are passed directly at root level, treat as profile
            return {"customer_id": cid, "profile": data}
        return data


class SinglePredictionResponse(BaseModel):
    """Prediction result for a single customer scored against calibrated threshold."""

    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_prediction: int = Field(..., ge=0, le=1)
    risk_tier: RiskTierType
    threshold_applied: float
    model_algorithm: str


class BatchPredictionItem(BaseModel):
    """Scored prediction item within a batch response."""

    customer_id: str | None = None
    churn_probability: float
    churn_prediction: int
    risk_tier: RiskTierType


class BatchPredictionRequest(BaseModel):
    """Batch scoring request supporting a list of raw customer profiles or feature vectors."""

    items: list[CustomerProfileRaw | CustomerFeaturesVector | CustomerPredictionRequest | dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of customer profiles to score in bulk (up to 1,000)."
    )

    @model_validator(mode="before")
    @classmethod
    def accept_bare_list_or_items_dict(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"items": data}
        if isinstance(data, dict) and "items" in data:
            return data
        if isinstance(data, dict) and not data:
            raise ValueError("Empty batch request is not allowed.")
        return data


class BatchPredictionResponse(BaseModel):
    """Bulk prediction response."""

    predictions: list[BatchPredictionItem]
    count: int
    threshold_applied: float
    model_algorithm: str


class DriverContribution(BaseModel):
    """Signed parent feature attribution impact."""

    model_config = ConfigDict(populate_by_name=True)

    parent_feature: str
    shap_impact: float
    feature: str | None = None
    impact: float | None = None

    @model_validator(mode="before")
    @classmethod
    def populate_compatible_names(cls, data: Any) -> Any:
        if isinstance(data, dict):
            feat = data.get("parent_feature") or data.get("feature")
            imp = data.get("shap_impact") if "shap_impact" in data else data.get("impact")
            return {
                "parent_feature": feat,
                "shap_impact": imp,
                "feature": feat,
                "impact": imp,
            }
        return data


class CustomerExplainResponse(BaseModel):
    """Real-time SHAP explanation response for frontline customer retention triage."""

    churn_probability: float
    base_value: float
    risk_drivers: list[DriverContribution]
    protective_factors: list[DriverContribution]
    top_risk_driver: str | None = None
    top_protective_factor: str | None = None
    attribution_method: str = "exact"


class HealthResponse(BaseModel):
    """Service liveness and model metadata response."""

    model_config = ConfigDict(populate_by_name=True)

    status: str = "healthy"
    algorithm: str = "unknown"
    model_algorithm: str = "unknown"
    optimal_threshold: float = 0.50
    holdout_roc_auc: float = 0.0
    holdout_pr_auc: float = 0.0
    model_timestamp: str = ""
    feature_count: int = 27

    @model_validator(mode="before")
    @classmethod
    def align_algorithm_names(cls, data: Any) -> Any:
        if isinstance(data, dict):
            algo = data.get("algorithm") or data.get("model_algorithm", "unknown")
            data["algorithm"] = algo
            data["model_algorithm"] = algo
        return data
