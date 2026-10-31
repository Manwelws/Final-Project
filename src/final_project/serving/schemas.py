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

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    customer_id: str | int | None = Field(default=None, alias="CustomerID")
    gender: GenderType = Field(default="Male", alias="Gender")
    senior_citizen: BinaryChoice = Field(default="No", alias="Senior Citizen")
    partner: BinaryChoice = Field(default="No", alias="Partner")
    dependents: BinaryChoice = Field(default="No", alias="Dependents")
    tenure_months: float = Field(default=1.0, ge=0.0, alias="Tenure Months")
    phone_service: BinaryChoice = Field(default="Yes", alias="Phone Service")
    multiple_lines: BinaryChoice = Field(default="No", alias="Multiple Lines")
    internet_service: InternetServiceType = Field(default="DSL", alias="Internet Service")
    online_security: BinaryChoice = Field(default="No", alias="Online Security")
    online_backup: BinaryChoice = Field(default="No", alias="Online Backup")
    device_protection: BinaryChoice = Field(default="No", alias="Device Protection")
    tech_support: BinaryChoice = Field(default="No", alias="Tech Support")
    streaming_tv: BinaryChoice = Field(default="No", alias="Streaming TV")
    streaming_movies: BinaryChoice = Field(default="No", alias="Streaming Movies")
    contract: ContractType = Field(default="Month-to-month", alias="Contract")
    paperless_billing: BinaryChoice = Field(default="Yes", alias="Paperless Billing")
    payment_method: PaymentMethodType = Field(
        default="Electronic check", alias="Payment Method"
    )
    monthly_charges: float = Field(default=70.0, ge=0.0, alias="Monthly Charges")
    total_charges: float = Field(default=70.0, ge=0.0, alias="Total Charges")
    cltv: float = Field(default=3000.0, ge=0.0, alias="CLTV")


class CustomerFeaturesVector(BaseModel):
    """Direct dictionary of the 27 numeric model features."""

    model_config = ConfigDict(extra="allow")

    features: dict[str, float] = Field(
        ...,
        description="Key-value mapping of all 27 preprocessed numeric model features."
    )


class CustomerPredictionRequest(BaseModel):
    """Flexible wrapper accepting either a raw profile or preprocessed feature vector."""

    model_config = ConfigDict(populate_by_name=True)

    profile: CustomerProfileRaw | None = None
    vector: CustomerFeaturesVector | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_one_present(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "profile" in data or "vector" in data:
                return data
            if "features" in data:
                return {"vector": {"features": data["features"]}}
            # If plain fields are passed directly at root level, treat as profile
            return {"profile": data}
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

    items: list[CustomerProfileRaw | CustomerFeaturesVector | dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of customer profiles to score in bulk (up to 1,000)."
    )


class BatchPredictionResponse(BaseModel):
    """Bulk prediction response."""

    predictions: list[BatchPredictionItem]
    count: int
    threshold_applied: float
    model_algorithm: str


class DriverContribution(BaseModel):
    """Signed parent feature attribution impact."""

    parent_feature: str
    shap_impact: float


class CustomerExplainResponse(BaseModel):
    """Real-time SHAP explanation response for frontline customer retention triage."""

    churn_probability: float
    base_value: float
    risk_drivers: list[DriverContribution]
    protective_factors: list[DriverContribution]
    top_risk_driver: str | None = None
    top_protective_factor: str | None = None


class HealthResponse(BaseModel):
    """Service liveness and model metadata response."""

    status: str = "healthy"
    model_algorithm: str
    optimal_threshold: float
    holdout_roc_auc: float
    holdout_pr_auc: float
    model_timestamp: str
    feature_count: int
