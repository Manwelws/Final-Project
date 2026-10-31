"""Serving package providing REST API endpoints, schemas, and encoding adapters."""

from final_project.serving.schemas import (
    CustomerProfileRaw,
    CustomerFeaturesVector,
    CustomerPredictionRequest,
    SinglePredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerExplainResponse,
    HealthResponse,
)
from final_project.serving.adapter import (
    encode_raw_profile,
    encode_batch_profiles,
    resolve_to_dataframe,
    MODEL_FEATURE_NAMES,
)
from final_project.serving.api import app

__all__ = [
    "app",
    "CustomerProfileRaw",
    "CustomerFeaturesVector",
    "CustomerPredictionRequest",
    "SinglePredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "CustomerExplainResponse",
    "HealthResponse",
    "encode_raw_profile",
    "encode_batch_profiles",
    "resolve_to_dataframe",
    "MODEL_FEATURE_NAMES",
]
