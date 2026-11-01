"""Production Customer Churn Prediction & Automated Retraining System."""

from __future__ import annotations

__version__ = "0.1.0"

from final_project.cli import main
from final_project.pipeline import run_training, PipelineRunSummary
from final_project.data import load_and_split_data, load_features_and_target, DataSplit
from final_project.models import (
    train_baseline_model,
    TrainingResult,
    get_candidate_model,
    compute_scale_pos_weight,
    evaluate_model_cv,
    ModelCVMetrics,
    benchmark_candidates,
)
from final_project.tuning import run_tuning_study, TuningResult
from final_project.evaluation import (
    run_evaluation,
    compute_metrics_at_threshold,
    optimize_threshold,
    ThresholdMetrics,
    OptimizationResult,
    EvaluationSummary,
    EvaluationReport,
)
from final_project.explainability import (
    run_explanation,
    group_shap_by_parent_features,
    explain_customer_instance,
    ExplanationResult,
    CustomerExplanation,
)
from final_project.drift import (
    calculate_psi,
    compute_dataset_psi,
    compute_exponential_sample_weights,
)
from final_project.domain import (
    CostMatrix,
    DriftPolicy,
    RetrainingPolicy,
)
from final_project.registry import (
    ModelMetadata,
    ModelBundle,
    RiskPrediction,
)
from final_project.reporting import (
    render_evaluation_plots,
    generate_evaluation_report,
)
from final_project.simulation import (
    SimulationConfig,
    SimulationRunner,
    SimulationRunResults,
    DualDriftMonitor,
    StreamingBatchEngine,
    run_simulation,
)
from final_project.serving import (
    app,
    CustomerProfileRaw,
    CustomerFeaturesVector,
    CustomerPredictionRequest,
    SinglePredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerExplainResponse,
    HealthResponse,
    encode_raw_profile,
    encode_batch_profiles,
)

__all__ = [
    "__version__",
    "main",
    "run_training",
    "PipelineRunSummary",
    "load_and_split_data",
    "load_features_and_target",
    "DataSplit",
    "train_baseline_model",
    "TrainingResult",
    "get_candidate_model",
    "compute_scale_pos_weight",
    "evaluate_model_cv",
    "ModelCVMetrics",
    "benchmark_candidates",
    "run_tuning_study",
    "TuningResult",
    "run_evaluation",
    "compute_metrics_at_threshold",
    "optimize_threshold",
    "ThresholdMetrics",
    "OptimizationResult",
    "EvaluationSummary",
    "EvaluationReport",
    "run_explanation",
    "group_shap_by_parent_features",
    "explain_customer_instance",
    "ExplanationResult",
    "CustomerExplanation",
    "calculate_psi",
    "compute_dataset_psi",
    "compute_exponential_sample_weights",
    "CostMatrix",
    "DriftPolicy",
    "RetrainingPolicy",
    "ModelMetadata",
    "ModelBundle",
    "RiskPrediction",
    "render_evaluation_plots",
    "generate_evaluation_report",
    "SimulationConfig",
    "SimulationRunner",
    "SimulationRunResults",
    "DualDriftMonitor",
    "StreamingBatchEngine",
    "run_simulation",
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
]
