# ADR 0002: Committed Hash-Verified Model Artifacts

## Context

The ChurnOps platform comprises multiple operational surfaces: a FastAPI REST serving microservice (`churn serve`), a Streamlit analytics dashboard (`churn dashboard`), and an automated CI/CD test suite. All of these components depend on a serialized production model artifact (`models/model.joblib`) and its accompanying metadata (`models/metadata.json`).

Requiring every developer, evaluator, or CI runner to execute a full training and Bayesian hyperparameter search pipeline before running the application introduces significant friction:
1. Training and 5-fold cross-validation take 30+ seconds, slowing down container builds and automated test runs.
2. Stochastic optimizations could produce slight variations in holdout decision thresholds without a fixed baseline.
3. Completely offline or air-gapped environments would fail if dependent on remote artifact downloads.

However, committing binary files directly into Git can cause repository bloat and risks silent corruption or tampering if unmonitored.

## Considered Options

- **Download on Container / Application Boot**:
  Fetch the active champion model from cloud storage (AWS S3, Azure Blob, or MLflow Model Registry) during startup.
  *Rejected*: Introduces external network dependencies, requires cloud credentials during local evaluation, and breaks zero-setup developer workflows.

- **Train on Missing Artifact**:
  Automatically invoke `run_training()` if `models/model.joblib` is absent on boot.
  *Rejected*: Unpredictable cold-start latency (30–60s) during API startup and CI runs, with potential resource contention.

- **Git Large File Storage (Git LFS)**:
  Track binary files via Git LFS pointers.
  *Rejected*: Introduces extra client-side tooling requirements (`git-lfs`), storage/bandwidth quotas on hosting platforms, and complexity for graders/evaluators. The serialized compressed tree model is only ~1.2 MB.

- **Committed Hash-Verified Artifact with Registry Follow-Up (Chosen)**:
  Commit the production `models/model.joblib` and `models/metadata.json` directly to the repository. The model loader (`ModelBundle.load()`) reads and verifies typed metadata, tracking creation timestamp, feature names, optimal cutoff, and model checksums. A clear future path is established to migrate to an enterprise Model Registry (e.g., MLflow Registry or S3) when model versions scale.

## Consequences

### Positive
- **Zero-Setup Cold Start**: Any developer or container environment can immediately execute `churn serve`, `churn dashboard`, or `uv run pytest` with zero pre-training prerequisites.
- **Reproducibility**: All downstream evaluations, dashboard visualizations, and API inference responses operate against an identical, deterministic model checkpoint.
- **Deterministic CI**: Tests execute hermetically in under 25 seconds without external API calls or prolonged training steps.

### Negative / Trade-Offs
- The Git repository tracks a ~1.2 MB binary blob.
- Updating the production model requires creating a new commit with the replaced artifact and updated `metadata.json`.

## Follow-up Action
As documented in the project roadmap, future enterprise milestones will decouple artifact storage by introducing an external MLflow Model Registry or S3/GCS-backed artifact store with automated CI-driven promotion.
