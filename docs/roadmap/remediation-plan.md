# Remediation Roadmap

Status: **ticketed** (Parent Epic: [#50](https://github.com/Manwelws/Final-Project/issues/50))
Source: full-codebase review of `cd5e144...e66d439` (66 commits, 98 files).
Branch rule: `master` is protected by convention — every stage ships as `fix/<stage>-<slug>` → push → PR. Do not batch stages into one PR.

## Destination

Every finding from the review is resolved: code fixed where the code is wrong, docs
corrected where the docs are wrong, and the pipeline, simulator, API, image and
dashboard re-verified end to end.

## Decisions locked

| # | Decision |
|---|----------|
| D1 | Plan-then-fix; this document is the plan of record. |
| D2 | Fix **all** severities (blocker, high, medium, low). |
| D3 | Docker builds the venv from `uv.lock` with `uv sync --frozen --no-dev`; the project is installed into the venv. |
| D4 | `models/model.joblib` is **committed** (remove `*.joblib` from `.gitignore`) and SHA-256-verified at startup. An artifact registry is a follow-up ADR, not this effort. |
| D5 | "Live AUC" means: rolling 2-week AUC on labelled weeks, scored by a model frozen **before** those weeks. Docs follow the corrected run; the trigger is never retuned to hit a pre-written number. |
| D6 | The API is changed to match the documented contract; `{"items":[…]}` stays as a backwards-compatible alias. |
| D7 | Non-tree candidates (Logistic, KNN, Bagging, Voting, Stacking, MLP) are **kept and fully documented**; no "experimental" label. The benchmark itself decides the champion. |
| D8 | Threshold selection moves to a **70/10/20 train/validation/test** split; the model is refit on train+validation, the threshold comes from validation, and test is touched once for the report. |
| D9 | Dashboard mock data is removed; empty states replace fabricated runs and hardcoded KPIs. |
| D10 | API auth, CORS, rate limiting and `/docs` exposure are **out of scope** (local/demo posture). Revisit before any public deployment. |
| D11 | CI is included: GitHub Actions runs `uv sync --frozen`, `pytest`, and `docker build`. |
| D12 | Specs stay as historical intent with an "As-built deltas" section; `AGENTS.md`, `README.md`, `docs/presentation.md` and `GLOSSARY.md` are corrected in place. |

## Stages

| Stage | Issue | Theme | Depends on | Exit criterion |
|---|---|---|---|---|
| 0 | [#51](https://github.com/Manwelws/Final-Project/issues/51) | Safety net | — | Clean clone passes tests; CI green |
| 1 | [#52](https://github.com/Manwelws/Final-Project/issues/52) | ML correctness | 0 | Corrected simulator timeline; honest metrics |
| 2 | [#53](https://github.com/Manwelws/Final-Project/issues/53) | Serving contract | 0 | Documented requests work verbatim |
| 3 | [#54](https://github.com/Manwelws/Final-Project/issues/54) | Deployment | 0, 1 | `docker build` from clean clone serves traffic |
| 4 | [#55](https://github.com/Manwelws/Final-Project/issues/55) | Operations | 0 | Structured logs; no bare excepts; no leaks |
| 5 | [#56](https://github.com/Manwelws/Final-Project/issues/56) | Dashboard truthfulness | 1 | No fabricated or hardcoded data in UI |
| 6 | [#57](https://github.com/Manwelws/Final-Project/issues/57) | Docs / spec reconciliation | 1–5 | Docs match code; no stale numbers |
| 7 | [#58](https://github.com/Manwelws/Final-Project/issues/58) | Verification | 1–6 | Full suite, image, simulator, API, dashboard, re-review |

Stages 1, 2 and 4 can run in parallel once Stage 0 lands. Stage 3 needs Stage 1
because the artifact is regenerated there.

---

## Stage 0 — Safety net

Goal: make the suite trustworthy before changing behaviour.

- **R0-1 Hermetic tests.** Replace repo-relative reads of `data/processed`, `data/drift` and `models/` with fixtures that build synthetic artifacts in `tmp_path`. Affected: `tests/test_simulator_integration.py:7,42,49`, `tests/test_dashboard.py:57,88`, `tests/test_api.py:21,67`, `tests/test_data.py:7`.
- **R0-2 Assert invariants, not literals.** Drop `optimal_threshold == 0.12` and `feature_count == 27` from `tests/test_api.py:67`; account for the new 70/10/20 split in `tests/test_data.py:18-23` (currently `5634/1409`).
- **R0-3 CI.** `.github/workflows/ci.yml`: checkout → install uv → `uv sync --frozen` → `uv run pytest -q` → `docker build .`. Token already has `workflow` scope.
- **R0-4 Fix the test-count drift** in `README.md:714` (says 65; actual: 22 files, 75 tests — `README.md:9` and `:624` already say 75). `AGENTS.md:12`'s "no formal test suite" claim belongs to R6-1.

Acceptance: `git clone` into a temp dir → `uv sync --frozen` → `uv run pytest` is green with no network and no pre-existing artifacts; CI green on the PR.

---

## Stage 1 — ML correctness

Goal: every number the pipeline and simulator publish is out-of-sample and correctly labelled.

- **R1-1 In-sample decay trigger (blocker for credibility).** `simulation.py:404-416,460-465,493-500`: exclude the scored window from the training set. Fit on weeks `< current`; score on the current labelled week. Store the **current-week** AUC separately from best-ever.
- **R1-2 Never-retrain baseline leak.** `simulation.py:317`: train the never-retrain model with `label_release_week < drift_start_week` (currently `<=`, so it sees drifted week 5).
- **R1-3 Threshold leakage.** `evaluation.py:103-108` → `registry.py:174-179`: tune on validation (D8), refit on train+val, report the held-out test once. Add `val_size` to `load_and_split_data` (`data.py:29-48`).
- **R1-4 XGBoost `balanced`.** `models.py:80-86`: apply `scale_pos_weight` only when `balanced=True`; `tuning.py` must stop passing it unconditionally. Re-run the benchmark and replace the published table (`README.md:536-541`).
- **R1-5 Single-class guards.** `evaluation.py:99-101`, `simulation.py:425,437,495`, `data.py:45`: skip AUC/Brier with `None` when `y.nunique() < 2`; never abort the run.
- **R1-6 Cost matrix honoured.** `reporting.py:216-217`: render `CostMatrix` values instead of the literal `* 200` / `$20` in the observations block.
- **R1-7 Regenerate artifacts.** Re-run `churn train`, `churn evaluate`, `churn explain`, `notebooks/simulate_retraining.py`; commit the refreshed `model.joblib`, `metadata.json`, `data/drift/drift_metrics.csv`, `models/retrained/**`.

Acceptance: `drift_metrics.csv` contains no `best 1.0000`, no in-sample 1.0 rows, and a trigger timeline that follows from the data; the never-retrain curve starts before drift; the published benchmark matches a fresh `churn benchmark --all`.

---

## Stage 2 — Serving contract

Goal: the documented API works exactly as documented, and bad input fails loudly.

- **R2-1 Batch shape.** `/predict/batch` accepts a bare array; keep `{"items":[…]}` as an alias (`serving/schemas.py:108-119`, `serving/api.py:130`).
- **R2-2 Response field names.** `/explain` → `[feature, impact]`-compatible payload or documented names; `/health` → `algorithm` (`schemas.py:131-146`).
- **R2-3 Strict schemas.** Remove permissive defaults (`schemas.py:37-54`), set `extra="forbid"` (`:60`), reject unknown categoricals instead of coercing to `0` (`adapter.py:48-52`). `{}` must return 422.
- **R2-4 No silent imputation.** `adapter.py:129,138,142`: require all 27 model features; never substitute `0.0`/`3000.0`.
- **R2-5 Liveness vs readiness.** `api.py:55-58,74`: `/health` always 200 and never loads; `/ready` returns 503 until the bundle is loaded.
- **R2-6 Cache the explainer.** `api.py:197-201` → build `TreeExplainer` once per bundle; reuse across requests.
- **R2-7 SHAP exactness.** `explainability.py:160-174`: non-tree winners must use an exact/consistent explainer (or the response must state it is approximate). Test the fallback.
- **R2-8 Second waterfall.** `explainability.py:318`: plot a false-positive (predicted churn, actually retained) customer.
- **R2-9 CLI flag bug.** `cli.py:240`: `--headless` is `store_true, default=True` — make it a real toggle (`--headless/--no-headless`).

Acceptance: every request/response example in `README.md` runs verbatim; `{}`, unknown keys, missing features and single-class batches return documented errors; `/explain` latency is stable under repeated calls.

---

## Stage 3 — Deployment

Goal: a clean clone builds an image that serves without host tricks.

- **R3-1 Dockerfile.** `Dockerfile:15-16,53`: `COPY pyproject.toml uv.lock src/`; `uv sync --frozen --no-dev`; run `/app/.venv/bin/uvicorn`; keep the non-root user; add `PYTHONPATH` only if the install genuinely needs it.
- **R3-2 Artifact in the image.** Commit `models/model.joblib` (D4), remove/scope `.gitignore:20`, `COPY models/`, verify SHA-256 on startup and fail loudly on mismatch (`registry.py:125`).
- **R3-3 Reproducibility.** Pin exact dependency versions in `pyproject.toml:11-29`; pin both base images by digest (`Dockerfile:3,19`).
- **R3-4 Compose.** Keep the dev bind-mount working; ensure `docker-compose.yml` still starts a healthy container without it.
- **R3-5 Azure guide.** `docs/deployment/azure.md`: ACR build now includes the artifact; document the SHA check and the healthcheck.

Acceptance: `docker build .` succeeds from a fresh clone; `docker run` serves `/health`, `/predict`, `/explain`; compose up is healthy; the image contains `model.joblib`.

---

## Stage 4 — Operations

Goal: failures are visible and nothing leaks.

- **R4-1 Structured logging.** Replace the ~72 `print()` calls in `src/` with `logging`; log model load, validation failures, retrain triggers and explain timings. `api.py:36-38` first.
- **R4-2 Specific exceptions.** `registry.py:131`, `dashboard/state.py:76`, `explainability.py:131`, `evaluation.py:316`: no bare `except Exception`; log the path and continue only where intended.
- **R4-3 Resource hygiene.** Close figures (`reporting.py:65,82,98,121,145`) and use `with open(...)` for JSON writes (`simulation.py:489`).

Acceptance: no `print(` under `src/`; no `except Exception` without a specific handler; a long simulation run leaks no file descriptors; startup/load failures appear as ERROR logs.

---

## Stage 5 — Dashboard truthfulness

Goal: the UI never shows invented data.

- **R5-1 Remove mock MLflow.** `dashboard/state.py:116-210`: delete `generate_mock_mlflow_runs`; return an explicit "no runs found — run `churn tune`" state.
- **R5-2 Remove hardcoded values.** `dashboard/app.py:83` champion string; `dashboard/pages/kpi_overview.py:390-393` tiles — compute from `metadata.json` / artifacts.
- **R5-3 Correct drift plot.** `dashboard/pages/drift_monitor.py:37,224`: plot the current-week live AUC; show best-ever separately and labelled as such.
- **R5-4 Honest empty states.** Every page renders a usable message when artifacts or tracking are absent.

Acceptance: with no `mlflow.db` and no retrained artifacts, every page shows a truthful empty state; with real artifacts, no number is hardcoded.

---

## Stage 6 — Docs and spec reconciliation

Goal: stop the docs lying, without erasing history.

- **R6-1 `AGENTS.md`.** Fix: no-test-suite claim (`:12`), `kodeagnt`/empty README (`:38-39`), DECISION/ALTERNATIVES/WHY blocks now in `simulation.py` (`:44`), weeks 3 & 7 timeline, never-retrain ~0.847.
- **R6-2 `README.md`.** API contracts (post-Stage 2), full candidate list incl. MLP (D7), corrected benchmark table (R1-4), corrected never-retrain AUC, test count.
- **R6-3 Specs.** Add "As-built deltas" to `specs/implementation_plan.md`, `milestone_2_spec.md`, `milestone_3_spec.md`, `milestone_4_spec.md`: MLflow store (`sqlite:///mlflow.db`), artifact path (`models/retrained/week_N`), candidate set, timeline.
- **R6-4 Presentation.** `docs/presentation.md:139` (speaker script): fix the "weeks 5, 7, and 8" trigger claim to the corrected timeline.
- **R6-5 Glossary.** Sharpen the entries the review exposed: **live AUC**, **reference data**, **decay trigger**, **threshold** (cost-optimal). Glossary stays implementation-free.
- **R6-6 ADR 0002.** "Model artifacts are committed and hash-verified; registry is the follow-up" — hard to reverse, surprising, a real trade-off.

Acceptance: `grep` finds no stale trigger week, test count, threshold or candidate claims; a new reader can follow README end to end without hitting a 422.

---

## Stage 7 — Verification

- **R7-1** `uv run pytest -q` from a clean clone (no artifacts) and with artifacts.
- **R7-2** `docker build .` + `docker run`; smoke `/health`, `/ready`, `/predict`, `/predict/batch` (both shapes), `/explain`.
- **R7-3** `python notebooks/simulate_retraining.py`; assert the trigger timeline and that no `best 1.0` appears.
- **R7-4** Dashboard smoke with and without `mlflow.db`.
- **R7-5** Re-run the code review (`/matt/code-review`) against `e66d439...HEAD` and confirm every finding is closed or explicitly out of scope.

Acceptance: all five green; the re-review reports zero blockers and zero unexplained highs.

---

## Out of scope (this effort)

- **API auth, CORS allowlist, rate limiting, hiding `/docs`** (D10). Recorded as a pre-public-deployment requirement, not a demo blocker.
- **Remote model registry** (MLflow/OCI/Blob). Superseded for now by D4; revisit via ADR 0002's follow-up.
- **Branch protection, release automation, observability backend.** Not review findings.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| R1-1 changes the documented trigger timeline, which the report and slides quote. | Keep a before/after table in the PR; update `presentation.md` and `AGENTS.md` in the same PR that changes the math. |
| D8 shrinks the training slice from 80% to 70%. | Refit on train+validation before evaluation, so the deployed model still sees 80%; the 10% is held out only for threshold selection. |
| Committing `model.joblib` grows git history by ~16 MB per retrain. | Commit only on release-worthy artifacts, not every local run; record the registry follow-up in ADR 0002. |
| Hermetising tests (R0-1) could mask a real regression. | Preserve each assertion's intent; only replace the fixture source, never the expectation. |
| Stage 1 and Stage 5 both touch drift metrics. | Land Stage 1 first; Stage 5 consumes the corrected column names. |
| Pinning deps to exact versions may surface latent incompatibilities (Python 3.14 + ML stack). | Pin from the existing `uv.lock`, run the full suite before merging, and bump one package per PR if a conflict appears. |

## Traceability

| Finding | Stage |
|---|---|
| Docker build/start failure | 3 (R3-1) |
| Model artifact missing from clean build | 3 (R3-2) |
| In-sample decay AUC / spurious weeks 3,5,7,8 | 1 (R1-1) |
| Mislabeled `retrain_model_auc_recent` | 1 (R1-1), 5 (R5-3) |
| Never-retrain baseline sees drift | 1 (R1-2) |
| Threshold fitted on test | 1 (R1-3) |
| "Unweighted" XGBoost is weighted | 1 (R1-4) |
| Single-class AUC crash | 1 (R1-5) |
| Report ignores cost matrix | 1 (R1-6) |
| README/API contract mismatch (batch, explain, health) | 2 (R2-1,2), 6 (R6-2) |
| `{}` accepted; missing features → 0.0 | 2 (R2-3,4) |
| `/health` reloads and conflates readiness | 2 (R2-5) |
| SHAP explainer rebuilt per request | 2 (R2-6) |
| SHAP not exact for non-tree | 2 (R2-7) |
| Wrong second waterfall | 2 (R2-8) |
| `--headless` cannot be false | 2 (R2-9) |
| Untrusted `joblib.load` | 3 (R3-2) |
| Unpinned deps + unused lockfile | 3 (R3-3) |
| No logging / observability | 4 (R4-1) |
| Swallowed exceptions | 4 (R4-2) |
| Unclosed figures/handles | 4 (R4-3) |
| Fabricated MLflow + hardcoded KPIs | 5 (R5-1,2) |
| Stale AGENTS/README/specs/presentation | 6 |
| Tests depend on untracked artifacts; no CI | 0 (R0-1,3) |
| Test-count drift in docs | 0 (R0-4), 6 |
| Scope creep: MLP/non-tree | 6 (document, D7) |

## Suggested execution order

1. R0-1..R0-4 (one PR).
2. R1-1..R1-7 (one PR; the largest).
3. In parallel after R0: Stage 2 PR, Stage 4 PR.
4. R3-1..R3-5 (after Stage 1).
5. R5-1..R5-4 (after Stage 1).
6. R6-1..R6-6 (after 1–5).
7. R7-1..R7-5.

## Open items still needing your call

- None blocking. Q11 (CI) is assumed **yes**; say no and R0-3 is dropped.
