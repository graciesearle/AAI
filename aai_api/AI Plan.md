# Advanced AI Repo Work Plan (API + Container + DESD Contract)

## 1) Goal Right Now

Build a production-style inference microservice in the separate AI repo so DESD can call it reliably.

This repo should own:

- model artifact packaging (`.pth` + metadata)
- inference API runtime (`/predict`, `/health`)
- container image build/release
- contract tests against DESD expectations

This repo should not own DESD business rules (authoritative grading, lifecycle permissions, audit logic).

---

## 2) Runtime Architecture

Request flow:

1. Producer sends image to DESD endpoint.
2. DESD calls AI service `/predict`.
3. AI service loads model + runs preprocessing + inference + post-processing.
4. AI service returns quality scores and model outputs.
5. DESD computes authoritative A/B/C grade and recommendation.
6. DESD stores audit logs and returns final response.

Rule split:

- AI service may return a diagnostic `overall_grade`.
- DESD remains authoritative for final grade policy.

---

## 3) Required Inference Contract and Task Profiles

Important: contract must be profile-based, not a single fixed shape forever.

Current implemented DESD profile (quality profile, used by producer predict):

- required response fields:
  - `color_score` (0..100 float)
  - `size_score` (0..100 float)
  - `ripeness_score` (0..100 float)
  - `confidence` (float)
  - `predicted_class` (non-empty string)
- optional response fields:
  - `overall_grade` (diagnostic only)
  - `class_probabilities` (object)
  - `explanation_payload` (object)
  - `transparency_refs` (list)
  - `model_version_used` (string)
- request fields:
  - `producer_id` (required)
  - `product_id` (optional)
  - `model_version` (optional)
  - `image` (multipart file)

Future profiles (non-breaking extension approach):

- task1 recommendation profile:
  - separate endpoint and schema version
  - should not reuse task2 quality response fields
- task4 explainability profile:
  - explanation endpoint can return feature attributions, rule traces, and references
  - should be versioned independently of prediction contract

Contract rule:

- every model bundle must declare `task_profile` and `schema_version`
- DESD must reject activation if profile/schema is incompatible with endpoint contract

---

## 4) Model Artifact Handling Plan (Multi-Artifact + Version Drift Safe)

`fresh_rotten_resnet50.pth` is one artifact type, not the whole system design.

Target packaging rule:

- each model version is a bundle directory
- bundle can contain one or many artifacts
- `manifest.json` is the source of truth for how to load and serve the bundle

Recommended bundle layout:

```
models/
   <model_name>/<model_version>/
      manifest.json
      artifacts/
         model.pth
         label_map.json
         preprocessing.json
         output_schema.json
         xai_config.json
         extra_assets/...
```

Required manifest fields (minimum):

- `model_name`
- `model_version`
- `task_profile` (for example: `task2_quality`, `task1_recommendation`, `task4_xai`)
- `schema_version`
- `framework`
- `entrypoint` (loader/inference class reference)
- `artifacts` (array of `{type, path, checksum}`)
- `input_schema`
- `output_schema`
- `metrics`
- `created_at`

Loading behavior:

- service loads manifest first
- loader selected by `task_profile` + `framework`
- required artifact types validated before model activation/use
- unknown optional artifacts are ignored unless declared as required in manifest

This design handles future versions that add/remove files without breaking older bundles.

## 4.1 How DESD Should Handle Upload/Activation (Current vs Next)

Current state in DESD:

- upload endpoint stores model metadata (`model_name`, `model_version`, `manifest_json`, `artifact_path`, `checksum`)
- it does not yet unpack and validate artifact internals server-side

Planned hardening:

1. validate manifest required keys on upload
2. validate `task_profile` + `schema_version` compatibility on activation
3. keep one active version per `model_name` and/or task profile
4. block activation if required artifact descriptors are missing
5. keep compatibility matrix per DESD endpoint profile

---

## 5) Confirmed Framework and File Plan for `local/Advanced AI code REPO COPY`

Confirmed decision: AI service is also Django + DRF for consistency with DESD.

Core service files:

1. `local/Advanced AI code REPO COPY/manage.py`
2. `local/Advanced AI code REPO COPY/ai_service/settings.py`
3. `local/Advanced AI code REPO COPY/ai_service/urls.py`
4. `local/Advanced AI code REPO COPY/ai_core/config.py`
5. `local/Advanced AI code REPO COPY/ai_core/manifest.py`
6. `local/Advanced AI code REPO COPY/ai_core/views.py`

Task modules (modular by task profile):

1. Task 2 quality module
   - `local/Advanced AI code REPO COPY/task2_quality/serializers.py`
   - `local/Advanced AI code REPO COPY/task2_quality/runtime.py`
   - `local/Advanced AI code REPO COPY/task2_quality/views.py`
   - `local/Advanced AI code REPO COPY/task2_quality/urls.py`

2. Task 1 recommendation module (temporary integration boilerplate)
   - `local/Advanced AI code REPO COPY/task1_recommendation/serializers.py`
   - `local/Advanced AI code REPO COPY/task1_recommendation/runtime.py`
   - `local/Advanced AI code REPO COPY/task1_recommendation/views.py`
   - `local/Advanced AI code REPO COPY/task1_recommendation/urls.py`

3. Task 4 XAI module (temporary integration boilerplate)
   - `local/Advanced AI code REPO COPY/task4_xai/serializers.py`
   - `local/Advanced AI code REPO COPY/task4_xai/runtime.py`
   - `local/Advanced AI code REPO COPY/task4_xai/views.py`
   - `local/Advanced AI code REPO COPY/task4_xai/urls.py`

Container/runtime files:

1. `local/Advanced AI code REPO COPY/requirements.txt`
2. `local/Advanced AI code REPO COPY/Dockerfile`
3. `local/Advanced AI code REPO COPY/docker-compose.yml`
4. `local/Advanced AI code REPO COPY/.env.example`
5. `local/Advanced AI code REPO COPY/README.md`

Model bundle area:

1. `local/Advanced AI code REPO COPY/models/produce-quality/...`
2. `local/Advanced AI code REPO COPY/models/recommendation-engine/...`
3. `local/Advanced AI code REPO COPY/models/xai-engine/...`

---

## 6) Delivery Phases (Execution Order)

### Phase A: Stabilize Inference Runtime

1. Extract notebook inference logic into reusable Python modules.
2. Validate model load from `.pth` in script form.
3. Implement deterministic preprocessing + post-processing.

Exit criteria:

- local script can run single-image inference and emit required fields.

### Phase B: API Layer

1. Add `/predict` and `/health` in the chosen framework (FastAPI or Django/DRF).
2. Add request/response schemas and error handling.
3. Return required DESD contract fields.

Exit criteria:

- curl/manual request returns valid JSON contract payload.

### Phase C: Containerization

1. Add Dockerfile and image startup command.
2. Add model root/path via env variables.
3. Run local container and verify endpoints.

Exit criteria:

- image starts and serves `/health` and `/predict`.

### Phase D: Contract + Integration Validation

1. Add contract tests in AI repo.
2. Start DESD with AI profile and execute producer predict flow.
3. Pin image tag used for demo/testing.

Exit criteria:

- DESD predict endpoint works against tagged AI image.

---

## 7) Container Build/Run Plan

Local image flow:

1. Build image in AI repo:
   - `docker build -t desd-ai-service:latest .`
2. Run image with model mount:
   - `docker run --rm -p 8001:8001 -e MODEL_ROOT=/models -e DEFAULT_MODEL_NAME=produce-quality -e DEFAULT_MODEL_VERSION=1.0.0 -v $(pwd)/models:/models desd-ai-service:latest`
3. In DESD repo, run integration profile:
   - `docker compose --profile ai up -d`

Registry flow (for team handoff):

1. Tag and push image.
2. Set `AI_SERVICE_IMAGE` in DESD `.env`.
3. `docker compose --profile ai pull ai-service && docker compose --profile ai up -d`

---

## 8) Cross-Repo Consistency Rules

1. DESD authoritative grade policy is never moved into AI repo.
2. AI service contract changes must be versioned and communicated before merge.
3. Every new AI image tag must have:
   - manifest version
   - contract test results
   - model metrics snapshot
4. DESD smoke test uses a pinned AI image tag, not floating latest, for demo runs.

---

## 9) Definition Of Done For AI Repo Side

AI repo work is done when:

- `/predict` returns all required DESD fields
- model artifact bundle is versioned and documented
- image builds and runs reproducibly
- contract tests pass in CI/local
- DESD can call AI service successfully in compose profile mode

---

## 10) Immediate Next Actions (Now)

1. Move notebook inference logic into `task2_quality/runtime.py` with real model loading.
2. Replace stub responses in `task1_recommendation/runtime.py` and `task4_xai/runtime.py` with real logic.
3. Enforce strict manifest validation in `ai_core/manifest.py` for activation readiness.
4. Add contract tests for Task 2 response schema and profile checks.
5. Run containerized smoke tests and pin first stable image tag for DESD integration.
