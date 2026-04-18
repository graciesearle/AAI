# Task 2 Hook-Up Guide (Plugin-Only Folder)

This folder is intentionally API-light. Keep only domain code such as:

- `runtime.py`
- `model_inference.py`
- `postprocess.py`
- `serializers.py`

The API endpoint and lifecycle operations are in `aai_api/api_adapters/`.

## Quick Start: Move Local AI Code Into This Workspace

1. Copy your trained/inference code into this folder.
2. Keep model execution in `model_inference.py` and orchestration in `runtime.py`.
3. Ensure `runtime.py` exposes:

```python
run_quality_inference(image_file, model_root, model_name, model_version, manifest)
```

4. Ensure output keys match `QualityPredictResponseSerializer` in `serializers.py`.

## How It Hooks to API

- Prediction endpoint: `POST /api/task2/predict/`
  - Adapter: `aai_api/api_adapters/task2.py`
- Lifecycle endpoints:
  - `GET /api/task3/models/`
  - `POST /api/task3/models/upload/`
  - `POST /api/task3/models/activate/`
  - `POST /api/task3/models/rollback/`
  - Adapter: `aai_api/api_adapters/task3.py`

You do not need to create or edit Django views in this folder.

## Upload + Activate Model (Fast Path)

1. Produce checkpoint file (e.g. `model.pth`).
2. Upload via Task 3 API.
3. Activate uploaded version.
4. Call Task 2 predict without `model_version` to use active version.

## Manifest Contract

Model bundles are stored as:

- `<model_root>/produce-quality/<version>/manifest.json`
- `<model_root>/produce-quality/<version>/artifacts/model.pth`

Entrypoint in manifest should point to runtime function:

- `task2_3_4.task2_quality.runtime:run_quality_inference`

## Minimal Update Checklist

- [ ] Runtime signature stable.
- [ ] Manifest and artifact path valid.
- [ ] Checkpoint load works in `model_inference.py`.
- [ ] New deps added to `aai_api/requirements.txt`.
- [ ] Validate:

```bash
pytest aai_api/tests/test_task2_contract.py aai_api/tests/test_task3_lifecycle_api.py -q
```
