# Task 4 (XAI) Hook-Up Guide (Plugin-Only Folder)

This folder is intentionally API-light. Keep only:

- `runtime.py` (explanation logic)
- `serializers.py` (request/response contract)

The API endpoint is in `aai_api/api_adapters/task4.py`.

## Quick Start: Move Your Local XAI Code

1. Copy your local XAI logic into this folder.
2. Keep external integrations inside `runtime.py` or modules imported by it.
3. Ensure `runtime.py` exposes:

```python
build_explanation(model_name, model_version, context, manifest)
```

4. Return output shape expected by `ExplainResponseSerializer`.

## How It Hooks to API

- Endpoint: `POST /api/task4/explain/`
- Adapter: `aai_api/api_adapters/task4.py`
- Adapter handles validation + optional manifest loading, then calls your runtime.

No Django view/url files are needed in this folder.

## Versioning and Activation

Task 4 can use explicit `model_version` in request body.

If you want activation behavior similar to Task 2, upload a manifest/version using lifecycle APIs and pass that version to Task 4 requests.

## Minimal Update Checklist

- [ ] Runtime signature stable.
- [ ] Output matches serializer.
- [ ] Any new package added to `aai_api/requirements.txt`.
- [ ] Validate:

```bash
pytest aai_api/tests/test_smoke_api.py -q
```
