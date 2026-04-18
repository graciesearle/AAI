# Task 1 Hook-Up Guide (Plugin-Only Folder)

This folder is intentionally API-light. You only need to maintain:

- `runtime.py` (core inference/recommendation logic)
- `serializers.py` (request/response contract)

The API endpoint is handled in `aai_api/api_adapters/task1.py`.

## Quick Start: Move Your Local AI Code Here

1. Copy your local Task 1 python file(s) into this folder.
2. Keep your heavy logic in `runtime.py` (or import your module from there).
3. Ensure `runtime.py` exposes:

```python
build_recommendations(model_name, model_version, recent_items, manifest)
```

4. Return a dict that matches `RecommendationResponseSerializer` in `serializers.py`.

## How It Connects to API

- Endpoint: `POST /api/task1/recommend/`
- Adapter: `aai_api/api_adapters/task1.py`
- The adapter validates input, loads optional manifest, calls your runtime.

No URL/view changes are needed in this folder.

## Model Bundle (Optional)

If you ship model metadata/artifacts, place them under model root with:

- `<model_root>/recommendation-engine/<version>/manifest.json`

If missing, Task 1 still works with runtime defaults.

## Minimal Update Checklist

- [ ] Runtime function signature kept stable.
- [ ] Output shape matches serializer.
- [ ] Any new dependencies added to `aai_api/requirements.txt`.
- [ ] Run tests from repo root:

```bash
pytest aai_api/tests/test_smoke_api.py -q
```
