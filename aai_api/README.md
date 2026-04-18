# Advanced AI Repository (Handoff-Ready)

This repository is organized by delivery responsibility so Task 1, Task 2/3/4, and API integration work can be handed off independently.

## Top-Level Layout

- `aai_api/`: Django/DRF service shell, container files, health/config, integration tests.
- `task1/`: Task 1 recommendation implementation.
- `task2_3_4/`: Task 2 quality, Task 3 lifecycle, Task 4 XAI, and Task 2 raw/training assets.

## Why Task 2/3/4 Are Grouped

Assessment guidance in `task2_3_4/reference/AAI_DOCS/faqs.md` states the AI repository may be split as:

- One area for Task 1
- One area for Tasks 2, 3, and 4

This structure follows that requirement and keeps CV + lifecycle + XAI together.

## Task 2 Implementation Status

Task 2 post-processing logic has been intentionally reset to a blank slate for the next implementation pass.

- Current runtime baseline: `task2_3_4/task2_quality/runtime.py`
- Next implementation owner can add quality scoring and grading logic there (or in helper modules it imports).

## API Endpoints

- `GET /api/health/`
- `POST /api/task1/recommend/`
- `POST /api/task2/predict/`
- `POST /api/task3/...` (lifecycle endpoints)
- `POST /api/task4/explain/`

## Local Development

From repository root:

```bash
python manage.py migrate
python manage.py runserver
```

Django settings module is now namespaced as `aai_api.ai_service.settings` through `manage.py`.

## Docker

Run with compose file under `aai_api/`:

```bash
docker compose -f aai_api/docker-compose.yml up -d --build
```

Health check:

```bash
curl http://localhost:8001/api/health/
```

Build image for DESD integration:

```bash
docker build -f aai_api/Dockerfile -t desd-ai-service:latest .
```

Or run helper:

```bash
./aai_api/setup_ai.sh
```

# Simply build for DESD

- Check that in AAI:
  python manage.py check: success
  python manage.py migrate: success

- Then From AAI/aai_api folder:
  docker build -t desd-ai-service:latest .
  From DESD folder (with profile):
  docker compose --profile ai up -d --build

## Ownership Split (Recommended)

- API + Integration Owner: `aai_api/`
- Task 1 Owner: `task1/`
- Task 2/3/4 Owner(s): `task2_3_4/`

## Notes

- DESD policy decisions remain outside this repo.
- This repo focuses on AI model/runtime behavior, lifecycle integration APIs, and contracts.
