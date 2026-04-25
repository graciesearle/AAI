# Advanced AI Repository

- `GET /api/health/`
- `POST /api/task1/recommend/`
- `POST /api/task2/predict/`
- `GET/POST /api/task3/...` (Lifecycle management)
- `GET/PATCH /api/task3/interactions/` (Interaction logging & Feedback)
- `POST /api/task4/explain/`

## Local Development

Use venv, then run AAI from `AAI/aai_api`:

```bash
python manage.py migrate
python manage.py runserver 8001
```

Django settings module is namespaced as `aai_api.ai_service.settings` through `manage.py`.

### First-Time Setup With DESD (AAI Local, DESD Docker)

Use this when AAI runs locally on your machine and DESD runs in Docker.

1. Create an AAI user (first time only) and generate a token:

```bash
python manage.py createsuperuser
python manage.py drf_create_token <username>
```

2. Update DESD `.env` with integration settings:

```dotenv
AI_INFERENCE_BASE_URL=http://host.docker.internal:8001
AI_LIFECYCLE_BASE_URL=http://host.docker.internal:8001
AI_LIFECYCLE_TOKEN=<token-from-drf_create_token>
```

3. Recreate DESD services to load `.env` changes:

```bash
cd DESD
docker compose up -d --force-recreate web scheduler
```

4. Verify AAI token auth from your host:

```bash
curl -H "Authorization: Token <token-from-drf_create_token>" http://localhost:8001/api/task3/models/
```

### Base URL Quick Reference

- AAI local + DESD Docker: `AI_LIFECYCLE_BASE_URL=http://host.docker.internal:8001`
- AAI Docker service + DESD Docker (same network): `AI_LIFECYCLE_BASE_URL=http://ai-service:8001`

Set `AI_INFERENCE_BASE_URL` to the same host as `AI_LIFECYCLE_BASE_URL` for consistent Task 2 + Task 3 routing.

## Docker

Run with compose file under `aai_api/`:

```bash
docker compose -f aai_api/docker-compose.yml up -d --build
```

Model artifacts are persisted with a named Docker volume mounted at `/app/models`
in `aai_api/docker-compose.yml`. This protects lifecycle bundles from container
restarts while keeping local code bind-mounted for development.

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

### First-Time Setup With DESD (AAI Docker Image, DESD Docker)

Use this when you want DESD to run against the `desd-ai-service:latest` image.

1. Build the AAI image from repository root:

```bash
docker build -f aai_api/Dockerfile -t desd-ai-service:latest .
```

2. Start AAI once (from `AAI/aai_api`) so you can create token-backed auth data:

```bash
docker compose up -d --build
docker compose exec ai-web python manage.py migrate
docker compose exec ai-web python manage.py createsuperuser
docker compose exec ai-web python manage.py drf_create_token <username>
```

3. Copy the token value and set DESD `.env` (from `DESD/.env`):

```dotenv
AI_SERVICE_IMAGE=desd-ai-service:latest
AI_INFERENCE_BASE_URL=http://ai-service:8001
AI_LIFECYCLE_BASE_URL=http://ai-service:8001
AI_LIFECYCLE_TOKEN=<token-from-drf_create_token>
```

4. Start DESD with the AI profile (from `DESD/`):

```bash
docker compose --profile ai up -d --build
docker compose up -d --force-recreate web scheduler
```

5. Verify token auth path is working:

```bash
docker compose --profile ai exec ai-service python manage.py check
curl -H "Authorization: Token <token-from-drf_create_token>" http://localhost:8001/api/task3/models/
```

Notes:

- Tokens are stored in the AAI database, not baked into the Docker image layer.
- If you rebuild/recreate AAI with a new database, regenerate token and update `DESD/.env`.
- Use `AI_LIFECYCLE_BASE_URL=http://ai-service:8001` only when DESD and AAI run in the same Docker network.

### Persistence Smoke Check

Use this to verify a model bundle is still present after container restart.

PowerShell (from repository root):

```powershell
.\local\task3_persistence_smoke.ps1 -ModelName produce-quality -ModelVersion persistence-smoke-v1
```

Direct management command usage:

```bash
python manage.py lifecycle_persistence_smoke --model-name produce-quality --model-version persistence-smoke-v1
python manage.py lifecycle_persistence_smoke --model-name produce-quality --model-version persistence-smoke-v1 --verify-only
```

## Quick Build For DESD

From AAI repo root:

```bash
docker build -f aai_api/Dockerfile -t desd-ai-service:latest .
```

From DESD folder:

```bash
docker compose --profile ai up -d --build
```

## Ownership Split (Recommended)

- API + Integration Owner: `aai_api/`
- Task 1 Logic Owner: `task1/`
- Task 2 & 4 Logic Owner: `task2_3_4/`
- Task 3 Implementation Owner: `aai_api/` (Service level)

## Notes

- DESD policy decisions remain outside this repo.
- This repo focuses on AI model/runtime behavior, lifecycle integration APIs, and contracts.

## Task 3 Auth Documentation

- See `docs/task3_lifecycle_api_auth.md` for secured Task 3 endpoint list,
  token setup, request examples, and DESD integration verification steps.
