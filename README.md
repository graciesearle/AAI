# Advanced AI Django Service Scaffold

This folder is now a Django + DRF AI service scaffold for Tasks 1, 2, and 4.

## Current status

Ready for development as a modular baseline:

- Django project configured and containerized
- Task 2 quality endpoint scaffolded
- Task 1 recommendation endpoint scaffolded
- Task 4 XAI endpoint scaffolded
- Manifest-based model bundle loader scaffolded

## Endpoint map

- GET /api/health/
- POST /api/task2/predict/
- POST /api/task1/recommend/
- POST /api/task4/explain/

## Run locally

1. Create env file from template:

cp .env.example .env

2. Start service:

```bash
docker compose up -d --build
```

3. Verify health:

```bash
curl http://localhost:8001/api/health/
```

## Build image for DESD integration

Build this service as the image name DESD expects:

```bash
docker build -t desd-ai-service:latest .
```

Or run the helper script (recommended):

```bash
./setup_ai.sh
```

What the helper does:

- builds the AI image
- prints a ready-to-paste DESD `.env` block
- copies that block to clipboard automatically (when clipboard utility is available)
- prints the exact DESD run and health-check CLI commands

## How image generation and storage works (local flow)

- The image is not stored in the DESD repo folder.
- The image is stored in your local Docker image store (managed by Docker Desktop/Engine).
- The image is identified by tag (default: `desd-ai-service:latest`).
- DESD references that tag via `AI_SERVICE_IMAGE` and starts it through Docker Compose.

So there is no manual "copy image into DESD" step.

What each developer does locally:

1. Run `./setup_ai.sh` in the AI repo.
2. This builds/updates local image tag `desd-ai-service:latest`.
3. Paste the printed env block into DESD `.env` (config only, not image data).
4. Run DESD with `docker compose --profile ai up -d --build`.

Useful verification commands:

```bash
docker image ls desd-ai-service
docker compose ps
curl http://localhost:8001/api/health/
```

If AI code changes, rebuild by running `./setup_ai.sh` again.
The tag is refreshed locally and DESD will use the updated image on next startup.

Portable usage options (for separate repos / multiple developers):

```bash
# optional: point explicitly to DESD repo path
DESD_ROOT=/absolute/path/to/DESD ./setup_ai.sh

# optional: tag for remote registry and include that tag in the printed env block
AI_REMOTE_IMAGE=ghcr.io/<org-or-user>/desd-ai-service:latest ./setup_ai.sh

# optional: also push to remote registry
AI_REMOTE_IMAGE=ghcr.io/<org-or-user>/desd-ai-service:latest AI_PUSH_REMOTE=1 ./setup_ai.sh
```

Location note:

- DESD does not use local filesystem paths to find the AI service.
- DESD only needs the image tag (`AI_SERVICE_IMAGE`) and service URL/path env keys.
- That means AI and DESD repos can live in different directories on each developer machine.

## Connect to DESD (full wiring)

1. In DESD `.env`, set these values:

```env
AI_INFERENCE_BASE_URL=http://ai-service:8001
AI_INFERENCE_PREDICT_PATH=/api/task2/predict/
AI_SERVICE_IMAGE=desd-ai-service:latest
```

Note:

- `AI_INFERENCE_PREDICT_PATH` is currently for Task 2 producer quality integration only.
- DESD currently calls only Task 2 from its AI client.
- When Task 1 and Task 4 are wired in DESD, add separate endpoint keys (for example `AI_RECOMMEND_PATH` and `AI_EXPLAIN_PATH`) and separate client methods for those contracts.

2. Start DESD with the AI profile:

```bash
docker compose --profile ai up -d --build
```

3. Validate from DESD side:

```bash
docker compose ps
```

4. Run a producer prediction through DESD endpoint:

- POST to `/api/ai/producer-quality/predict/` from DRF browsable API or API client.
- DESD then calls this service at `/api/task2/predict/` inside the compose network.

## How to test Task 2 now

The current Task 2 runtime attempts trained-model inference first and falls back
to image-signal inference when checkpoint loading fails.
Use these tests to validate contract and DESD integration.

Model training and lifecycle upload instructions are documented here:

- `docs/task2/training_and_upload.md`

This guide includes dynamic versioning plus CLI and environment-variable overrides.

### A) Direct AI service test (quick contract check)

From DESD root:

```bash
curl -sS -X POST 'http://localhost:8001/api/task2/predict/' \
	-F producer_id=1 \
	-F image=@marketplace/static/marketplace/images/default-product.jpg
```

Expected keys in response:

- `color_score`, `size_score`, `ripeness_score`, `confidence`, `predicted_class`

### B) End-to-end DESD -> AI test

In browser (recommended):

1. Log in as a producer in DESD.
2. Open `/api/ai/producer-quality/predict/`.
3. Upload an image and submit.
4. Confirm response is `201` and includes authoritative grade and recommendation.

CLI alternative (inside DESD web container):

```bash
docker compose exec -T web python manage.py shell <<'PY'
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient

if 'testserver' not in settings.ALLOWED_HOSTS:
		settings.ALLOWED_HOSTS.append('testserver')

User = get_user_model()
producer = User.objects.filter(role=User.Role.PRODUCER).first()
client = APIClient()
client.force_authenticate(user=producer)

with open('/app/marketplace/static/marketplace/images/default-product.jpg', 'rb') as f:
		image = SimpleUploadedFile('default-product.jpg', f.read(), content_type='image/jpeg')
		response = client.post('/api/ai/producer-quality/predict/', {'image': image}, format='multipart')

print('STATUS', response.status_code)
print('BODY', getattr(response, 'data', response.content.decode('utf-8', errors='ignore')))
PY
```

## Readiness status

Current state is integration-ready with real Task 2 inference wiring and Task 3 lifecycle APIs.

Ready now:

- container runs and exposes all required task endpoints
- DESD can call Task 2 predict endpoint with current contract
- task modules are separated for Task 1, 2, and 4

Still to implement:

- train and activate a production-quality Task 2 checkpoint for default runtime usage
- harden Task 3 endpoint authn/authz policy for production deployment
- real recommendation/XAI logic for Task 1 and Task 4
- broader performance and stress tests beyond current contract coverage

## Development priorities

1. Implement real Task 2 model runtime in task2_quality/runtime.py
2. Add strict manifest/schema validation gates in ai_core/manifest.py
3. Replace Task 1 and Task 4 stub runtimes with real model/explainability logic
4. Add contract tests matching DESD integration requirements

## Model bundle convention

Model bundles are versioned under models/<model_name>/<model_version>/manifest.json.

Three starter bundle slots are included:

- models/produce-quality/1.0.0 (Task 2)
- models/recommendation-engine/0.1.0 (Task 1)
- models/xai-engine/0.1.0 (Task 4)
