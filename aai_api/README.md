# Advanced AI Repository 

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

Use venv and then cd aai_api\
python manage.py runserver 8001
Instead of docker so we don't need to build it for DESD (quick development)

Because of this we need to change the DESD env.
 and add this line to the bottom:
`AI_INFERENCE_BASE_URL=http://host.docker.internal:8001`


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
