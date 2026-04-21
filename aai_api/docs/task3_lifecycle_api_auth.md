# Task 3 Lifecycle And Interaction API Auth Guide

This document defines the secured Task 3 API surface in AAI and how clients (including DESD) should authenticate.

## Base URL

- Local Docker: http://localhost:8001
- Docker network (DESD -> AAI service): http://ai-service:8001

## Secured Endpoints

All endpoints below use DRF TokenAuthentication and require an Authorization header.

- GET /api/task3/models/
- POST /api/task3/models/upload/
- POST /api/task3/models/activate/
- POST /api/task3/models/rollback/
- GET /api/task3/interactions/
- PATCH /api/task3/interactions/<id>/override/

Expected behavior:

- Missing token: 401 Unauthorized with detail saying credentials were not provided.
- Valid token: endpoint returns normal JSON payload.

## Create A Token

1. Create a user (if needed):
   - python manage.py createsuperuser
2. Generate token for that username:
   - python manage.py drf_create_token <username>

You should receive a token string to use in Authorization headers.

## Auth Header Format

Authorization: Token <token_value>

## Example Requests

PowerShell:

- Invoke-RestMethod -Method Get -Uri http://localhost:8001/api/task3/models/ -Headers @{ Authorization = "Token <token_value>" }

Curl:

- curl -H "Authorization: Token <token_value>" http://localhost:8001/api/task3/models/

Interaction override example:

- curl -X PATCH \
  -H "Authorization: Token <token_value>" \
  -H "Content-Type: application/json" \
  -d '{"producer_accepted": false, "override_grade": "B"}' \
  http://localhost:8001/api/task3/interactions/1/override/

## DESD Integration Notes

DESD lifecycle synchronization reads these settings:

- AI_LIFECYCLE_BASE_URL
- AI_LIFECYCLE_TOKEN
- AI_MODEL_LIST_PATH
- AI_MODEL_UPLOAD_PATH
- AI_MODEL_ACTIVATE_PATH
- AI_MODEL_ROLLBACK_PATH

DESD adds Authorization: Token <AI_LIFECYCLE_TOKEN> on lifecycle client requests automatically.

## Quick Verification From DESD

Run in DESD container:

- docker compose exec web python manage.py reconcile_ai_lifecycle

If token wiring is missing or invalid, this command fails with an AAI lifecycle auth error.
If token wiring is correct, it returns lifecycle drift/report output from AAI.
