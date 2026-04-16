#!/bin/bash

# AI repo setup helper
# Builds the AI service image and prints/copies the DESD env keys needed for integration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESD_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DESD_ROOT="${DESD_ROOT:-$DESD_ROOT_DEFAULT}"
IMAGE_TAG="${AI_IMAGE_TAG:-desd-ai-service:latest}"
REMOTE_IMAGE_TAG="${AI_REMOTE_IMAGE:-}"
PUSH_REMOTE="${AI_PUSH_REMOTE:-0}"
FINAL_IMAGE_TAG="${IMAGE_TAG}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

if ! command -v docker >/dev/null 2>&1; then
    echo -e "${RED}Error:${NC} Docker is not installed or not on PATH."
    exit 1
fi

if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
else
    echo -e "${RED}Error:${NC} Neither 'docker compose' nor 'docker-compose' is available."
    exit 1
fi

echo -e "${GREEN}Step 1:${NC} Building AI image '${IMAGE_TAG}'..."
docker build -t "${IMAGE_TAG}" "${SCRIPT_DIR}"

if [ -n "${REMOTE_IMAGE_TAG}" ]; then
    echo -e "${GREEN}Step 1b:${NC} Tagging image for remote registry as '${REMOTE_IMAGE_TAG}'..."
    docker tag "${IMAGE_TAG}" "${REMOTE_IMAGE_TAG}"

    case "${PUSH_REMOTE}" in
        1|true|TRUE|yes|YES)
            echo -e "${GREEN}Step 1c:${NC} Pushing image to remote registry..."
            docker push "${REMOTE_IMAGE_TAG}"
            ;;
    esac

    FINAL_IMAGE_TAG="${REMOTE_IMAGE_TAG}"
fi

env_block=$(cat <<EOF
# --- AI service wiring (Task 2 active now) ---
AI_SERVICE_IMAGE=${FINAL_IMAGE_TAG}
AI_INFERENCE_BASE_URL=http://ai-service:8001
AI_INFERENCE_PREDICT_PATH=/api/task2/predict/
AI_INFERENCE_TIMEOUT_SECONDS=15

# --- Optional future wiring (Task 1 + Task 4) ---
# AI_RECOMMEND_BASE_URL=http://ai-service:8001
# AI_RECOMMEND_PATH=/api/task1/recommend/
# AI_EXPLAIN_BASE_URL=http://ai-service:8001
# AI_EXPLAIN_PATH=/api/task4/explain/
EOF
)

echo
echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  COPY THIS BLOCK INTO DESD .env${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "${env_block}"
echo -e "${CYAN}==============================================================${NC}"
echo

clipboard_ok=false
if command -v pbcopy >/dev/null 2>&1; then
    printf "%s\n" "${env_block}" | pbcopy
    clipboard_ok=true
elif command -v xclip >/dev/null 2>&1; then
    printf "%s\n" "${env_block}" | xclip -selection clipboard
    clipboard_ok=true
elif command -v xsel >/dev/null 2>&1; then
    printf "%s\n" "${env_block}" | xsel --clipboard --input
    clipboard_ok=true
fi

if [ "${clipboard_ok}" = true ]; then
    echo -e "${GREEN}Clipboard:${NC} DESD env block copied successfully."
else
    echo -e "${YELLOW}Clipboard:${NC} No clipboard utility found; copy the block manually."
fi

echo
echo -e "${GREEN}Step 2:${NC} Run DESD with AI profile (CLI commands):"
if [ -f "${DESD_ROOT}/docker-compose.yml" ]; then
    echo "  cd \"${DESD_ROOT}\""
else
    echo -e "${YELLOW}Note:${NC} DESD root not auto-detected from this repo location."
    echo "  Set DESD root explicitly, for example:"
    echo "  export DESD_ROOT=/absolute/path/to/DESD"
    echo "  cd \"\${DESD_ROOT}\""
fi
echo "  ${COMPOSE_CMD} --profile ai up -d --build"
echo "  ${COMPOSE_CMD} ps"

echo
echo -e "${GREEN}Step 3:${NC} Verify AI health (CLI):"
echo "  curl http://localhost:8001/api/health/"
echo "  ${COMPOSE_CMD} logs --tail 80 ai-service"

if [ -n "${REMOTE_IMAGE_TAG}" ]; then
    echo
    echo -e "${GREEN}Remote image mode:${NC}"
    echo "  If DESD runs on another machine, pull the pushed image there:"
    echo "  docker pull ${REMOTE_IMAGE_TAG}"
fi

echo
echo -e "${GREEN}Done:${NC} AI image built and integration instructions printed."
