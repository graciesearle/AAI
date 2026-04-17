#!/usr/bin/env bash
set -euo pipefail

# Reusable Task 2 checkpoint smoke test runner.
# Usage:
#   ./task2_raw_files/test_task2_checkpoint.sh /abs/path/to/model.pth [--image /abs/path/to/image.jpg]
# 
#  e.g., bash AAI/task2_raw_files/test_task2_checkpoint.sh "C:/Users/jacob/Documents/Projects/UNI/AAI/task2_raw_files/fresh_rotten_resnet501.pth"
#
# Optional env overrides:
#   PYTHON_BIN=/abs/path/to/python
#   TRAIN_SCRIPT=/abs/path/to/train_fresh_rotten_classifier.py
#   DATASET_ROOT=/abs/path/to/dataset/root
#   FULL_EVAL_ON_LOAD=0|1
#   FORCE_TRAIN=0|1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_SCRIPT_DEFAULT="${SCRIPT_DIR}/train_fresh_rotten_classifier.py"
DATASET_ROOT_DEFAULT="${REPO_ROOT}/task2_data/Fruit And Vegetable Diseases Dataset"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-$TRAIN_SCRIPT_DEFAULT}"
DATASET_ROOT="${DATASET_ROOT:-$DATASET_ROOT_DEFAULT}"
FULL_EVAL_ON_LOAD="${FULL_EVAL_ON_LOAD:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

detect_python_bin() {
  local candidates=()

  # 1) Explicit override (highest priority).
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi

  # 2) Common local venv locations across Windows + Unix-like platforms.
  candidates+=(
    "${REPO_ROOT}/../.venv/Scripts/python.exe"
    "${REPO_ROOT}/../.venv/bin/python"
    "${REPO_ROOT}/.venv/Scripts/python.exe"
    "${REPO_ROOT}/.venv/bin/python"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  # 3) Fall back to PATH lookup.
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  return 1
}

if ! PYTHON_BIN_RESOLVED="$(detect_python_bin)"; then
  echo "Error: unable to find a Python interpreter."
  echo "Set PYTHON_BIN=/abs/path/to/python and rerun."
  exit 1
fi

PYTHON_BIN="$PYTHON_BIN_RESOLVED"

if [[ $# -lt 1 ]]; then
  echo "Error: checkpoint path is required."
  echo "Usage: $0 /abs/path/to/model.pth [--image /abs/path/to/image.jpg]"
  exit 1
fi

CHECKPOINT_PATH="$1"
shift

PREDICT_IMAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      if [[ $# -lt 2 ]]; then
        echo "Error: --image requires a file path."
        exit 1
      fi
      PREDICT_IMAGE="$2"
      shift 2
      ;;
    *)
      echo "Error: unknown argument: $1"
      echo "Usage: $0 /abs/path/to/model.pth [--image /abs/path/to/image.jpg]"
      exit 1
      ;;
  esac
done

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Error: train script not found: $TRAIN_SCRIPT"
  exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Error: checkpoint not found: $CHECKPOINT_PATH"
  exit 1
fi

if [[ -z "$PREDICT_IMAGE" ]]; then
  # Pick first dataset image automatically for inference smoke test.
  PREDICT_IMAGE="$(find "$DATASET_ROOT" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' -o -iname '*.tif' -o -iname '*.tiff' \) | head -n 1 || true)"
fi

if [[ -z "$PREDICT_IMAGE" || ! -f "$PREDICT_IMAGE" ]]; then
  echo "Error: no valid prediction image found."
  echo "Provide one via: --image /abs/path/to/image.jpg"
  exit 1
fi

echo "=== Task 2 Checkpoint Smoke Test ==="
echo "Python       : $PYTHON_BIN"
echo "Train script : $TRAIN_SCRIPT"
echo "Dataset root : $DATASET_ROOT"
echo "Checkpoint   : $CHECKPOINT_PATH"
echo "Predict image: $PREDICT_IMAGE"
echo "Force train  : $FORCE_TRAIN"
echo "Full eval    : $FULL_EVAL_ON_LOAD"

echo
echo "[1/2] Load checkpoint without retraining"
TASK2_SAVE_MODEL_PATH="$CHECKPOINT_PATH" \
TASK2_FORCE_TRAIN="$FORCE_TRAIN" \
TASK2_FULL_EVAL_ON_LOAD="$FULL_EVAL_ON_LOAD" \
"$PYTHON_BIN" "$TRAIN_SCRIPT"

echo
echo "[2/2] Inference smoke test with checkpoint"
TASK2_SAVE_MODEL_PATH="$CHECKPOINT_PATH" \
TASK2_FORCE_TRAIN="$FORCE_TRAIN" \
TASK2_FULL_EVAL_ON_LOAD="$FULL_EVAL_ON_LOAD" \
TASK2_PREDICT_IMAGE="$PREDICT_IMAGE" \
"$PYTHON_BIN" "$TRAIN_SCRIPT"

echo
echo "Done. If you saw 'Loaded existing model from' and a prediction block, checkpoint loading works."
