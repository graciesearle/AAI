from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class ModelInferenceError(RuntimeError):
    pass


_MODEL_CACHE: dict[str, dict[str, Any]] = {}


def _build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def _normalize_binary_label(label: str) -> str:
    normalized = label.strip().lower()
    fresh_tokens = ("fresh", "healthy", "ripe", "good")
    rotten_tokens = ("rotten", "stale", "spoiled", "diseased", "bad")

    if any(token in normalized for token in fresh_tokens):
        return "fresh"
    if any(token in normalized for token in rotten_tokens):
        return "rotten"

    # Sensible default for unknown labels from legacy checkpoints.
    return "fresh"


def _resolve_artifact_path(
    *,
    model_root: Path,
    model_name: str,
    model_version: str,
    manifest: dict[str, Any],
) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ModelInferenceError("Manifest does not contain any artifacts")

    selected = None
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        if artifact.get("type") == "model_weights":
            selected = artifact
            break

    if selected is None:
        selected = artifacts[0]
        if not isinstance(selected, dict):
            raise ModelInferenceError("Manifest artifact entry is invalid")

    relative_path = str(selected.get("path", "")).strip()
    if not relative_path:
        raise ModelInferenceError("Manifest artifact path is empty")

    bundle_root = model_root / model_name / model_version
    artifact_path = (bundle_root / relative_path).resolve()

    try:
        artifact_path.relative_to(bundle_root.resolve())
    except ValueError as exc:
        raise ModelInferenceError("Manifest artifact resolves outside model bundle") from exc

    if not artifact_path.exists() or not artifact_path.is_file():
        raise ModelInferenceError(f"Model artifact not found: {artifact_path}")

    return artifact_path


def _cache_key(path: Path) -> str:
    stat = path.stat()
    return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ModelInferenceError(f"Unable to load model checkpoint: {exc}") from exc

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
        class_names = checkpoint.get("class_names")
        image_size = checkpoint.get("image_size", 224)
    elif isinstance(checkpoint, dict):
        model_state_dict = checkpoint
        class_names = checkpoint.get("class_names")
        image_size = checkpoint.get("image_size", 224)
    else:
        raise ModelInferenceError("Unsupported checkpoint format")

    if not isinstance(model_state_dict, dict):
        raise ModelInferenceError("Checkpoint does not include a valid state_dict")

    if not isinstance(class_names, list) or not class_names:
        class_names = ["fresh", "rotten"]

    try:
        image_size = int(image_size)
    except (TypeError, ValueError):
        image_size = 224

    if image_size <= 0:
        image_size = 224

    return {
        "model_state_dict": model_state_dict,
        "class_names": [str(name) for name in class_names],
        "image_size": image_size,
    }


def _load_runtime_model(path: Path) -> dict[str, Any]:
    key = _cache_key(path)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    checkpoint = _load_checkpoint(path)
    class_names = checkpoint["class_names"]

    model = _build_model(num_classes=len(class_names))
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except Exception as exc:
        raise ModelInferenceError(f"Checkpoint state_dict mismatch: {exc}") from exc

    model.eval()

    runtime_model = {
        "model": model,
        "class_names": class_names,
        "image_size": checkpoint["image_size"],
        "artifact_path": str(path),
    }
    _MODEL_CACHE[key] = runtime_model
    return runtime_model


def _build_preprocess(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def infer_binary_from_artifact(
    *,
    image_file,
    model_root: Path,
    model_name: str,
    model_version: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    artifact_path = _resolve_artifact_path(
        model_root=model_root,
        model_name=model_name,
        model_version=model_version,
        manifest=manifest,
    )
    runtime_model = _load_runtime_model(artifact_path)

    model: nn.Module = runtime_model["model"]
    class_names: list[str] = runtime_model["class_names"]
    image_size: int = runtime_model["image_size"]

    image_file.seek(0)
    image = Image.open(image_file).convert("RGB")
    preprocess = _build_preprocess(image_size)
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    if probabilities.ndim != 1:
        raise ModelInferenceError("Unexpected model output shape")

    if probabilities.shape[0] != len(class_names):
        raise ModelInferenceError("Model output size does not match class names")

    per_class = {
        class_names[index]: float(probabilities[index].item())
        for index in range(len(class_names))
    }

    fresh_prob = 0.0
    rotten_prob = 0.0
    for class_name, prob in per_class.items():
        mapped = _normalize_binary_label(class_name)
        if mapped == "fresh":
            fresh_prob += prob
        else:
            rotten_prob += prob

    if fresh_prob == 0.0 and rotten_prob == 0.0:
        max_index = int(torch.argmax(probabilities).item())
        mapped = _normalize_binary_label(class_names[max_index])
        if mapped == "fresh":
            fresh_prob = 1.0
        else:
            rotten_prob = 1.0

    predicted_class = "fresh" if fresh_prob >= rotten_prob else "rotten"
    confidence_01 = fresh_prob if predicted_class == "fresh" else rotten_prob

    class_probabilities = {
        "fresh": round(float(fresh_prob), 6),
        "rotten": round(float(rotten_prob), 6),
    }

    return {
        "predicted_class": predicted_class,
        "confidence_01": max(0.0, min(1.0, float(confidence_01))),
        "class_probabilities": class_probabilities,
        "raw_class_probabilities": {k: round(v, 6) for k, v in per_class.items()},
        "artifact_path": runtime_model["artifact_path"],
    }
