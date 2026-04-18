from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Dict, List, Tuple

import torch
from PIL import Image
from torchvision import models, transforms


class ModelInferenceError(RuntimeError):
    """Raised when loading artifacts or running inference fails."""


def _build_model(num_classes: int = 2) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


def _read_uploaded_image(image_file: BinaryIO, image_size: int = 224) -> torch.Tensor:
    try:
        image_file.seek(0)
    except Exception:
        pass

    image = Image.open(image_file).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess(image).unsqueeze(0)


def _load_checkpoint(artifact_path: Path) -> Dict:
    try:
        return torch.load(str(artifact_path), map_location="cpu")
    except Exception as exc:
        raise ModelInferenceError(f"Unable to read checkpoint at '{artifact_path}': {exc}") from exc


def infer_binary_from_artifact(
    *,
    image_file: BinaryIO,
    artifact_path: Path,
) -> Tuple[str, float, Dict[str, float], List[str]]:
    if not artifact_path.exists():
        raise ModelInferenceError(f"Artifact file not found: {artifact_path}")

    checkpoint = _load_checkpoint(artifact_path)
    if not isinstance(checkpoint, dict):
        raise ModelInferenceError("Checkpoint format is invalid (expected dictionary).")

    class_names = checkpoint.get("class_names") or ["fresh", "rotten"]
    if not isinstance(class_names, list) or len(class_names) < 2:
        class_names = ["fresh", "rotten"]

    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ModelInferenceError("Checkpoint missing model_state_dict/state_dict.")

    model = _build_model(num_classes=len(class_names))
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as exc:
        raise ModelInferenceError(f"Failed to load model weights: {exc}") from exc

    model.eval()

    try:
        x = _read_uploaded_image(image_file, image_size=int(checkpoint.get("image_size", 224)))
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
    except Exception as exc:
        raise ModelInferenceError(f"Inference execution failed: {exc}") from exc

    idx = int(torch.argmax(probs).item())
    confidence = float(probs[idx].item())
    class_probabilities = {
        str(name): round(float(probs[i].item()) * 100.0, 4)
        for i, name in enumerate(class_names)
    }
    predicted_class = str(class_names[idx])

    return predicted_class, confidence, class_probabilities, class_names
