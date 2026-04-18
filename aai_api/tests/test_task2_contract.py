import hashlib
import json
from io import BytesIO
from pathlib import Path
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from PIL import Image
import torch

from task2_3_4.task2_quality.model_inference import _build_model


class Task2QualityContractTests(TestCase):
    def _make_image(self):
        image_data = BytesIO()
        image = Image.new("RGB", (24, 24), color=(210, 90, 40))
        image.save(image_data, format="PNG")
        image_data.seek(0)
        return SimpleUploadedFile(
            name="sample.png",
            content=image_data.read(),
            content_type="image/png",
        )

    def _write_model_bundle(self, *, model_root: Path, model_name: str, model_version: str, valid: bool):
        bundle_root = model_root / model_name / model_version
        artifacts_root = bundle_root / "artifacts"
        artifacts_root.mkdir(parents=True, exist_ok=True)

        artifact_path = artifacts_root / "model.pth"
        if valid:
            model = _build_model(num_classes=2)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": ["fresh", "rotten"],
                "image_size": 224,
                "model_name": model_name,
                "model_version": model_version,
            }
            torch.save(checkpoint, artifact_path)
            artifact_bytes = artifact_path.read_bytes()
        else:
            artifact_bytes = b"this-is-not-a-valid-pytorch-checkpoint"
            artifact_path.write_bytes(artifact_bytes)

        artifact_checksum = hashlib.sha256(artifact_bytes).hexdigest()
        manifest = {
            "model_name": model_name,
            "model_version": model_version,
            "task_profile": "task2_quality",
            "schema_version": "task2-quality-v1",
            "framework": "pytorch",
            "entrypoint": "task_areas.task2_3_4.task2_quality.runtime:run_quality_inference",
            "artifacts": [
                {
                    "type": "model_weights",
                    "path": "artifacts/model.pth",
                    "checksum": artifact_checksum,
                }
            ],
            "input_schema": {
                "image": "multipart-file",
                "producer_id": "int",
                "product_id": "int?",
                "model_version": "str?",
            },
            "output_schema": {
                "color_score": "float",
                "size_score": "float",
                "ripeness_score": "float",
                "confidence": "float",
                "predicted_class": "str",
                "overall_grade": "str",
                "class_probabilities": "object",
                "explanation_payload": "object",
                "transparency_refs": "array",
                "model_version_used": "str",
            },
            "metrics": {},
            "created_at": "2026-01-01T00:00:00Z",
        }
        (bundle_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_predict_returns_non_stub_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_root = Path(temp_dir)
            model_name = "produce-quality"
            model_version = "1.0.0"
            self._write_model_bundle(
                model_root=model_root,
                model_name=model_name,
                model_version=model_version,
                valid=True,
            )

            with self.settings(
                MODEL_ROOT=model_root,
                DEFAULT_MODEL_NAME=model_name,
                DEFAULT_MODEL_VERSION=model_version,
            ):
                response = self.client.post(
                    "/api/task2/predict/",
                    {
                        "producer_id": 1,
                        "image": self._make_image(),
                    },
                    format="multipart",
                )

        self.assertEqual(response.status_code, 200)
        self.assertIn(response.data["predicted_class"], {"fresh", "rotten"})
        self.assertEqual(response.data.get("overall_grade"), "UNSET")
        self.assertIn("class_probabilities", response.data)
        self.assertIsInstance(response.data["model_version_used"], str)
        self.assertTrue(response.data["model_version_used"].strip())
        self.assertEqual(
            str(response.data.get("explanation_payload", {}).get("note", "")).lower(),
            "task2-blank-slate",
        )

    @override_settings(MODEL_ROOT=Path(tempfile.gettempdir()) / "missing-aai-model-root")
    def test_predict_returns_503_when_manifest_missing(self):
        response = self.client.post(
            "/api/task2/predict/",
            {
                "producer_id": 1,
                "image": self._make_image(),
            },
            format="multipart",
        )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Manifest not found", response.data.get("detail", ""))

    def test_predict_returns_503_when_model_inference_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_root = Path(temp_dir)
            model_name = "produce-quality"
            model_version = "1.0.0"
            self._write_model_bundle(
                model_root=model_root,
                model_name=model_name,
                model_version=model_version,
                valid=False,
            )

            with self.settings(
                MODEL_ROOT=model_root,
                DEFAULT_MODEL_NAME=model_name,
                DEFAULT_MODEL_VERSION=model_version,
            ):
                response = self.client.post(
                    "/api/task2/predict/",
                    {
                        "producer_id": 1,
                        "image": self._make_image(),
                    },
                    format="multipart",
                )

            self.assertEqual(response.status_code, 503)
            self.assertIn("Model inference failed", response.data.get("detail", ""))
