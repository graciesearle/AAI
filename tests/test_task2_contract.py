from io import BytesIO
from pathlib import Path
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from PIL import Image


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

    def test_predict_returns_non_stub_response(self):
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
        self.assertIn("overall_grade", response.data)
        self.assertIn("class_probabilities", response.data)
        self.assertIsInstance(response.data["model_version_used"], str)
        self.assertTrue(response.data["model_version_used"].strip())
        self.assertNotEqual(
            str(response.data.get("explanation_payload", {}).get("note", "")).lower(),
            "stub-response",
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
