from io import BytesIO
from pathlib import Path
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from PIL import Image


class Task3LifecycleApiTests(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.override = override_settings(MODEL_ROOT=Path(self.temp_dir.name))
        self.override.enable()

    def tearDown(self):
        self.override.disable()
        self.temp_dir.cleanup()

    def _upload_model(self, version: str, artifact_name: str = "model.pth"):
        artifact = SimpleUploadedFile(
            name=artifact_name,
            content=f"weights-{version}".encode("utf-8"),
            content_type="application/octet-stream",
        )
        response = self.client.post(
            "/api/task3/models/upload/",
            {
                "model_name": "produce-quality",
                "model_version": version,
                "framework": "pytorch",
                "artifact": artifact,
            },
            format="multipart",
        )
        return response

    def _make_image(self):
        image_data = BytesIO()
        image = Image.new("RGB", (24, 24), color=(20, 150, 60))
        image.save(image_data, format="PNG")
        image_data.seek(0)
        return SimpleUploadedFile(
            name="sample.png",
            content=image_data.read(),
            content_type="image/png",
        )

    def test_upload_list_and_activate_model(self):
        upload = self._upload_model("2.0.0")
        self.assertEqual(upload.status_code, 201)
        self.assertEqual(upload.data["model_version"], "2.0.0")

        listing = self.client.get("/api/task3/models/")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.data["count"], 1)

        activate = self.client.post(
            "/api/task3/models/activate/",
            {
                "model_name": "produce-quality",
                "model_version": "2.0.0",
            },
        )
        self.assertEqual(activate.status_code, 200)
        self.assertEqual(activate.data["model_version"], "2.0.0")

    def test_rollback_switches_to_previous_version(self):
        self.assertEqual(self._upload_model("2.0.0").status_code, 201)
        self.assertEqual(self._upload_model("2.1.0").status_code, 201)

        activate_v1 = self.client.post(
            "/api/task3/models/activate/",
            {"model_name": "produce-quality", "model_version": "2.0.0"},
        )
        self.assertEqual(activate_v1.status_code, 200)

        activate_v2 = self.client.post(
            "/api/task3/models/activate/",
            {"model_name": "produce-quality", "model_version": "2.1.0"},
        )
        self.assertEqual(activate_v2.status_code, 200)

        rollback = self.client.post(
            "/api/task3/models/rollback/",
            {"model_name": "produce-quality"},
        )
        self.assertEqual(rollback.status_code, 200)
        self.assertEqual(rollback.data["model_version"], "2.0.0")

    def test_task2_uses_active_version_when_model_not_provided(self):
        self.assertEqual(self._upload_model("3.0.0").status_code, 201)

        activate = self.client.post(
            "/api/task3/models/activate/",
            {"model_name": "produce-quality", "model_version": "3.0.0"},
        )
        self.assertEqual(activate.status_code, 200)

        predict = self.client.post(
            "/api/task2/predict/",
            {
                "producer_id": 77,
                "image": self._make_image(),
            },
            format="multipart",
        )

        self.assertEqual(predict.status_code, 200)
        self.assertEqual(predict.data["model_version_used"], "3.0.0")
