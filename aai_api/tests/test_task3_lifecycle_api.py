from io import BytesIO
from pathlib import Path
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from PIL import Image
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient
import torch

from task2_3_4.task2_quality.task2_model import build_model


User = get_user_model()


class Task3LifecycleApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username="ai_engineer", password="SecurePass123")
        self.token = Token.objects.create(user=self.user)
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {self.token.key}")

        self.temp_dir = tempfile.TemporaryDirectory()
        self.override = override_settings(MODEL_ROOT=Path(self.temp_dir.name))
        self.override.enable()

    def tearDown(self):
        self.override.disable()
        self.temp_dir.cleanup()

    def _upload_model(self, version: str, artifact_name: str = "model.pth", valid_checkpoint: bool = False):
        if valid_checkpoint:
            model = build_model(num_classes=2, device=torch.device("cpu"), use_pretrained=False)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": ["fresh", "rotten"],
                "image_size": 224,
                "model_name": "produce-quality",
                "model_version": version,
            }
            artifact_bytes = tempfile.NamedTemporaryFile(delete=False)
            artifact_bytes.close()
            torch.save(checkpoint, artifact_bytes.name)
            content = Path(artifact_bytes.name).read_bytes()
            Path(artifact_bytes.name).unlink(missing_ok=True)
        else:
            content = f"weights-{version}".encode("utf-8")

        artifact = SimpleUploadedFile(
            name=artifact_name,
            content=content,
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

    def test_lifecycle_mutation_endpoints_require_authentication(self):
        anon = APIClient()

        upload = anon.post(
            "/api/task3/models/upload/",
            {
                "model_name": "produce-quality",
                "model_version": "4.0.0",
                "framework": "pytorch",
                "artifact": SimpleUploadedFile(
                    name="model.pth",
                    content=b"weights",
                    content_type="application/octet-stream",
                ),
            },
            format="multipart",
        )
        self.assertEqual(upload.status_code, 401)

        activate = anon.post(
            "/api/task3/models/activate/",
            {
                "model_name": "produce-quality",
                "model_version": "4.0.0",
            },
            format="json",
        )
        self.assertEqual(activate.status_code, 401)

        rollback = anon.post(
            "/api/task3/models/rollback/",
            {
                "model_name": "produce-quality",
            },
            format="json",
        )
        self.assertEqual(rollback.status_code, 401)

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
        self.assertEqual(self._upload_model("3.0.0", valid_checkpoint=True).status_code, 201)

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
