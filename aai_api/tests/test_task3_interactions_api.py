from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient

from aai_api.ai_core.models import InferenceLog

User = get_user_model()


class Task3InteractionApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username="ai_interactions", password="SecurePass123")
        self.token = Token.objects.create(user=self.user)
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {self.token.key}")

    def _create_log(self, *, producer_id: int, model_version: str = "1.0.0"):
        return InferenceLog.objects.create(
            producer_id=producer_id,
            product_id=producer_id,
            model_version=model_version,
            confidence=90.0,
            color_score=80.0,
            size_score=81.0,
            ripeness_score=82.0,
            predicted_grade="A",
        )

    def test_list_and_override_interaction(self):
        older = self._create_log(producer_id=100)
        newer = self._create_log(producer_id=101)

        listing = self.client.get("/api/task3/interactions/")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.data["count"], 2)
        self.assertEqual(listing.data["results"][0]["id"], newer.id)
        self.assertEqual(listing.data["results"][1]["id"], older.id)

        override = self.client.patch(
            f"/api/task3/interactions/{older.id}/override/",
            {
                "producer_accepted": False,
                "override_grade": "B",
            },
            format="json",
        )
        self.assertEqual(override.status_code, 200)
        self.assertEqual(override.data["producer_accepted"], False)
        self.assertEqual(override.data["override_grade"], "B")

    def test_override_rejection_requires_override_grade(self):
        log = self._create_log(producer_id=102)

        response = self.client.patch(
            f"/api/task3/interactions/{log.id}/override/",
            {
                "producer_accepted": False,
            },
            format="json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("override_grade", response.data)

    def test_interaction_endpoints_require_authentication(self):
        log = self._create_log(producer_id=103)
        anon = APIClient()

        listing = anon.get("/api/task3/interactions/")
        self.assertEqual(listing.status_code, 401)

        patch_response = anon.patch(
            f"/api/task3/interactions/{log.id}/override/",
            {
                "producer_accepted": False,
                "override_grade": "C",
            },
            format="json",
        )
        self.assertEqual(patch_response.status_code, 401)
