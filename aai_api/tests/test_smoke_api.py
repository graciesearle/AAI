from django.test import SimpleTestCase


class SmokeApiTests(SimpleTestCase):
    def test_health_endpoint(self):
        response = self.client.get("/api/health/")
        self.assertEqual(response.status_code, 200)

    def test_task2_predict_requires_post(self):
        response = self.client.get("/api/task2/predict/")
        self.assertEqual(response.status_code, 405)

    def test_task1_recommend_requires_post(self):
        response = self.client.get("/api/task1/recommend/")
        self.assertEqual(response.status_code, 405)

    def test_task4_explain_requires_post(self):
        response = self.client.get("/api/task4/explain/")
        self.assertEqual(response.status_code, 405)
