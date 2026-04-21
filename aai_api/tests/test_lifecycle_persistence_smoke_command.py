from pathlib import Path
import tempfile

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase, override_settings

from aai_api.ai_core.lifecycle import list_model_versions


class LifecyclePersistenceSmokeCommandTests(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.override = override_settings(MODEL_ROOT=Path(self.temp_dir.name))
        self.override.enable()

    def tearDown(self):
        self.override.disable()
        self.temp_dir.cleanup()

    def test_prepare_and_verify_bundle(self):
        call_command(
            "lifecycle_persistence_smoke",
            model_name="produce-quality",
            model_version="9.9.9",
        )

        bundle_root = Path(self.temp_dir.name) / "produce-quality" / "9.9.9"
        self.assertTrue((bundle_root / "manifest.json").exists())
        self.assertTrue((bundle_root / "artifacts" / "smoke.bin").exists())

        versions = list_model_versions(Path(self.temp_dir.name))
        self.assertTrue(
            any(
                item.get("model_name") == "produce-quality" and item.get("model_version") == "9.9.9"
                for item in versions
            )
        )

        call_command(
            "lifecycle_persistence_smoke",
            model_name="produce-quality",
            model_version="9.9.9",
            verify_only=True,
        )

    def test_verify_only_fails_for_missing_bundle(self):
        with self.assertRaises(CommandError):
            call_command(
                "lifecycle_persistence_smoke",
                model_name="produce-quality",
                model_version="missing-version",
                verify_only=True,
            )
