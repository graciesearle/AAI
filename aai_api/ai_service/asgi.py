import os
import sys
from pathlib import Path

from django.core.asgi import get_asgi_application

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aai_api.ai_service.settings")
application = get_asgi_application()
