#!/usr/bin/env python
import os
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aai_api.ai_service.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError("Couldn't import Django. Is it installed and on PYTHONPATH?") from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
