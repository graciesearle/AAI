#!/usr/bin/env python
import sys
from pathlib import Path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from aai_api.manage import main

    main()
