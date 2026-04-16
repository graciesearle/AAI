"""Compatibility wrapper for legacy Task 2 training script path.

The maintained training pipeline now lives in task2_quality.training_pipeline.
"""

from task2_quality.training_pipeline import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
