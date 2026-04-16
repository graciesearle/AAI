"""Compatibility wrapper for legacy Task 2 post-processing path.

The maintained implementation now lives in task2_quality.postprocess.
"""

from task2_quality.postprocess import (  # noqa: F401
    QualityScores,
    assign_overall_grade,
    clamp,
    generate_quality_attributes,
    main,
    normalize_label,
    process_prediction,
    update_inventory_and_discount,
)


if __name__ == "__main__":
    main()
