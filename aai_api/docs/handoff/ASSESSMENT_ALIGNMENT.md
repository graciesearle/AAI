# Assessment Alignment And Handoff Map

## Source Documents

- `task2_3_4/reference/AAI_DOCS/faqs.md`
- `task2_3_4/reference/AAI_DOCS/ai case study.md`
- `task2_3_4/reference/AAI_DOCS/spec.md`

## Structural Alignment

- Task 1 separated into `task1/`.
- Tasks 2, 3, and 4 grouped in `task2_3_4/`.
- API/container/integration concerns isolated in `aai_api/`.

## Delivery Responsibilities

- API and integration contract: `aai_api/`
- Task 1 recommendation: `task1/task1_recommendation/`
- Task 2 quality and training: `task2_3_4/task2_quality/`, `task2_3_4/task2_raw_files/`
- Task 3 lifecycle: `task2_3_4/task3_lifecycle/`
- Task 4 XAI: `task2_3_4/task4_xai/`

## Shared Code To Prevent Drift

- Grading and inventory thresholds centralized in `task2_3_4/shared/quality_rules.py`.
- Consumers:
  - `task2_3_4/task2_quality/postprocess.py`
  - `task2_3_4/task2_raw_files/train_fresh_rotten_classifier.py`

## Handoff Checklist

- Each task owner updates task-specific README and test notes.
- Contract schema changes are versioned and communicated before merge.
- Task 2/4 changes touching quality grade logic must update only shared rules.
- Demo evidence paths are documented in `docs/` and linked from report.
