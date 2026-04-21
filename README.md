# AAI Task 2 - Deep Learning Produce Quality Pipeline

This repo is organised as one API service plus task modules for the case study.

## File Structure

- `aai_api/`: runnable Django/DRF service (API adapters, core app logic, tests, Docker setup) - For DESD connection.
- `task1/`: Task 1 recommendation module.
- `task2_3_4/`: Task 2 quality model, Task 3 lifecycle, Task 4 XAI modules.
- `models/`: stored model artifacts and lifecycle registry data.

## How AAI Considers The Case Study

The system treats the case study as four connected capabilities:

1. Task 1: recommendation.
2. Task 2: produce quality prediction.
3. Task 3: model lifecycle management (register/activate/sync).
4. Task 4: explainability outputs (XAI).

`aai_api/` is the runtime entry point; it calls the task modules via adapters/services for use with DESD.
