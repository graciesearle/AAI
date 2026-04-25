# AAI Task 2 - Deep Learning Produce Quality Pipeline

This repo is organised as one API service plus task modules for the case study.

## File Structure

- `aai_api/`: runnable Django/DRF service (API adapters, core app logic, tests, Docker setup) - For DESD connection.
- `task1/`: Task 1-Next-Basket: Next-Basket Classification. Task1-Recommendation: Frequent Itemset Mining.
- `task2_3_4/`: Task 2: Produce quality model, Task 3: Model lifecycle management (register/activate/sync), Task 4: Explainability outputs (XAI).
- `models/`: stored model artifacts and lifecycle registry data.

## How AAI Considers The Case Study

The system treats the case study as four connected capabilities:

1. Task 1-Next-Basket: Next-Basket Classification.
2. Task 1-Recommendation: Frequent Itemset Mining
3. Task 2: Produce quality prediction.
4. Task 3: Model lifecycle management (register/activate/sync).
5. Task 4: Explainability outputs (XAI).

`aai_api/` is the runtime entry point; it calls the task modules via adapters/services for use with DESD.

The testing framework used is `pytest`, source code adheres to PEP8 and includes docstrings.