# AAI REPO
This repo is organised as one API service plus task modules for the case study.

## File Structure

- `aai_api/`: runnable Django/DRF service (API adapters, core app logic, tests, Docker setup) - Task 3 and For DESD connection.
- `task1/`: Task 1-Next-Basket: Next-Basket Classification. Task1-Recommendation: Frequent Itemset Mining.
- `task2_3_4/`: Contains model logic for Task 2 (Quality) and Task 4 (XAI). Task 3 (Lifecycle) is documented here but implemented within the service layer.
- `models/`: stored model artifacts and lifecycle registry data.

## How AAI Considers The Case Study

The system treats the case study as four connected capabilities:

1. Task 1-Next-Basket: Next-Basket Classification.
2. Task 1-Recommendation: Frequent Itemset Mining
3. Task 2: Produce quality prediction.
4. Task 3: Model lifecycle management and interaction logging (See [task3_lifecycle/README.md](./task2_3_4/task3_lifecycle/README.md)).
5. Task 4: Explainability outputs (XAI).

`aai_api/` is the runtime entry point; it calls the task modules via adapters/services for use with DESD.

The testing framework used is `pytest`, source code adheres to PEP8 and includes docstrings.


## Getting Started

To set up the environment and install all dependencies (including those for Task 3 lifecycle and Task 1/2/4 logic):

```bash
# From the AAI root directory
pip install -r aai_api/requirements.txt
```

To run the API service locally:

```bash
cd aai_api
python manage.py runserver 8001
```

