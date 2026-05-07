# Task 3: Model Lifecycle & Interaction Management

Task 3 focuses on the operational management of the AI models and the creation of a "feedback loop" through interaction logging.

Unlike Task 2 and 4, which focus on specific model logic, Task 3 is implemented as a **service-level capability** within the main `aai_api` package to ensure it can manage all models in the system.

## Key Implementation Files

The core logic and API for this task have been integrated directly into the service layer:

- **Lifecycle Logic:** [`aai_api/ai_core/lifecycle.py`](../../aai_api/ai_core/lifecycle.py)  
  Handles the registration, activation, and rollback of model versions using a centralised registry (`models/_lifecycle_registry.json`).
- **API Adapters:** [`aai_api/api_adapters/task3.py`](../../aai_api/api_adapters/task3.py)  
  Exposes the lifecycle management and interaction logging endpoints to authorised AI Engineers.
- **Data Schemas:** [`aai_api/api_adapters/task3_serializers.py`](../../aai_api/api_adapters/task3_serializers.py)  
  Defines the validation logic for model uploads and interaction overrides.
- **Interaction Logging (Database):** [`aai_api/ai_core/models.py`](../../aai_api/ai_core/models.py)  
  Defines the `InferenceLog` model used to capture every prediction, confidence score, and producer feedback.

## Capabilities Demonstrated

1. **Model Versioning & Rollback:** AI Engineers can upload new model bundles, activate them without downtime, or rollback to a previous known-good version if performance drops.
2. **Interaction Database:** Every inference performed by the system is logged with full metadata (timestamp, producer, inputs, outputs, confidence).
3. **The Feedback Loop:** When a producer overrides an AI-assigned grade, the system captures the "correct" grade via the `/override/` endpoint. This data provides the primary dataset for future model retraining.

## Data for Retraining & Testing

Task 3 serves as the data collection hub for all AI capabilities in the repository:

### 1. Task 1 (Next-Basket & Recommendations)
- **Training Data:** The initial models are trained on [`task1/task1_recommendation/Groceries_dataset.csv`](../../task1/task1_recommendation/Groceries_dataset.csv), which contains real-world transaction patterns.
- **Retraining:** Future iterations of the recommendation engine can be retrained by exporting transaction logs from the DESD marketplace and formatting them to match the CSV structure.

### 2. Task 2 (Quality Prediction)
- **Interaction Database:** Every request to `/api/task2/predict/` generates an `InferenceLog`.
- **Exporting for Retraining:** AI Engineers can query `/api/task3/interactions/` and filter for `producer_accepted=False`. This isolates the cases where the model failed to match human expertise, creating a high-value "corrective dataset" for fine-tuning.
- **Testing:** The system allows testing against historical training data by passing specific images from the training set to the prediction endpoint and comparing the output against the `InferenceLog` history.


## Integration with DESD (Marketplace)

While the marketplace logic lives in a separate repository, Task 3 provides the critical interface for that integration:
- **Versioning:** DESD uses the activation API to switch between model versions globally.
- **Feedback:** When a producer uses the DESD dashboard to correct an AI grade, the marketplace sends a `PATCH` request to the `/override/` endpoint, completing the retraining loop.


## API Endpoint Map

| Task | Endpoint | Description |
| :--- | :--- | :--- |
| **Task 1** | `/api/task1/recommend/` | Frequent Itemset Mining recommendations. |
| **Task 1** | `/api/task1/next-basket/` | Next-Basket classification for customers. |
| **Task 2** | `/api/task2/predict/` | AI-driven quality grading for products. |
| **Task 3** | `/api/task3/models/` | Model versioning and lifecycle management. |
| **Task 3** | `/api/task3/interactions/` | Interaction logging and producer feedback loop. |
| **Task 4** | `/api/task4/explain/` | Explainability (XAI) outputs for quality predictions. |

## Verification

To verify the functionality of Task 3, run the dedicated test suite:

```bash
# Run from the AAI root directory
pytest aai_api/tests/test_task3_lifecycle_api.py
```
