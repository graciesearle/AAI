# AAI Task 2 - Deep Learning Produce Quality Pipeline

This repository implements Task 2 with a computer vision, transfer-learning CNN
approach and does not use simple/classical machine learning algorithms.

## Compliance Summary

1. **Computer vision-based application**
   - Uses image datasets and CNN inference/training only.

2. **Advanced algorithm for visual classification**
   - Uses **ResNet-50** as the core architecture.

3. **Transfer learning required**
   - Enabled by default (`no_pretrained=False`) with ImageNet weights.

4. **Fresh vs Rotten (defect) detection**
   - Classification head predicts Healthy/Fresh vs Rotten/Defective classes.

5. **Detailed quality breakdown (Colour, Size, Ripeness)**
   - A deep regression head predicts three percentage scores directly.

6. **Overall grade A/B/C + inventory action**
   - Post-processing maps quality scores to grade and discount action.

7. **No simple ML algorithms in project**
   - No Random Forest, SVM, KNN, logistic regression, decision trees,
     or sklearn-based training components are used.

## Architecture

The model is a multitask neural network:

- **Backbone**: ResNet-50 (transfer learning)
- **Head 1**: Classification head (Healthy/Fresh vs Rotten)
- **Head 2**: Quality regression head (Colour, Size, Ripeness in [0, 100])

File: `train_fresh_rotten_classifier.py`

### Why this is still valid with binary labels

The dataset has binary class labels only. To train the quality head without
manual quality annotations, proxy quality targets are extracted from image
content (HSV-based CV features) during training and used as regression
supervision. This keeps the system fully computer-vision/deep-learning based.

## Repository Files

- `train_fresh_rotten_classifier.py`
   - Trains and evaluates multitask transfer-learning CNN.
   - Includes post-processing logic for quality grading and inventory actions.
   - Saves model checkpoint and learning curves.

- `fresh_rotten_resnet50.pth`
  - Saved model checkpoint path.

- `requirements.txt`
  - Project dependencies.

## Installation

```bash
pip install -r requirements.txt
```

## Training

Edit `CONFIG` in `train_fresh_rotten_classifier.py`.

For file-based Kaggle loading (no terminal exports), set these fields:

- `auto_download_from_kaggle=True`
- `kaggle_dataset="owner/dataset-slug"`
- `kaggle_username="YOUR_KAGGLE_USERNAME"`
- `kaggle_key="YOUR_KAGGLE_KEY"`

The script will automatically download and extract data into
`dataset_dir` when the folder is missing.

Then run:

```bash
python train_fresh_rotten_classifier.py
```

## Inference + Grading

The training script can optionally run single-image inference with
post-processing when `CONFIG.predict_image` is set.

The post-processing API accepts deep-model quality scores:

```python
from train_fresh_rotten_classifier import process_prediction

result = process_prediction(
    label="Apple__Healthy",
    confidence=0.91,
    quality_scores={"colour": 88.4, "size": 82.3, "ripeness": 85.2},
)
```

Returned fields include:

- `defect_detected`
- `quality_scores`
- `overall_grade`
- `inventory_action`

## Notes

- Transfer learning is intentionally enabled by default for Task 2.
- No simple ML algorithm is required or used in this repository.
