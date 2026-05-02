# Network Intrusion Detection with UNSW-NB15

This project explores network intrusion detection on the UNSW-NB15 dataset using two different approaches:

1. An initial anomaly-detection pipeline based on Isolation Forest.
2. A supervised classification pipeline based on Random Forest, which became the final model because it produced better results for this dataset.

The repository keeps both approaches so the progression from experimentation to the final solution is clear.

## Why Two Models Were Used

The first attempt used Isolation Forest because it is a common choice for anomaly detection when labeled attack data is limited or when the goal is to detect unusual traffic patterns without strong supervision. In practice, that approach did not give satisfactory results on this project, especially for separating attack categories reliably.

After that, the project was retrained with Random Forest. Since UNSW-NB15 provides labeled attack data, the supervised model was a better fit and gave stronger classification performance for the available classes.

## Dataset

The dataset used in this project is:

- Name: UNSW-NB15 Dataset
- Official link: https://research.unsw.edu.au/projects/unsw-nb15-dataset

The raw CSV files are stored in `data/raw/`, including the training and testing splits used for experimentation.

## Project Structure

```text
data/
  raw/                  # UNSW-NB15 source CSV files
  processed/            # Processed data artifacts
isolation_forest/
  preprocess_data.py
  train.py
  utils.py
  artifacts/
random_forest_classifier/
  src/train.py          # Random Forest training pipeline
  models/model.onnx     # Exported model
  artifacts/
```

## Installation

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run

### Random Forest

The final classifier is implemented in `random_forest_classifier/src/train.py`. It trains a Random Forest model, evaluates it, and saves outputs such as the trained model, performance report, class mapping, and confusion matrix.

### Isolation Forest

The Isolation Forest workflow is kept in the `isolation_forest/` folder for reference and comparison. It was used as the first experiment, but the results were not strong enough for the final system.

## Outputs

Important saved artifacts include:

- `random_forest_classifier/artifacts/model_performance_report.txt`
- `random_forest_classifier/artifacts/class_mapping.json`
- `random_forest_classifier/models/model.onnx`
- `isolation_forest/artifacts/model_metadata.json`
- `isolation_forest/artifacts/optimal_threshold.npy`

## Notes

- The project is based on the UNSW-NB15 intrusion detection dataset.
- The Isolation Forest model was kept only as an early experiment.
- The Random Forest model is the better-performing and preferred approach in this repository.
