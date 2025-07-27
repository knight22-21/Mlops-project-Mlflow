# MLOps Project - MLflow with DagsHub Integration

This project is focused on practicing and implementing **MLflow** for experiment tracking and **DagsHub** for version control of machine learning pipelines. The goal is to build, track, and optimize machine learning models with **XGBoost** and **Random Forest** for rainfall prediction, while leveraging MLflow’s features like autologging and artifact management. The repository also demonstrates a basic pipeline, hyperparameter tuning, and DagsHub integration.

## Project Overview

The project utilizes **MLflow** for tracking experiments, logging metrics, and saving models. Additionally, **DagsHub** is used to manage versioning and track the experiment data in a GitHub-like environment designed for data science projects.

## Key Objectives

1. **Simple MLflow Pipeline**: Set up a basic pipeline to use MLflow for tracking experiments and logging metrics.
2. **Autologging Pipeline**: Implement MLflow's autologging feature to automatically track models and metrics.
3. **DagsHub Integration**: Set up DagsHub to track experiments and artifacts, including model versions and confusion matrices.
4. **Hyperparameter Tuning**: Implement hyperparameter tuning for XGBoost using GridSearchCV and log results using MLflow.

## Repository Structure

```bash
.
├── data/                       # Data files (train, test, sample_submission.csv)
├── mlartifacts/                # MLflow model artifacts
├── mlruns/                     # MLflow run logs and results
├── src/                        # Source code (including ML pipelines)
|    ├── pipeline-dag.py        # Pipeline using DagsHub integration
|    ├── pipeline-hyp.py        # Pipeline for hyperparameter tuning with GridSearchCV
|    ├── pipeline-xg.py         # XGBoost pipeline for training and predictions
|    └── pipeline.py            # Base MLflow pipeline for training
├── .gitignore                  # Git ignore configuration
├── LICENSE                     # Project License
├── README.md                   # Project documentation
└── confusion_matrix.png        # Confusion matrix for validation results
```

## Setup Instructions

### Prerequisites

1. Python 3.7+

### MLflow Setup

1. **Start MLflow Server**: If you're using a local MLflow tracking server, you can start it using the following command:

```bash
mlflow ui
```

By default, the MLflow UI will be available at `http://localhost:5000`.

2. **Set Tracking URI**: Make sure that the **tracking URI** is set correctly in your code. It can point to a local or remote MLflow instance.

```python
mlflow.set_tracking_uri("http://localhost:5000")  # or your remote MLflow URI
```

### DagsHub Integration

1. **DagsHub Account**: To track your experiments and data on DagsHub, create an account at [DagsHub](https://dagshub.com) if you haven't already.
2. **Initialize DagsHub**: Use the `dagshub` library to integrate DagsHub with your project:

```python
import dagshub
dagshub.init(repo_owner='user-name', repo_name='Project-Name', mlflow=True)
```

This will connect your MLflow experiments to your DagsHub repository.

## How to Run

### 1. **Training Pipeline (XGBoost)**

This pipeline trains an XGBoost model, tracks the metrics, logs the confusion matrix, and saves the model to the MLflow server.

To run the basic pipeline:

```bash
python pipeline-xg.py
```

### 2. **Hyperparameter Tuning Pipeline (GridSearchCV)**

This pipeline performs hyperparameter tuning using `GridSearchCV` for the XGBoost model and logs the results to MLflow.

To run the pipeline for hyperparameter tuning:

```bash
python pipeline-hyp.py
```

### 3. **DagsHub Pipeline**

This pipeline demonstrates the integration of **DagsHub** with the MLflow pipeline, ensuring version control and tracking of the experiments and artifacts.

To run the DagsHub pipeline:

```bash
python pipeline-dag.py
```

### 4. **Standard Pipeline**

This is a basic ML pipeline that trains the model without hyperparameter tuning or DagsHub integration.

To run the standard pipeline:

```bash
python pipeline.py
```

## Logs and Artifacts

1. **MLflow**: The experiment logs and model artifacts are stored under the `mlruns/` and `mlartifacts/` directories respectively.

2. **Confusion Matrix**: The confusion matrix for the validation set is saved as `confusion_matrix.png` in the project root.

## Model Evaluation

The performance of the models is evaluated using accuracy and confusion matrices. The confusion matrix is logged as an artifact for further analysis.

## Model Submission

Once the models are trained, predictions are made on the test dataset, and a submission file (`submission.csv`) is generated. This submission can be used for evaluation in external systems.


## Future Work

1. **Model Improvement**: Implement feature engineering or try different models (e.g., Gradient Boosting, LightGBM).
2. **Pipeline Automation**: Automate training pipelines using orchestration tools like Airflow or Kubeflow.
3. **Model Deployment**: Deploy the trained models as REST APIs using Flask or FastAPI, and track them using MLflow’s model registry.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
