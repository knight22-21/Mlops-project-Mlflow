import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Rainfall Prediction - xgboost")

# Enable autologging
mlflow.autolog()

# Load train and test datasets
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# Feature and target
target = "rainfall"
RMV = ['rainfall', 'id']
FEATURES = [col for col in train.columns if col not in RMV]

# Split the dataset into train/test sets
X = train[FEATURES]
y = train[target]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(test[FEATURES])

# Start MLflow run
with mlflow.start_run():
    
    # Define and train XGBoost model
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    xgb_model.fit(X_train_scaled, y_train)
    
    # Predict on validation set
    y_pred = xgb_model.predict(X_valid_scaled)
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    mlflow.log_metric("accuracy", accuracy)
    
    # Confusion matrix
    cm = confusion_matrix(y_valid, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Log confusion matrix as artifact
    mlflow.log_artifact("confusion_matrix.png")

    # Optional metadata
    mlflow.set_tags({"Author": "Krishna", "Project": "Rainfall Prediction"})

# Make predictions on test set
test_predictions = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Prepare and save submission
sub = pd.read_csv("./data/sample_submission.csv")
sub["rainfall"] = test_predictions
sub.to_csv("./data/submission2.csv", index=False)
