import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Rainfall Prediction - rf")

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

# Standardization of features (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(test[FEATURES])

# Define Random Forest model
rf_model = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42)

# MLFlow Experiment Start
with mlflow.start_run():
    
    # Train the model
    rf_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_valid_scaled)
    accuracy = accuracy_score(y_valid, y_pred)
    
    # Log metrics, parameters, and model
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", rf_model.max_depth)
    mlflow.log_param("n_estimators", rf_model.n_estimators)
    
    # Confusion matrix
    cm = confusion_matrix(y_valid, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot = True, fmt = 'd' , cmap='Blues', xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    
    # Save confusion matrix plot
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    # Log confusion matrix plot
    mlflow.log_artifact("confusion_matrix.png")
    
    # Log the trained model
    mlflow.sklearn.log_model(rf_model, "model")
    
    # Log tags for metadata
    mlflow.set_tags({"Author": 'Krishna', "Project": "Rainfall Prediction"})
    
    
# Output the model's accuracy
print(f"Model accuracy: {accuracy:.4f}")

# Make the predictions on the test data
test_predictions = rf_model.predict_proba(X_test_scaled)[:, 1]

# Prepare submission DataFrame
sub = pd.read_csv("./data/sample_submission.csv")
sub['rainfall'] = test_predictions

# Save the submission file
sub.to_csv("./data/submission.csv", index=False)
