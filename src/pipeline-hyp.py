import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Set experiment 
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Rainfall Prediction - XGBoost GridSearch")

# Load train and test data
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# Set target and features
target = "rainfall"
drop_cols = ["id", target]
FEATURES = [col for col in train.columns if col not in drop_cols]

X = train[FEATURES]
y = train[target]

# Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Define model and hyperparameters
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 6],
    'learning_rate': [0.05, 0.1]
}

# Grid Search
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Start MLflow run
with mlflow.start_run():

    # Fit model
    grid_search.fit(X_train_scaled, y_train)

    # Log child runs for each hyperparameter combination
    for i, params in enumerate(grid_search.cv_results_['params']):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", grid_search.cv_results_['mean_test_score'][i])

    # Best model details
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best CV Score:", best_score)

    # Predict and evaluate on validation set
    y_pred = best_model.predict(X_valid_scaled)
    accuracy = accuracy_score(y_valid, y_pred)
    mlflow.log_metric("val_accuracy", accuracy)

    # Log best model and params
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_score", best_score)
    mlflow.sklearn.log_model(best_model, artifact_path="best_xgb_model")

    # Optional: save test predictions
    X_test_scaled = scaler.transform(test[FEATURES])
    test_preds = best_model.predict_proba(X_test_scaled)[:, 1]
    sub = pd.read_csv("./data/sample_submission.csv")
    sub["rainfall"] = test_preds
    sub.to_csv("./data/submission_xgb_grid.csv", index=False)
    mlflow.log_artifact("./data/submission_xgb_grid.csv")

    # Add metadata
    mlflow.set_tags({
        "Author": "Krishna",
        "Stage": "Hyperparameter Tuning",
        "Framework": "XGBoost"
    })
