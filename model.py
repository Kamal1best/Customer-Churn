from scipy.stats import uniform, randint
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from cleaning_preprocessing import preprocess_data, clean_data
from ML import split_data
from feature_engineering import perform_feature_engineering

# Load and clean data
df = pd.read_csv('churn.csv')
data = clean_data(df.copy())
preprocessed_data = preprocess_data(data)
engineered_data = perform_feature_engineering(preprocessed_data)
feature_names = engineered_data.drop('churn_risk_score', axis=1).columns

X_train, X_test, y_train, y_test = split_data(engineered_data)

# Initialize LightGBM model
lgb_model = LGBMClassifier(random_state=42, verbose=-1)

# Define hyperparameter search space
param_dist = {
    'num_leaves': randint(20, 100),
    'max_depth': [-1, 10, 20],
    'learning_rate': uniform(0.01, 0.1),
    'n_estimators': [100, 200, 300],
    'min_child_samples': randint(10, 60),
    'subsample': uniform(0.7, 0.3)
}

# Start MLflow experiment
with mlflow.start_run(run_name="LightGBM_RandomSearch"):

    # Perform randomized hyperparameter search
    random_search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='recall',
        n_jobs=1,
        random_state=42
    )

    y_pred_lg = random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_score = random_search.best_score_

    # Log best parameters
    for k, v in best_params.items():
        print(f"{k}: {v}")
        mlflow.log_param(k, v)

    # Log best cross-validation recall score
    mlflow.log_metric("best_cv_recall", best_score)

    # Final model and prediction
    best_lgb = random_search.best_estimator_
    y_pred = best_lgb.predict(X_test)

    # Log test accuracy
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", acc)

    # Print results
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Log model to MLflow with input example and signature
    input_example = X_test[:1]
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(best_lgb, "best_lgb_model", input_example=input_example, signature=signature)

    # Save model locally
    joblib.dump(best_lgb, 'best_lgb_model.pkl')
    print("\n✅ Model saved as 'best_lgb_model.pkl'")

    # 🔍 Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    #  Feature Importance plot
    importances = best_lgb.feature_importances_
    features = feature_names[:len(importances)] 
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances, y=features)
    plt.title("Feature Importances")
    plt.tight_layout()
    fi_path = "feature_importance.png"
    plt.savefig(fi_path)
    plt.close()
    mlflow.log_artifact(fi_path)
