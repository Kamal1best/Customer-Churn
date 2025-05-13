# Import required libraries
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from feature_engineering import perform_feature_engineering
from cleaning_preprocessing import preprocess_data ,clean_data
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import numpy as np
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, early_stopping
from skopt import BayesSearchCV  
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from skopt.space import Integer, Categorical

def split_data_app(data):
    common_features = [
    'membership_category(Basic Membership)', 'feedback(Products always in Stock)',
    'membership_category(No Membership)', 'log_customer_tenure',
    'feedback(Quality Customer Care)', 'feedback(Reasonable Price)',
    'log_points_in_wallet', 'membership_category(Silver Membership)',
    'feedback(User Friendly Website)', 'membership_category(Gold Membership)',
    'membership_category(Platinum Membership)', 'membership_category(Premium Membership)'
    ]
    X = data[common_features]
    y = data['churn_risk_score']
    X = X.dropna()
    y = y.loc[X.index]
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y
def split_data(data):
# Features that are common between Random Forest and Logistic Regression
  common_features = [
    'membership_category(Basic Membership)', 'feedback(Products always in Stock)',
    'membership_category(No Membership)', 'log_customer_tenure',
    'feedback(Quality Customer Care)', 'feedback(Reasonable Price)',
    'log_points_in_wallet', 'membership_category(Silver Membership)',
    'feedback(User Friendly Website)', 'membership_category(Gold Membership)',
    'membership_category(Platinum Membership)', 'membership_category(Premium Membership)'
  ]

  X = data[common_features]
  y = data['churn_risk_score']
  X = X.dropna()
  y = y.loc[X.index]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Apply SMOTE to training data only
  smote = SMOTE(random_state=42)
  X_train, y_train = smote.fit_resample(X_train, y_train)
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  return X_train, X_test, y_train, y_test



def svm_model(X_train, X_test, y_train, y_test):
    # Create an SVM model with a linear kernel using StandardScaler in a pipeline
    svm_model = SVC(kernel='linear', random_state=42)

    # Train the model
    svm_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = svm_model.predict(X_test)

    # Evaluate the model
    print("Evaluation of SVM Model (Linear Kernel):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))




def LogisticRegression_model(X_train, X_test, y_train, y_test):
    lr_model = LogisticRegression(solver='liblinear', random_state=42)
    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)

    print("Evaluation of Logistic Regression Model:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    
def knn_model(X_train, X_test, y_train, y_test):
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    print("Evaluation of K-Nearest Neighbors Classifier:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))






def lg_model(X_train, X_test, y_train, y_test):
    model = LGBMClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    





def  SVM_after_tuning(X_train, X_test, y_train, y_test):

    pipeline = make_pipeline(SVC(probability=True, random_state=42))

    param_dist = {
            'svc__C': uniform(0.1, 10),              
            'svc__gamma': ['scale', 'auto'],
            'svc__kernel': ['linear', 'rbf', 'poly'], 
            'svc__degree': randint(2, 5)   }

    random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            verbose=1,
            n_jobs=1,
            random_state=42
        )

    y_pred_svm =random_search.fit(X_train, y_train)

    print(" Best Parameters:", random_search.best_params_)
    print(f" Best CV Score: {random_search.best_score_:.4f}")

    y_pred_svm = random_search.predict(X_test)
    print("\n Evaluation on Test Set:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_svm))
    return y_pred_svm



def lg_model_tuning(X_train, X_test, y_train, y_test):
        lgb_model = LGBMClassifier(random_state=42, verbose=-1)

        param_dist = {
            'num_leaves': randint(20, 100),
            'max_depth': [-1, 10, 20],
            'learning_rate': uniform(0.01, 0.1),
            'n_estimators': [100, 200, 300],
            'min_child_samples': randint(10, 60),
            'subsample': uniform(0.7, 0.3)}
        

        # RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=lgb_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=3,
            scoring='recall',
            n_jobs=1,
            random_state=42
        )

        random_search.fit(X_train, y_train)

        print("Best Parameters:")
        for k, v in random_search.best_params_.items():
            print(f"   {k}: {v}")

        print(f"\n Best Recall (Cross-Validation): {random_search.best_score_:.4f}")

        best_lgb = random_search.best_estimator_
        y_pred_lg = best_lgb.predict(X_test)

        print(f"\n Test Accuracy: {accuracy_score(y_test, y_pred_lg):.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred_lg))
        
        return y_pred_lg


def knn_model_tuning(X_train, X_test, y_train, y_test):

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13,15, 17, 19, 21, 31, 41, 45, 61],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }


    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='recall', n_jobs=1)
    grid_search.fit(X_train, y_train)

    best_knn = grid_search.best_estimator_

    y_pred_knn = best_knn.predict(X_test)

    print(" Best Parameters:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)
    print("Accuracy on Test Set:", accuracy_score(y_test, y_pred_knn))
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred_knn))
    return y_pred_knn


def LogisticRegression_tuning(X_train, X_test, y_train, y_test):
    lr_model = LogisticRegression(solver='liblinear', random_state=42)

    param_space = {
        'C': Integer(1, 1000), 
        'penalty': Categorical(['l1', 'l2']),
        'max_iter': Integer(100, 1000)}
    bayes_search = BayesSearchCV(
        estimator=lr_model,
        search_spaces=param_space,
        n_iter=50,
        cv=10,
        scoring='recall',
        n_jobs=1,
        random_state=42,
        verbose=0
    )

    bayes_search.fit(X_train, y_train)

    best_lr_model = bayes_search.best_estimator_
    y_pred_lr = best_lr_model.predict(X_test)

    print(" Evaluation of Tuned Logistic Regression Model (Bayesian Optimization):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr))

    print(" Best Parameters found by Bayesian Optimization:")
    print(bayes_search.best_params_)
    
    return y_pred_lr
    
    
    



