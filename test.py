import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import base64

from cleaning_preprocessing import preprocess_data , clean_data
from feature_engineering import perform_feature_engineering
from ML import split_data_app
from sklearn.metrics import accuracy_score, classification_report
best_lgb = joblib.load('best_lgb_model.pkl')
df= pd.read_csv('test_data.csv')
data = df.copy()
cleaned_data = clean_data(data)
preprocessed_data = preprocess_data(data)
engineered_data = perform_feature_engineering(preprocessed_data)
X_test, y_test = split_data_app(engineered_data)

y_pred = best_lgb.predict(X_test)

    # Metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
print(accuracy)
