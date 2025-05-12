import streamlit as st
import pandas as pd
import joblib
import os
from cleaning_preprocessing import preprocess_data, clean_data
from feature_engineering import perform_feature_engineering
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, datetime
import mlflow
import mlflow.sklearn

# إعداد Streamlit
st.set_page_config(page_title="Customer Churn Prediction App", layout="wide")
st.title("Customer Churn Prediction App")

# ✅ تهيئة MLflow
mlflow.set_tracking_uri("file:mlruns")

# ✅ تحميل الموديل
MODEL_PATH = "best_lgb_model.pkl"  # أو المسار الصحيح لنموذجك
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        st.warning("Model file not found. Please train the model first.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# واجهة المستخدم
st.sidebar.header("Upload or Input Data")
input_method = st.sidebar.radio("Choose input method", ["Upload CSV", "Enter Data Manually"])

data = None
manual_mode = False

# معالجة الإدخال
if input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Dataset")
        st.dataframe(data.head())

        if model is not None:
            try:
                # التنظيف والمعالجة
                cleaned = clean_data(data.copy())
                preprocessed = preprocess_data(cleaned)
                engineered = perform_feature_engineering(preprocessed)
                
                # التأكد من وجود كل الميزات المطلوبة
                common_features = [
                    'membership_category(Basic Membership)', 'feedback(Products always in Stock)',
                    'membership_category(No Membership)', 'log_customer_tenure',
                    'feedback(Quality Customer Care)', 'feedback(Reasonable Price)',
                    'log_points_in_wallet', 'membership_category(Silver Membership)',
                    'feedback(User Friendly Website)', 'membership_category(Gold Membership)',
                    'membership_category(Platinum Membership)', 'membership_category(Premium Membership)'
                ]
                
                for feature in common_features:
                    if feature not in engineered.columns:
                        engineered[feature] = 0
                
                engineered = engineered[common_features]
                
                # التنبؤ
                predictions = model.predict(engineered)
                data['Churn Prediction'] = predictions
                
                # عرض النتائج
                st.subheader("Prediction Results")
                st.write(data[['Churn Prediction']].value_counts().rename_axis('Churn').reset_index(name='Count'))
                
                # التصورات البيانية
                visualize_results(data)
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

elif input_method == "Enter Data Manually":
    manual_mode = True
    manual_input = get_manual_input()
    data = pd.DataFrame([manual_input])
    
    if st.sidebar.button('Predict'):
        if model is not None:
            try:
                # المعالجة والتنبؤ (مشابه لما سبق)
                processed_data = process_and_predict(data.copy(), model)
                display_prediction_result(processed_data)
            except Exception as e:
                st.error(f"Error during manual prediction: {e}")
        else:
            st.warning("Model not loaded. Cannot make predictions.")

# الدوال المساعدة
def visualize_results(data):
    """عرض النتائج بشكل بياني"""
    churn_counts = data['Churn Prediction'].value_counts().sort_index()
    churn_labels = ["Not Churn", "Churn"] if len(churn_counts) == 2 else churn_counts.index.astype(str)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x=churn_labels, y=churn_counts.values, palette="Set2", ax=ax1)
    ax1.set_title("Churn Distribution")
    
    ax2.pie(churn_counts.values, labels=churn_labels, autopct='%1.1f%%', 
            colors=["#66c2a5", "#fc8d62"])
    ax2.set_title("Churn Percentage")
    
    st.pyplot(fig)

def get_manual_input():
    """الحصول على الإدخال اليدوي من المستخدم"""
    return {
        'age': st.sidebar.number_input('Age', 18, 100, 30),
        'gender': st.sidebar.selectbox('Gender', ['M', 'F']),
        # ... (بقية الحقول كما هي)
    }

def process_and_predict(data, model):
    """معالجة البيانات والتنبؤ"""
    cleaned = clean_data(data)
    preprocessed = preprocess_data(cleaned)
    engineered = perform_feature_engineering(preprocessed)
    
    # التأكد من وجود كل الميزات المطلوبة
    common_features = [
        'membership_category(Basic Membership)', 'feedback(Products always in Stock)',
        # ... (بقية الميزات)
    ]
    
    for feature in common_features:
        if feature not in engineered.columns:
            engineered[feature] = 0
    
    engineered = engineered[common_features]
    data['Churn Prediction'] = model.predict(engineered)
    return data

def display_prediction_result(data):
    """عرض نتيجة التنبؤ"""
    prediction = data['Churn Prediction'].iloc[0]
    if prediction == 1:
        st.error("⚠️ Warning: This customer is at risk of leaving!")
    else:
        st.success("✅ Good news: This customer is likely to stay!")
