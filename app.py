import streamlit as st
import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from cleaning_preprocessing import preprocess_data, clean_data
from feature_engineering import perform_feature_engineering
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile

# Helper functions
def plot_results(data):
    """Display results as visualizations"""
    churn_counts = data['Churn Prediction'].value_counts().sort_index()
    labels = ["Not Churn", "Churn"] if len(churn_counts) == 2 else churn_counts.index.astype(str)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(x=labels, y=churn_counts.values, palette="Set2", ax=ax1)
    ax1.set_title("Churn Distribution")

    ax2.pie(churn_counts.values, labels=labels, autopct='%1.1f%%', 
            colors=["#66c2a5", "#fc8d62"])
    ax2.set_title("Churn Percentage")

    st.pyplot(fig)

    if len(churn_counts) == 2 and churn_counts[1] > churn_counts[0]:
        st.warning("Warning: More customers are churning than staying!")

def get_manual_input():
    """Get manual input data"""
    return {
        'age': st.sidebar.number_input('Age', 18, 100, 30),
        'gender': st.sidebar.selectbox('Gender', ['Male', 'Female']),
        'region_category': st.sidebar.selectbox('Region Category', ['City', 'Town', 'Village']),
        'membership_category': st.sidebar.selectbox('Membership Category', [
            'No Membership', 'Basic Membership', 'Silver Membership',
            'Gold Membership', 'Platinum Membership', 'Premium Membership']),
        'medium_of_operation': st.sidebar.selectbox('Medium of Operation', ['Desktop', 'Smartphone']),
        'internet_option': st.sidebar.selectbox('Internet Option', ['Wi-Fi', 'Mobile Data', 'Fiber Optic']),
        'days_since_last_login': st.sidebar.slider('Days Since Last Login', 0, 60, 10),
        'avg_time_spent': st.sidebar.slider('Average Time Spent', 0.0, 1000.0, 300.0),
        'avg_transaction_value': st.sidebar.slider('Average Transaction Value', 0.0, 100000.0, 20000.0),
        'avg_frequency_login_days': st.sidebar.selectbox('Average Login Frequency (days)', ['10', '15', '22', '6', '17', '20+']),
        'points_in_wallet': st.sidebar.slider('Points in Wallet', 0.0, 1000.0, 500.0),
        'used_special_discount': st.sidebar.selectbox('Used Special Discount', ['Yes', 'No']),
        'offer_application_preference': st.sidebar.selectbox('Offer Application Preference', ['Yes', 'No']),
        'preferred_offer_types': st.sidebar.selectbox('Preferred Offer Types', [
            'Gift Vouchers/Coupons', 'Credit/Debit Card Offers', 'Without Offers']),
        'past_complaint': st.sidebar.selectbox('Past Complaint', ['Yes', 'No']),
        'complaint_status': st.sidebar.selectbox('Complaint Status', ['Solved', 'Unsolved', 'Solved in Follow-up']),
        'feedback': st.sidebar.selectbox('Feedback', [
            'Poor Product Quality', 'No reason specified', 'Poor Website', 'Poor Customer Service',
            'Reasonable Price', 'Too many ads', 'User Friendly Website',
            'Products always in Stock', 'Quality Customer Care']),
        'joining_date': st.sidebar.date_input('Joining Date', datetime.today())
    }

def process_data(data):
    """Process data before prediction"""
    cleaned = clean_data(data)
    preprocessed = preprocess_data(cleaned)
    engineered = perform_feature_engineering(preprocessed)

    required_features = [
        'membership_category(Basic Membership)', 
        'feedback(Products always in Stock)',
        'membership_category(No Membership)', 
        'log_customer_tenure',
        'feedback(Quality Customer Care)', 
        'feedback(Reasonable Price)',
        'log_points_in_wallet', 
        'membership_category(Silver Membership)',
        'feedback(User Friendly Website)', 
        'membership_category(Gold Membership)',
        'membership_category(Platinum Membership)', 
        'membership_category(Premium Membership)'
    ]

    for feature in required_features:
        if feature not in engineered.columns:
            engineered[feature] = 0

    engineered = engineered[required_features]
    data['Churn Prediction'] = model.predict(engineered)
    return data

def display_prediction(data):
    """Display prediction result"""
    prediction = data['Churn Prediction'].iloc[0]
    if prediction == 1:
        st.error("⚠️ Warning: This customer is at risk of churning!")
    else:
        st.success("✅ Good news: This customer is likely to stay!")
    
    st.subheader("Prediction Details")
    st.write(data)

# Streamlit UI
st.set_page_config(page_title="Customer Churn Prediction App", layout="wide")
st.title("Customer Churn Prediction App")

# 1. Initialize MLflow safely
try:
    # Create temporary directory for MLflow if there are permission issues
    temp_dir = tempfile.mkdtemp()
    mlflow.set_tracking_uri(f"file:{temp_dir}")
    st.success("MLflow initialized successfully")
except Exception as e:
    st.error(f"Error initializing MLflow: {str(e)}")

# 2. Load the model
MODEL_PATH = "best_lgb_model.pkl"
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        st.success("Prediction model loaded successfully")
    else:
        st.warning("Model file not found. Please ensure it's in the correct path.")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

# 3. Data input interface
st.sidebar.header("Data Input Method")
input_method = st.sidebar.radio("Choose input method", ["Upload CSV", "Manual Input"])

data = None

# 4. Handle file upload
if input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data")
            st.dataframe(data.head())
            
            if model is not None:
                # Data processing
                cleaned_data = clean_data(data.copy())
                preprocessed_data = preprocess_data(cleaned_data)
                engineered_data = perform_feature_engineering(preprocessed_data)
                
                # List of expected features
                required_features = [
                    'membership_category(Basic Membership)', 
                    'feedback(Products always in Stock)',
                    'membership_category(No Membership)', 
                    'log_customer_tenure',
                    'feedback(Quality Customer Care)', 
                    'feedback(Reasonable Price)',
                    'log_points_in_wallet', 
                    'membership_category(Silver Membership)',
                    'feedback(User Friendly Website)', 
                    'membership_category(Gold Membership)',
                    'membership_category(Platinum Membership)', 
                    'membership_category(Premium Membership)'
                ]
                
                # Ensure all required features exist
                for feature in required_features:
                    if feature not in engineered_data.columns:
                        engineered_data[feature] = 0
                
                # Reorder columns as required
                engineered_data = engineered_data[required_features]
                
                # Make predictions
                predictions = model.predict(engineered_data)
                data['Churn Prediction'] = predictions
                
                # Show results
                st.subheader("Prediction Results")
                st.write(data[['Churn Prediction']].value_counts().rename_axis('Status').reset_index(name='Count'))
                
                # Show visualizations
                plot_results(data)
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

# 5. Handle manual input
elif input_method == "Manual Input":
    manual_input = get_manual_input()
    data = pd.DataFrame([manual_input])

    if st.sidebar.button('Predict'):
        if model is not None:
            try:
                # Process data and make prediction
                processed_data = process_data(data.copy())
                display_prediction(processed_data)
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        else:
            st.warning("Model not loaded. Cannot make predictions.")
