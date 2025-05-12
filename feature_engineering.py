import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse  

def perform_feature_engineering(data):
    # Step 1: Clean column names
    data.columns = data.columns.str.strip()

    # Step 2: Copy data for FE
    data_fe = data.copy()

    # Step 3: Customer Tenure
    if 'registration_days_ago' in data_fe.columns:
        data_fe['customer_tenure'] = data_fe['registration_days_ago']
    elif 'joining_date' in data_fe.columns:
        data_fe['joining_date'] = data_fe['joining_date'].astype(str).str.strip().str.replace(r'[^0-9/-]', '', regex=True)

        def try_parse_date(val):
            try:
                return parse(val, dayfirst=True)
            except:
                return pd.NaT

        data_fe['joining_date'] = data_fe['joining_date'].apply(try_parse_date)

        today = pd.to_datetime(datetime.today().date())
        data_fe['customer_tenure'] = (today - data_fe['joining_date']).dt.days

    # Step 4: Usage Pattern
    if 'avg_transaction_value' in data_fe.columns and 'avg_time_spent' in data_fe.columns:
        data_fe['usage_pattern'] = data_fe['avg_transaction_value'] * data_fe['avg_time_spent']

    # Step 5: Interaction Frequency
    if 'days_since_last_login' in data_fe.columns:
        data_fe['interaction_frequency'] = 1 / (1 + data_fe['days_since_last_login'])
    else:
        data_fe['interaction_frequency'] = np.random.uniform(0.1, 1.0, size=len(data_fe))

    # Step 6: Engagement Score
    if 'usage_pattern' in data_fe.columns:
        data_fe['engagement_score'] = data_fe['usage_pattern'] * data_fe['interaction_frequency']
    else:
        data_fe['engagement_score'] = data_fe['interaction_frequency']

    # Step 7: Log Transform
    numeric_cols = data_fe.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (data_fe[col] > 0).all():
            data_fe[f'log_{col}'] = np.log(data_fe[col] + 1e-6)

    # Step 8: Spend per day
    if 'total_spent' in data_fe.columns and 'customer_tenure' in data_fe.columns:
        data_fe['spend_per_day'] = data_fe['total_spent'] / (data_fe['customer_tenure'] + 1)

    # Step 9: Join Month & Year
    if 'joining_date' in data_fe.columns:
        data_fe['join_month'] = data_fe['joining_date'].dt.month
        data_fe['join_year'] = data_fe['joining_date'].dt.year

    return data_fe
