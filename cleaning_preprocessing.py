import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_data(data):
 #Remove Columns
 data = data.drop(['referral_id','security_no','last_visit_time','joined_through_referral'], axis=1)
 # Replace 'Unknown' values in the 'gender' column with 'F'
 data['gender'] = data['gender'].replace('Unknown', 'F')
 # Replace missing values represented by '?' in 'medium_of_operation' with 'Unknown'
 data['medium_of_operation'] = data['medium_of_operation'].replace('?', "Unknown")
 # Clean and convert 'avg_frequency_login_days'
 data['avg_frequency_login_days'] = data['avg_frequency_login_days'].replace(['?', 'Error'], np.nan)
 data['avg_frequency_login_days'] = pd.to_numeric(data['avg_frequency_login_days'])
# Replace negative values
 for col in data.select_dtypes(include=[np.number]).columns:
    if (data[col] < 0).any():
        valid_median = data.loc[data[col] >= 0, col].median()
        data[col] = data[col].apply(lambda x: valid_median if x < 0 else x) 
 # Fill missing values in 'avg_frequency_login_days' with the mean of the column
 for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            if data[col].nunique() < 10:
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                data[col] = data[col].fillna(data[col].median())               
 return data
def preprocess_data(data):
  # data = clean_data(data)
  def label_encode_categoricals(data, categorical_columns):
   for col in categorical_columns:
     le = LabelEncoder()
     data[col] = le.fit_transform(data[col])
   return data
  
  def one_hot_encode(data, columns):
    for col in columns:
        dummies = pd.get_dummies(data[col], prefix='', prefix_sep='')
        dummies = dummies.astype(int)
        dummies.columns = [f"{col}({val})" for val in dummies.columns]
        data = pd.concat([data, dummies], axis=1)
        data.drop(columns=col, inplace=True)
    return data
  
  data_encoded = data.copy()
  data_encoded = one_hot_encode(data_encoded, ['region_category', 'membership_category','preferred_offer_types','medium_of_operation','internet_option','complaint_status','feedback'])
  data_encoded = label_encode_categoricals(data_encoded, [ 'gender', 'used_special_discount','offer_application_preference','past_complaint'])
  return data_encoded
