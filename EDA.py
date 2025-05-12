import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from cleaning_preprocessing import clean_data
import pandas as pd
df= pd.read_csv('churn.csv')
data= df.copy()
cleaned_data = clean_data(data)
numerical_cols = cleaned_data.select_dtypes(include=np.number).columns.tolist()


def Histogram(cleaned_data): 
   cleaned_data[numerical_cols].hist(bins=15, figsize=(15, 10))
   plt.suptitle("Histograms of Numerical Features")
   plt.tight_layout()
   plt.show()



def Boxplot(cleaned_data):
   # Boxplots to identify outliers
   plt.figure(figsize=(15, 10))
   for i, col in enumerate(numerical_cols):
      plt.subplot(3, 3, i + 1)
      sns.boxplot(x=cleaned_data[col], color='skyblue')
      plt.title(f'Boxplot of {col}')
   plt.tight_layout()
   plt.show()


def heatmap(cleaned_data):
   # Correlation heatmap
   plt.figure(figsize=(12, 8))
   corr_matrix = cleaned_data[numerical_cols].corr()
   sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
   plt.title("Correlation Heatmap")
   plt.show()


def Pairplot(cleaned_data):
   # Pairplot for selected features
   selected_features = numerical_cols[:5]
   sns.pairplot(cleaned_data[selected_features])
   plt.suptitle("Pairplot of Selected Features")
   plt.show()
