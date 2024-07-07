import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import joblib
import json
import streamlit as st

# Load data
raw_train = pd.read_csv('train_loan.csv')
raw_test = pd.read_csv('test_loan.csv')

# Preprocessing steps...

# Fit and transform scalers, imputers, etc.

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(train_df, train_y, train_size=0.7, random_state=0)

# Build and train model
log = LogisticRegression()
log.fit(x_train, y_train)

# Predictions and evaluation
ytrainPred = log.predict(x_train)
ytestPred = log.predict(x_test)

print("Train Data evaluation metrics")
print(accuracy_score(y_train, ytrainPred))

print("Test Data evaluation metrics")
print(accuracy_score(y_test, ytestPred))

# Save artifacts
artifacts_dir = './artifacts/'  # Define artifacts directory relative to the script
os.makedirs(artifacts_dir, exist_ok=True)  # Ensure artifacts directory exists

# Serialize model
model_path = os.path.join(artifacts_dir, 'modellog.joblib')
joblib.dump(log, model_path)

# Serialize scaler or other artifacts
scaler_path = os.path.join(artifacts_dir, 'minmaxscaler.joblib')
joblib.dump(minmax, scaler_path)

# Save column names
training_data_columns.remove('Loan_ID')
training_data_columns.remove('CoapplicantIncome')
training_data_columns.remove('Loan_Status')

columns = {"features": training_data_columns}
columns_path = os.path.join(artifacts_dir, 'columns.json')
with open(columns_path, 'w') as f:
    json.dump(columns, f)

# Additional checks
print(f"Joblib version: {joblib.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")

# Optionally, check file existence
print(f"Model saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
print(f"Columns saved to: {columns_path}")
