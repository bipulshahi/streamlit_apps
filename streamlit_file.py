import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import json
import joblib

# Load data
raw_train = pd.read_csv('train_loan.csv')
raw_test = pd.read_csv('test_loan.csv')

# Prepare data
train_df = raw_train.copy()
test_df = raw_test.copy()

train_y = train_df['Loan_Status'].copy()
train_df = train_df.drop(['Loan_Status', 'Loan_ID', 'CoapplicantIncome'], axis=1)
test_df = test_df.drop(['Loan_ID', 'CoapplicantIncome'], axis=1)

# Impute missing values
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

cat_imputer = SimpleImputer(strategy="most_frequent")
train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])

num_imputer = SimpleImputer(strategy="mean")
train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
test_df[num_cols] = num_imputer.transform(test_df[num_cols])

# Feature engineering and transformations
train_df['ApplicantIncome'] = train_df['ApplicantIncome'] + train_df['CoapplicantIncome']
test_df['ApplicantIncome'] = test_df['ApplicantIncome'] + test_df['CoapplicantIncome']

train_df = train_df.drop('CoapplicantIncome', axis='columns')
test_df = test_df.drop('CoapplicantIncome', axis='columns')

train_df[cat_cols] = train_df[cat_cols].apply(LabelEncoder().fit_transform)
test_df[cat_cols] = test_df[cat_cols].apply(LabelEncoder().fit_transform)

train_df[num_cols] = np.log(train_df[num_cols])
test_df[num_cols] = np.log(test_df[num_cols])

# Scaling
minmax = MinMaxScaler()
train_df[num_cols] = minmax.fit_transform(train_df[num_cols])
test_df[num_cols] = minmax.transform(test_df[num_cols])

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(train_df, train_y, train_size=0.7, random_state=0)

# Training model
log = LogisticRegression()
log.fit(x_train, y_train)

# Evaluation
ytrainPred = log.predict(x_train)
ytestPred = log.predict(x_test)

print("Train Data evaluation metrics")
print(accuracy_score(y_train, ytrainPred))

print("Test Data evaluation metrics")
print(accuracy_score(y_test, ytestPred))

# Serialization of model, scaler, and columns
with open('modellog.pkl', 'wb') as f:
    pickle.dump(log, f)

with open('minmaxscaler.pkl', 'wb') as f:
    pickle.dump(minmax, f)

training_data_columns = list(train_df.columns)
training_data_columns.remove('Loan_Status')

columns = {"features": training_data_columns}

with open('columns.json', 'w') as f:
    json.dump(columns, f)

# Alternative using joblib for better compatibility with numpy arrays
joblib.dump(log, 'modellog.joblib')
joblib.dump(minmax, 'minmaxscaler.joblib')
