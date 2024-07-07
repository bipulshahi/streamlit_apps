# -*- coding: utf-8 -*-
"""functionUp_23rd_June_creatingModel.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mbCtpWYyM5Ec_7Jg24_KhAqxcf1NKLtW
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import os

raw_train = pd.read_csv('train_loan.csv')
raw_test = pd.read_csv('test_loan.csv')

raw_train.head()

raw_train.columns

raw_test.columns

raw_train['Loan_Status']

raw_train.nunique()

train_df = raw_train.copy()
test_df = raw_test.copy()

train_df.info()

train_y = train_df['Loan_Status'].copy()

train_df = train_df.drop('Loan_Status' , axis = 'columns')

train_df = train_df.drop('Loan_ID' , axis = 'columns')
test_df = test_df.drop('Loan_ID' , axis = 'columns')

train_df[train_df.duplicated()]

test_df[test_df.duplicated()]

test_df = test_df.drop_duplicates()

test_df[test_df.duplicated()]

#Missing values analysis
train_df.isna().sum()

train_df.nunique()

train_df.columns

#Numeric --> mean
#Categorical --> mode

num_cols = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]

cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

cat_imputer = SimpleImputer(strategy = "most_frequent")
cat_imputer.fit(train_df[cat_cols])

train_df[cat_cols] = cat_imputer.transform(train_df[cat_cols])
test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])

num_imputer = SimpleImputer(strategy = "mean")
num_imputer.fit(train_df[num_cols])

train_df[num_cols] = num_imputer.transform(train_df[num_cols])
test_df[num_cols] = num_imputer.transform(test_df[num_cols])

train_df.isna().sum()

train_df.head()

#preprocessing as per domain knowledge

train_df['ApplicantIncome'] = train_df['ApplicantIncome'] + train_df['CoapplicantIncome']
test_df['ApplicantIncome'] = test_df['ApplicantIncome'] + test_df['CoapplicantIncome']

#drop coapplicant income
train_df = train_df.drop('CoapplicantIncome' , axis = 'columns')
test_df = test_df.drop('CoapplicantIncome' , axis = 'columns')

train_df.head()

train_df.nunique()

train_df['Dependents'].unique()

train_df['Property_Area'].unique()

for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    print(le.classes_)
    test_df[col] = le.fit_transform(test_df[col])



train_df.head()

num_cols

num_cols.remove('CoapplicantIncome')
num_cols

import matplotlib.pyplot as plt

plt.hist(train_df['ApplicantIncome'])

plt.show()

mean_apIncome = train_df['ApplicantIncome'].mean()
std_apIncome = train_df['ApplicantIncome'].std()

print(mean_apIncome)
print(std_apIncome)

#Range of 3rd standards deviation
print(mean_apIncome - 3*std_apIncome)
print(mean_apIncome + 3*std_apIncome)

len(train_df)

#samples within 3rd standard deviation
((train_df['ApplicantIncome'] < mean_apIncome + 3*std_apIncome) & (train_df['ApplicantIncome'] > mean_apIncome - 3*std_apIncome)).sum()/len(train_df)

plt.hist(np.log(train_df['ApplicantIncome']))

plt.show()

mean_apIncome_log = np.log(train_df['ApplicantIncome']).mean()
std_apIncome_log = np.log(train_df['ApplicantIncome']).std()

print(mean_apIncome_log)
print(std_apIncome_log)

#range of 3rd standard deviation

print(mean_apIncome_log - 3*std_apIncome_log)
print(mean_apIncome_log + 3*std_apIncome_log)

#samples within 3rd standard deviation
print(((np.log(train_df['ApplicantIncome']) < (mean_apIncome_log + 3*std_apIncome_log)) & (np.log(train_df['ApplicantIncome']) > (mean_apIncome_log - 3*std_apIncome_log))).sum())
print(((np.log(train_df['ApplicantIncome']) < (mean_apIncome_log + 3*std_apIncome_log)) & (np.log(train_df['ApplicantIncome']) > (mean_apIncome_log - 3*std_apIncome_log))).sum()/len(train_df))

#log transformation on numerical columns
train_df[num_cols] = np.log(train_df[num_cols])
test_df[num_cols] = np.log(test_df[num_cols])

#scaling
minmax = MinMaxScaler()
train_df = minmax.fit_transform(train_df)
test_df = minmax.transform(test_df)

#Building up the model

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train_df,train_y,train_size=0.7,random_state=0)

x_train[7]

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log.fit(x_train,y_train)

ytrainPred = log.predict(x_train)
ytestPred = log.predict(x_test)

from sklearn.metrics import accuracy_score

print("Train Data evaluation metrics")
print(accuracy_score(y_train,ytrainPred))

print("Test Data evaluation metrics")
print(accuracy_score(y_test,ytestPred))

"""**Serialization or save the model**"""

import pickle

with open( 'modellog.pkl' , 'wb') as f:
    pickle.dump(log,f)

np.log(7654)

with open('minmaxscaler.pkl' , 'wb') as f:
    pickle.dump(minmax,f)

training_data_columns = list(raw_train.columns).copy()
print(training_data_columns)

training_data_columns.remove('Loan_ID')
training_data_columns.remove('CoapplicantIncome')
training_data_columns.remove('Loan_Status')

training_data_columns

import json

columns = {"features" : training_data_columns}

with open('columns.json' , 'w') as f:
    f.write(json.dumps(columns))

log.coef_

le.transform(['Urban'])

import pickle

#check pickle version
print(pickle.format_version)

import numpy as np
np.__version__

import sklearn
sklearn.__version__

import joblib
joblib.__version__

# Paths to locally stored joblib files
repo_dir = os.path.dirname(os.path.abspath(__file__))
model_name = os.path.join(repo_dir, "modellog.joblib")
scaler_name = os.path.join(repo_dir, "minmaxscaler.joblib")

# Your model training and other code...

# Save the model and scaler
with open(model_name, 'wb') as f:
    joblib.dump(log, f)

with open(scaler_name, 'wb') as f:
    joblib.dump(minmax, f)

