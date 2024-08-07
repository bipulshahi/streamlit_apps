import streamlit as st
import numpy as np
import joblib

# Paths to locally stored joblib files
scaling_name = "minmaxscaler.joblib"
model_name = "modellog.joblib"

# Load scaling and model from locally stored joblib files
try:
    _scaling = joblib.load(scaling_name)
except Exception as e:
    st.error(f"Error loading {scaling_name}: {e}")
    st.stop()

try:
    _model = joblib.load(model_name)
except Exception as e:
    st.error(f"Error loading {model_name}: {e}")
    st.stop()

# Prediction function remains unchanged
def prediction(Gender, Married, Dependents, Education, Self_Employed, Applicant_Income, Coapplicant_Income,
               LoanAmount, Loan_Amount_Term, Credit_History, Property_Area):
    # Encode categorical features (example function, adjust as needed)
    gender_encode = lambda x: 0 if x == 'Female' else 1
    married_encode = lambda x: 0 if x == 'No' else 1
    dependents_encode = lambda x: 0 if x == '0' else 1 if x == '1' else 2 if x == '2' else 3
    education_encode = lambda x: 0 if x == 'Graduate' else 1
    self_employed_encode = lambda x: 0 if x == 'No' else 1
    property_area_encode = lambda x: 0 if x == 'Rural' else 1 if x == 'Semiurban' else 2
    
    gender = gender_encode(Gender)
    married = married_encode(Married)
    dependents = dependents_encode(Dependents)
    education = education_encode(Education)
    selfemployed = self_employed_encode(Self_Employed)
    credithistory = Credit_History
    propertyarea = property_area_encode(Property_Area)
    
    # Log transform numerical features (example transformation, adjust as needed)
    applicantincome = Applicant_Income + Coapplicant_Income
    applicantincome_log = np.log(applicantincome)
    loanamount_log = np.log(LoanAmount)
    loanamountterm_log = np.log(Loan_Amount_Term)
    
    # Combine all features into a single array
    input_data = np.array([[gender, married, dependents, education, selfemployed, 
                            applicantincome_log, loanamount_log, loanamountterm_log, 
                            credithistory, propertyarea]])
    
    # Make prediction using the loaded model
    prediction = _model.predict(input_data)[0]
    
    return prediction

# Streamlit app code remains unchanged
def main():
    st.title("Welcome to Loan Application")
    st.header("Please enter your details to proceed with your loan application")

    Gender = st.selectbox("Gender", ("Male", "Female"))
    Married = st.selectbox("Married", ("Yes", "No"))
    Dependents = st.selectbox("Dependents", ("0", "1", "2", "3+"))
    Education = st.selectbox("Education", ("Graduate", "Not Graduate"))
    Self_Employed = st.selectbox("Self Employed", ("Yes", "No"))
    Applicant_Income = st.number_input("Applicant Income")
    Coapplicant_Income = st.number_input("Coapplicant Income")
    LoanAmount = st.number_input("Loan Amount")
    Loan_Amount_Term = st.number_input("Loan Amount Term")
    Credit_History = st.number_input("Credit History")
    Property_Area = st.selectbox("Property Area", ("Rural", "Urban", "Semi Urban"))

    if st.button("Predict"):
        result = prediction(Gender, Married, Dependents, Education, Self_Employed, Applicant_Income, Coapplicant_Income,
                            LoanAmount, Loan_Amount_Term, Credit_History, Property_Area)

        if result == "Y":
            st.success("Your loan application is approved")
        else:
            st.error("Your loan application is rejected")

if __name__ == "__main__":
    main()
