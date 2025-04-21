import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the machine learning model and encoders
model = joblib.load("best_xgbmodel.pkl")
gender_encoder = joblib.load("person_gender_encoder.pkl")
default_encoder = joblib.load("previous_loan_defaults_on_file_encoder.pkl")

def main():
    st.title("Loan Status Prediction App")

    # Input fields from user
    person_age = st.number_input("Enter Age", min_value=18, max_value=100)
    person_gender = st.radio("Select Gender", options=["male", "female"])
    person_income = st.number_input("Enter Monthly Income", min_value=1000)
    person_education = st.selectbox("Select Education", options=['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
    previous_loan_defaults_on_file = st.radio("Previous Loan Defaults", options=["No", "Yes"])
    person_emp_exp = st.number_input("Employment Experience (in years)", min_value=0)
    person_home_ownership = st.selectbox("Home Ownership", options=["Own", "Rent", "Mortgage"])
    loan_intent = st.selectbox("Loan Intent", options=['VENTURE', 'EDUCATION', 'MEDICAL', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
    loan_amnt = st.number_input("Loan Amount", min_value=500)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0)
    cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=1)

    data = {
        'person_age': int(person_age),
        'person_gender': person_gender,
        'person_income': float(person_income),
        'person_education': person_education,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file,
        'person_emp_exp': float(person_emp_exp),
        'person_home_ownership': person_home_ownership,
        'loan_intent': loan_intent,
        'loan_amnt': float(loan_amnt),
        'loan_int_rate': float(loan_int_rate),
        'cb_person_cred_hist_length': int(cb_person_cred_hist_length)
    }

    df = pd.DataFrame([list(data.values())], columns=list(data.keys()))

    # Encode gender and default using LabelEncoder
    df['person_gender'] = gender_encoder.transform(df['person_gender'])
    df['previous_loan_defaults_on_file'] = default_encoder.transform(df['previous_loan_defaults_on_file'])

    # Ordinal encode education
    education_order = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
    df['person_education'] = pd.Categorical(df['person_education'], categories=education_order, ordered=True).codes

    # One-hot encode
    df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'], drop_first=False)

    # Align with training model's expected features
    required_columns = model.get_booster().feature_names
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[required_columns]

    if st.button('Make Prediction'):
        result = make_prediction(df)
        prediction_text = "Approved" if result == 1 else "Rejected"
        st.success(f"The loan application is likely to be: {prediction_text}")

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
