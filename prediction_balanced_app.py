#load the necessary libraries
import streamlit as st
import pandas as pd
import pickle

#load the model
with open('loan_risk_balanced.pkl', 'rb') as f:
    model= pickle.load(f)
    
    
#streamlit app
st.title("Predicting Loan Risk App")
st.write("Enter the details below to predict the loan risk of the client")

#collect the input from the client
loannumber = st.number_input("Loan Number", min_value=1, step=1, value=6)
loanamount = st.number_input("Loan Amount", min_value=0.0, step=1000.0, value=40000.0)
totaldue = st.number_input("Total Due", min_value=0.0, step=1000.0, value=42000.0)
termdays = st.number_input("Term (days)", min_value=1, step=1, value=30)
age = st.number_input("Age", min_value=18, step=1, value=36)
age_category = st.selectbox("Age Category", ["Teen", "Adult", "Senior"])
bank_account_type = st.selectbox("Bank Account Type", ["Savings", "Current", "Other"])
bank_name_clients = st.selectbox("Bank Name", ['Diamond Bank', 'GT Bank', 'EcoBank', 'First Bank', 'Access Bank', 'UBA', 'Union Bank', 'FCMB', 'Zenith Bank', 'Stanbic IBTC', 'Fidelity Bank', 'Wema Bank', 'Sterling Bank', 'Skye Bank','Keystone Bank', 'Heritage Bank', 'Unity Bank','Standard Chartered'])
employment_status_clients = st.selectbox("Employment Status", ['Permanent', 'Unemployed', 'Self-Employed', 'Student', 'Retired', 'Contract'])
level_of_education_clients = st.selectbox("Education Level", ["Uneducated", "Primary", "Secondary", "Graduate", "Postgraduate"])
clients_direction = st.selectbox("Client Direction", ["North", "North-NorthEast", "NorthEast", "East-NorthEast", "East", "East-SouthEast", "SouthEast", "South-SouthEast", "South", "South-SouthWest", "SouthWest", "West-SouthWest", "West", "West-NorthWest", "NorthWest", "North-NorthWest"])

#prediction button
if st.button("Predict the loan risk"):
    data= {
           'loannumber': [loannumber],
           'loanamount': [loanamount],
           'totaldue': [totaldue],
           'termdays': [termdays],
           'age': [age],
           'age_category': [age_category],
           'bank_account_type': [bank_account_type],
           'bank_name_clients': [bank_name_clients],
           'employment_status_clients': [employment_status_clients],
           'level_of_education_clients': [level_of_education_clients],
           'clients_direction': [clients_direction]
           }
    input_df= pd.DataFrame(data)
    
    #predict
    prediction= model.predict(input_df)[0]
    
    #Display the result
    st.success(f"The Prediction of the Loan Risk is {prediction}")

