# train_risk_prediction.py

#import libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

#load data
df= pd.read_csv("predicting_loan_risk.csv")

#split into target and feature--- I am doing the summary of the balanced target column
X = df.drop(['good_bad_flag', 'customerid','creationdate', 'birthdate', 'approveddate','systemloanid', 'longitude_gps', 'latitude_gps'], axis=1)
y = df['good_bad_flag']

# i split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)


#split into num_cols and cat_cols
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

#column transformer
num_transformer= Pipeline(steps=[
    ('imputer', SimpleImputer(strategy= 'median')),
    ('scaler', StandardScaler())
])
cat_transformer= Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False, drop='first',handle_unknown='ignore'))
])

#combine our transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

#define the best model
model= GradientBoostingClassifier()

pipeline= Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])
pipeline.fit(X_train, y_train)

#save our model
with open('loan_risk.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
    
print("Model trained and saved as loan_risk.pkl")

#predict on an example
data= {
       'loannumber': [6],
       'loanamount': [40000.0],
       'totaldue': [42000.0],
       'termdays': [30],
       'age': [36],
       'age_category': ['Adult'],
       'bank_account_type': ['Savings'],
       'bank_name_clients': ['Access Bank'],
       'employment_status_clients': ['Permanent'],
       'level_of_education_clients': ['Graduate'],
       'clients_direction': ['NorthEast']
       }

#convert to a dataframe
sample_df= pd.DataFrame(data)

#load the model
with open('loan_risk.pkl', 'rb') as f:
    model= pickle.load(f)
    
#predict
prediction= model.predict(sample_df)


print(f"Loan Risk Prediction is {prediction}")











