# train_risk_prediction.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline   # use imblearn pipeline

# load data
df = pd.read_csv("predicting_loan_risk.csv")

# target & features
X = df.drop(['good_bad_flag','customerid','creationdate','birthdate',
             'approveddate','systemloanid','longitude_gps','latitude_gps'], axis=1)
y = df['good_bad_flag']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# categorical & numerical columns
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(include=["int64","float64"]).columns

# transformers
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# build full pipeline: preprocessing → SMOTE → model
model = RandomForestClassifier()

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', model)
])

# train
pipeline.fit(X_train, y_train)

# save model
with open('loan_risk_balanced.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print(" Model trained on SMOTE-balanced data and saved as loan_risk_balanced.pkl")

# test prediction
data = {
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

sample_df = pd.DataFrame(data)

with open('loan_risk_balanced.pkl','rb') as f:
    model = pickle.load(f)

prediction = model.predict(sample_df)
print(f"Loan Risk Prediction: {prediction[0]}")
