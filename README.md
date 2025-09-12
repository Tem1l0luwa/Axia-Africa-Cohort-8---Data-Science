# Loan Risk Prediction Project

This project uses machine learning to predict the likelihood of loan default based on borrower characteristics and loan features. It combines Power BI for visualization and Jupyter Notebook for model development.

## Project Structure

- Loan Prediction - Elizabeth Olatuja(cohort8).pbix: Power BI report with interactive visuals
- Elizabeth Olatuja Axia Cohort8 Project.ipynb: Jupyter Notebook with data preprocessing, modeling, and evaluation
- data: Raw and cleaned datasets
- images: Visual exports from Power BI
- README.md: Project documentation
- Medium: Project documentation(https://medium.com/@olatujatemiloluwa/predicting-loan-risk-fd69fde33e9b)
- Deployment Link (https://axia-africa-cohort-8---data-science-rcbfsacfszhf3aaausgniw.streamlit.app)

## Objectives

- Predict loan default risk using historical data
- Visualize borrower behavior and repayment trends
- Compare model performance (balanced vs. unbalanced)
- Identify high-risk segments by age, employment, education, and loan amount

## Machine Learning Models

- Logistic Regression
- Random Forest
- Decision Tree
-Gradient Boost
- Model comparison using accuracy, precision, recall, F1-score, ROC_auc_score, Variance Threshold and Feature Importance(shap)

## 📂 Datasets Used

- [Train Performance Data](https://raw.githubusercontent.com/Oyeniran20/axia_cohort_8/refs/heads/main/trainperf.csv)
- [Train Demographics Data](https://raw.githubusercontent.com/Oyeniran20/axia_cohort_8/refs/heads/main/traindemographics.csv)
- [Train Previous Loans Data](https://raw.githubusercontent.com/Oyeniran20/axia_cohort_8/refs/heads/main/trainprevloans.csv)
- pro_data_1.csv, pro_data_2.csv and pro_data_3.csv

## Power BI Insights

- Due Ratio by Loan Amount and Employment Status
- Bad Loan % by Age Group and Education Level
- Simulated Risk using Loan Amount Parameters
- Confusion Matrix and Model Accuracy Comparison
- viz (https://app.powerbi.com/groups/me/reports/0d0c25d2-1a8d-4ab4-b976-8c525199e64f/99bc6962a41185c07483?experience=power-bi)


## The Value of Balancing the Dataset

An unbalanced model may appear to perform well because it predicts the majority class correctly in most cases. However, its true weakness is exposed when identifying the minority class — in this case, risky clients who are likely to default. By applying SMOTE, the dataset is synthetically balanced, allowing the model to learn patterns from both good and bad loans more effectively. This enhances fairness, reduces bias, and increases predictive power where it matters most: identifying potential defaults before they occur.

## Deployment Process

The deployment process involved the following steps:
- Data Preparation – irrelevant columns were removed, and features were divided into numerical and categorical groups.
- Model Training – one pipeline was trained on the raw dataset (unbalanced), and another on the SMOTE-augmented dataset (balanced).
- Model Saving – trained models were serialized as .pkl files for later use.
- App Development – two Streamlit applications were built, each loading its respective model and collecting loan application details through an intuitive interface.

## Practical Implications

- For Financial Institutions: The balanced model provides a stronger safety net against loan defaults, identifying risky clients more reliably.
- For Analysts and Developers: The unbalanced model offers a baseline for comparison, illustrating the importance of addressing data imbalance.
- For End-Users: The web application makes predictions accessible in a user-friendly form, enabling real-time decision support.
