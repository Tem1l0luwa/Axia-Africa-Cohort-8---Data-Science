# Loan Risk Prediction Project

This project uses machine learning to predict the likelihood of loan default based on borrower characteristics and loan features. It combines Power BI for visualization and Jupyter Notebook for model development.

## Project Structure

- `Loan Prediction - Elizabeth Olatuja(cohort8).pbix`: Power BI report with interactive visuals
- `Elizabeth Olatuja Axia Cohort8 Project.ipynb`: Jupyter Notebook with data preprocessing, modeling, and evaluation
- `data/`: Raw and cleaned datasets
- `images/`: Visual exports from Power BI
- `README.md`: Project documentation

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
or pro_data_1.csv, pro_data_2.csv and pro_data_3.csv

## Power BI Insights

- Due Ratio by Loan Amount and Employment Status
- Bad Loan % by Age Group and Education Level
- Simulated Risk using Loan Amount Parameters
- Confusion Matrix and Model Accuracy Comparison
