# Bank Loan Default Prediction

## Project Overview
- Developed a machine learning solution to predict the likelihood of loan default using borrower financial and demographic attributes
- Engineered predictive features through categorical encoding, class imbalance handling (SMOTE), and feature scaling to optimse model convergence and address data skewness
- Trained and optimised multiple classification algorithms using cross-validation and hyperparameter tuning to identify the best-performing risk model

## Code and Resources
Python Version: 3.10

Packages: pandas, numpy, matplotlib, seaborn, scikit-learn, imblearn, xgboost

Python Requirements: pip install -r requirements.txt

## Dataset
Source: MIT Professional Education

Type: Educational / Synthetic Dataset

The dataset represents historical loan applications from a financial institution, simulating real-world credit approval scenarios similar to those used by retail banks and lending platforms. The dataset contains borrower and loan information including:
- Loan default or loan repaid
- Loan amount approved
- Amount due on existing mortgage
- Current value of property
- Reason for loan request
- Occupation
- Years at present job
- No. of major derogatory loan reports
- No. of delinquent credit lines
- Age of the oldest credit line in months
- Number of recent credit inquiries
- Number of existing credit lines
- Debt-to-income ratio

The data is intended for educational and model development rather than commercial credit decisions. No personal or real customer information is included.

## Data Preprocessing
The data was cleaned such that it was usable for the model. I conducted the following process:
- Checked the data types of the columns in the dataset
- Checked the data for duplicate rows and missing values
- The dataset has no duplicate rows but has missing values
- Input of median for missing values approach was used as robust to outliers and skewed distributions 

## EDA


## Model Building


## Model Performance


## Evaluation
