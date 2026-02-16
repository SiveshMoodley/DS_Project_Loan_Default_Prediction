# Bank Loan Default Prediction

## Project Overview
- Developed a machine learning solution to predict the likelihood of loan default using borrower financial and demographic attributes
- Engineered predictive features through categorical encoding, class imbalance handling (SMOTE), and feature scaling to optimse model convergence and address data skewness
- Trained and optimised multiple classification algorithms using cross-validation and hyperparameter tuning to identify the best-performing risk model

## Code and Resources
Python Version: 3.10

Packages: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost

Python Requirements: pip install -r requirements.txt

## Dataset
Source: MIT Professional Education

Type: Educational / Synthetic Dataset

The dataset represents historical loan applications from a financial institution, simulating real-world credit approval scenarios similar to those used by retail banks and lending platforms. The dataset contains borrower and loan information including:
- Loan default or loan repaid (BAD)
- Loan amount approved (LOAN)
- Amount due on existing mortgage (MORTDUE)
- Current value of property (VALUE)
- Reason for loan request (REASON)
- Occupation (JOB)
- Years at present job (YOJ)
- No. of major derogatory loan reports (DEROG)
- No. of delinquent credit lines (DELINQ)
- Age of the oldest credit line in months (CLAGE)
- Number of recent credit inquiries (NINQ)
- Number of existing credit lines (CLNO)
- Debt-to-income ratio (DEBTINC)

The data is intended for educational and model development rather than commercial credit decisions. No personal or real customer information is included.

## Data Preprocessing
The data was cleaned such that it was usable for the model. I conducted the following process:
- Checked the data types of the columns in the dataset
- Checked the data for duplicate rows and missing values
- The dataset has no duplicate rows but has missing values
- Input of median for missing values approach was used as robust to outliers and skewed distributions 

The data was prepared and split into training and test sets, with a 20% test size. Numerical features were scaled where required and categorical variables were encoded prior to modeling.

## EDA
Conducted exploratory analysis to examine the distributions of numerical variables and the frequency of categorical features. Key observations include:
- 20% imbalance in the target variable with 20% of clients defaulting
- Default risk is most strongly associated with credit behaviour and affordability indicators, notably DELINQ, DEROG, DEBTINC, and NINQ (with older credit history CLAGE reducing default likelihood)
- Certain applicant segments show higher default rates, particularly Sales / Self-employed roles and Home Improvement (HomeImp) loan purposes, suggesting useful risk segmentation.
- Several loan/asset variables (e.g., LOAN, MORTDUE, VALUE) show statistically significant differences between defaulters and non-defaulters, indicating meaningful separation for modeling.
  
## Model Building
Multiple classification models were trained and evaluated:
- Logistic Regression – serves as a linear baseline to assess separability of classes under linear decision boundaries
- Decision Tree – captures non-linear relationships and interaction effects without requiring feature transformation
- Random Forest – reduces variance and overfitting associated with single decision trees through bagging
- Gradient Boosting - sequentially corrects residual errors, improving performance on complex feature interactions
- XGBoost – optimised boosting framework designed to enhance generalisation and handle structured tabular data effectively

To address class imbalance, SMOTE was applied to improve minority class representation during training. Hyperparameter tuning was conducted using cross-validation prior to evaluation on the test set.

## Model Performance
Models were evaluated using Accuracy, ROC-AUC, Precision, and Recall, with particular focus on recall for defaulters.

- Logistic Regression performed poorly in identifying defaulters (Recall: 11%), making it unsuitable for high-stakes decisions
- Decision Tree improved recall (60%) but showed moderate ROC-AUC (0.84)
- Random Forest achieved strong ROC-AUC (0.93) but lower recall for defaulters (44%)
- XGBoost with SMOTE obtained reasonable rounded performance (ROC-AUC: 0.89, Recall: 76%)
- Gradient Boosting with SMOTE delivered the best balance, achieving Recall: 77%, Precision: 60%, Accuracy: 85%, ROC-AUC: 0.90, making it the recommended model

## Evaluation
- Identified key predictors of default risk, including DEBTINC, DELINQ, NINQ, DEROG, and YOJ, reinforcing the importance of borrower credit behavior and financial stability in risk assessment
- Addressed dataset class imbalance using SMOTE, significantly improving defaulter recall (77%) while accepting lower precision (60%) — an appropriate trade-off in high-risk lending contexts
- Determined Gradient Boosting with SMOTE as the optimal model, providing a strong balance between recall, precision, and ROC-AUC for decision-support use
- Highlighted limitations including dataset imbalance and potential data drift, underscoring the need for periodic retraining and fairness evaluation
- Recommended future enhancements including incorporation of dynamic financial indicators and exploration of cost-sensitive learning
