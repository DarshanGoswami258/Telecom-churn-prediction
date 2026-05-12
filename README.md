# Customer Churn Prediction – Telecom

## Project Overview

This project focuses on predicting whether a telecom customer is likely to leave the company (churn) using machine learning.

Customer churn is a major business problem because retaining existing customers is usually cheaper than acquiring new ones. By identifying customers who are likely to churn, telecom companies can take preventive actions such as offering discounts, improving support, or creating personalized retention strategies.

In this project, we performed:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Data Preprocessing
* Machine Learning Model Building
* Model Evaluation
* Model Explainability using SHAP
* Model Saving for Deployment

---

# Dataset Information

The dataset contains telecom customer information such as:

* Customer demographics
* Subscription details
* Internet services
* Payment methods
* Monthly charges
* Total charges
* Contract type
* Churn status

Target Variable:

* `Churn`

  * Yes = Customer left
  * No = Customer stayed

---

# Technologies Used

## Programming Language

* Python

## Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* SHAP
* Pickle

---

# Step-by-Step Workflow

# 1. Importing Libraries

First, we imported all the required Python libraries.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Why We Did This

These libraries help in:

* Handling data
* Performing calculations
* Creating visualizations
* Building machine learning models

## What It Improves

Makes data analysis faster, cleaner, and more efficient.

---

# 2. Loading the Dataset

```python
pd.read_csv()
```

The dataset was loaded into a Pandas DataFrame.

## Why We Did This

To access and analyze customer data.

## What It Improves

Allows structured analysis and preprocessing.

---

# 3. Understanding the Dataset

We checked:

* Shape of the dataset
* Data types
* Statistical summary
* Missing values
* Duplicate records

### Functions Used

```python
print(df.shape)
df.dtypes
df.describe()
df.isnull().sum()
df.duplicated()
```

## Why We Did This

Before building a model, it is important to understand:

* Number of rows and columns
* Numerical and categorical features
* Missing data
* Data quality issues

## What It Improves

Improves data reliability and prevents model errors.

---

# 4. Exploratory Data Analysis (EDA)

EDA was performed to identify patterns and relationships in the data.

## Visualizations Used

### Churn Distribution

```python
sns.countplot(x='Churn', data=df)
```

### Churn by Contract Type

```python
sns.countplot(x='Contract', hue='Churn', data=df)
```

### Tenure vs Churn

```python
sns.histplot(data=df, x='tenure', hue='Churn')
```

### Monthly Charges vs Churn

```python
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
```

### KDE Plots

Used to understand data distribution.

### Boxplots

Used to detect outliers.

---

## Why We Did EDA

EDA helps us understand:

* Which customers are more likely to churn
* Important business patterns
* Feature relationships
* Data distribution
* Outliers

## Insights Found

* Customers with month-to-month contracts had higher churn.
* Customers with high monthly charges were more likely to churn.
* Customers with shorter tenure showed higher churn rates.
* Long-term contract customers had lower churn.

## What It Improves

EDA improves:

* Feature selection
* Business understanding
* Model performance
* Decision-making

---

# 5. Identifying Numerical and Categorical Columns

```python
num = df.select_dtypes(include=np.number).columns.to_list()
cat = df.select_dtypes(include=object).columns.to_list()
```

## Why We Did This

Machine learning preprocessing depends on feature types.

* Numerical columns require scaling or distribution analysis.
* Categorical columns require encoding.

## What It Improves

Makes preprocessing organized and accurate.

---

# 6. Data Cleaning

## Converting TotalCharges to Numeric

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
```

## Why We Did This

`TotalCharges` was stored as a string because of blank spaces.

Machine learning models cannot work properly with incorrect data types.

## What It Improves

Ensures accurate calculations and model training.

---

## Handling Missing Values

```python
df.dropna(inplace=True)
```

Only a few rows had missing values.

## Why We Did This

Missing values can reduce model accuracy.

## What It Improves

Creates cleaner and more reliable training data.

---

## Dropping Customer ID

```python
df.drop('customerID', axis=1, inplace=True)
```

## Why We Did This

Customer ID does not help predict churn.

It is just a unique identifier.

## What It Improves

Reduces unnecessary information and prevents model confusion.

---

# 7. Target Variable Encoding

```python
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
```

## Why We Did This

Machine learning models require numerical target values.

## What It Improves

Allows classification algorithms to process the target variable.

---

# 8. Encoding Categorical Features

## Label Encoding

Used for binary columns.

```python
LabelEncoder()
```

## One-Hot Encoding

Used for columns with multiple categories.

## Why We Did This

Machine learning models cannot directly understand text values.

Encoding converts categories into numerical format.

## What It Improves

Improves model compatibility and prediction accuracy.

---

# 9. Splitting the Dataset

```python
train_test_split()
```

Dataset split:

* 80% Training Data
* 20% Testing Data

## Why We Did This

* Training data teaches the model.
* Testing data evaluates model performance.

## What It Improves

Helps measure real-world model performance.

---

# 10. Building the Machine Learning Model

## Algorithm Used

### XGBoost Classifier

```python
XGBClassifier()
```

Parameters used:

* n_estimators = 200
* max_depth = 4
* learning_rate = 0.05

---

## Why We Chose XGBoost

XGBoost is powerful for classification problems because:

* Handles complex relationships
* Works well with structured data
* Reduces overfitting
* Gives high accuracy
* Handles feature interactions effectively

## What It Improves

Provides better predictive performance compared to many traditional algorithms.

---

# 11. Model Evaluation

Metrics used:

```python
accuracy_score()
roc_auc_score()
classification_report()
```

## Why We Evaluated the Model

To measure:

* Prediction accuracy
* Classification quality
* Model reliability

## Results

* Accuracy achieved was around 87%
* Good classification performance for churn prediction

## What It Improves

Ensures the model performs well before deployment.

---

# 12. Model Explainability using SHAP

```python
shap.TreeExplainer(model)
```

SHAP visualizations were used to identify:

* Most important features
* Feature impact on predictions
* Positive and negative influence on churn

---

## Why We Used SHAP

Machine learning models can behave like black boxes.

SHAP helps explain:

* Why predictions happen
* Which features influence churn the most

## What It Improves

Improves:

* Transparency
* Business trust
* Decision-making
* Model interpretability

---

# 13. Saving the Model

```python
pickle.dump()
```

Saved files:

* `churn_model.pkl`
* `feature_names.pkl`

## Why We Did This

To reuse the trained model later without retraining.

## What It Improves

Makes deployment faster and more efficient.

---

# Key Business Insights

* Month-to-month customers are more likely to churn.
* High monthly charges increase churn probability.
* Customers with longer tenure are more loyal.
* Contract type strongly affects retention.
* Customer support services influence churn behavior.

---

# Project Outcome

Successfully built a machine learning model capable of predicting telecom customer churn with strong accuracy.

The project demonstrates:

* End-to-end machine learning workflow
* Data preprocessing skills
* Exploratory data analysis
* Feature engineering
* Predictive modeling
* Model evaluation
* Explainable AI

---

# Future Improvements

Possible future enhancements:

* Hyperparameter tuning
* Cross-validation
* Deployment using Streamlit or Flask
* Real-time churn prediction dashboard
* Advanced feature engineering
* Comparing multiple algorithms

---

# Conclusion

This project helps telecom companies identify customers who are likely to leave.

Using machine learning, businesses can take proactive actions to improve customer retention, reduce revenue loss, and improve customer satisfaction.

The project also demonstrates practical data science skills used in real-world business problems.
