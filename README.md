
# Industrial Copper Modeling

# Description

- Copper, being a crucial material in various manufacturing sectors, requires accurate sales predictions and pricing decisions to optimize revenue and profitability.

- The "Industrial Copper Modeling" project focuses on applying machine learning techniques to improve decision-making processes in the copper industry.


- Additionally, the project will develop a lead classification model to evaluate and classify leads based on their likelihood to become customers. 
# Aim

## Sales and Pricing Prediction:
- The first aspect of the project involves building a machine learning regression model to predict the selling price of copper based on various factors such as quantity, location, demand, and market trends.
-  The goal is to create a model that can accurately forecast copper prices, enabling businesses to make data-driven pricing decisions.

## Lead Classification:
- The second aspect of the project revolves around developing a machine learning classification model to evaluate and classify leads based on their potential to become customers. 
- By analyzing historical data and lead characteristics, the model will predict the likelihood of a lead resulting in a successful sale (WON) or an unsuccessful one (LOST).



# Dataset

The dataset provided for the project contains historical sales and lead data, including various attributes such as:

- Selling_Price: The target variable representing the price at which copper is sold.
- Itemdate and Delivery date : It says the date of item ordered and delivered.
- Quantity: The quantity of copper in the sale.
- Item type: It refers the different Copper Items. 
- Thickness and Width : The copper's width and thickness are given in the dataset.
- Material_Reference: A reference code for the type of copper material.
- Status: The outcome of the lead, whether WON or LOST.
- Other relevant features: Additional data related to customers, Country, and market conditions.
## Installation

Install my-project with npm

```bash
    import streamlit as st
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
    from imblearn.combine import SMOTETomek
    import pickle
```
# Approach
1. Data Understanding:

- Identify the types of variables (continuous, categorical) and their distributions.
- Convert rubbish values in 'Material_Reference' starting with '00000' into null.
- Treat reference columns as categorical variables.
- Consider dropping the 'INDEX' column as it may not be useful.

2. Data Preprocessing:

- Handle missing values using mean/median/mode.
- Treat outliers using IQR or Isolation Forest from the sklearn library.
- Identify skewness in the dataset and treat it with appropriate data transformations, such as log transformation or boxcox transformation, to handle high skewness in continuous variables.
- Encode categorical variables using suitable techniques, such as one-hot encoding, label encoding, or ordinal encoding.

3. Exploratory Data Analysis (EDA):

- Visualize outliers and skewness before and after treatment using Seaborn's boxplot, distplot, and violinplot.
4. Feature Engineering:

- Engineer new features if applicable, such as aggregating or transforming existing features to create more informative representations of the data.
- Drop highly correlated columns using a Seaborn heatmap.
5. Model Building and Evaluation:

- Split the dataset into training and testing/validation sets.
- Train and evaluate different regression models, such as ExtraTreesRegressor or XGBRegressor, using appropriate evaluation metrics.
- Train and evaluate different classification models, such as ExtraTreesClassifier, XGBClassifier, or Logistic Regression, using appropriate evaluation metrics such as accuracy, precision, recall, F1 score, and AUC curve.
- Optimize model hyperparameters using techniques such as cross-validation and grid search to find the best-performing model.
6. Model GUI (Streamlit):

- Create an interactive web page using Streamlit.
- Include task input (Regression or Classification).
- Provide input fields for each column value except 'Selling_Price' for the regression model and except 'Status' for the classification model.
- Perform the same feature engineering, scaling, and transformation steps used for training the ML models.
- Display the output with the predicted Selling_Price or Status (Won/Lost).



# Learning Outcomes

By completing this project, we can learn:

1. Develop proficiency in Python programming and its data analysis libraries.

2. Gain experience in data preprocessing techniques and handling missing values, outliers, and skewness
Understand and visualize data using EDA techniques.
Learn and apply advanced machine learning techniques for regression and classification tasks.

3. Optimize machine learning models using appropriate evaluation metrics and techniques.

4. Practice feature engineering to improve model performance.

5. Build a Streamlit web application to showcase machine learning models and make predictions on new data.Understand the challenges and best practices in the manufacturing domain and how machine learning can help solve them.

## Outcomes

- By the end of the project, a fully functional machine learning model and Streamlit web application is available.
-  The web application will provide valuable insights and predictions to users, allowing them to make informed decisions regarding copper pricing and lead prioritization.
