import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from imblearn.combine import SMOTETomek
import pickle


# Unpickle the data :
with open('regdata.pl', 'rb') as file:
    df_class = pickle.load(file)

 # Displaying the data

df2 = df_class.copy()


Xr = df2.drop(['selling_price',"customer"], axis=1)
yr = df2['selling_price']



# CLASSIFICATION PART
X = df_class.drop(['status',"customer"], axis=1)
y = df_class['status']
model = SMOTETomek()  # SAMPLING IS DONE
X1, y1 = model.fit_resample(X, y)







options = st.sidebar.selectbox('Menu',['About',"WorkFlow","Customer Prediction","Selling Price Prediction","Outcomes"])

if options == "Customer Prediction":
    with st.form("Enter the Input Features"):
        quantity_tons = st.number_input("Enter the Quantity in Tons")
        country = st.selectbox("Choose a country",[28.0, 25.0, 30.0, 32.0, 38.0, 78.0, 27.0, 77.0, 113.0, 79.0, 26.0, 39.0, 40.0, 84.0,80.0, 89.0, 107.0])
        item_type = st.selectbox("Choose a item-type \n '5.0 for W', \n '6.0 for WI', \n '3.0 for S', \n '1.0 for Others',\n  '2.0 for PL', \n '0.0 for IPL', \n '4.0 for SLAWR'",[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        application = st.selectbox("choose a application",
                                   [10.0, 41.0, 28.0, 59.0, 15.0, 4.0, 38.0, 56.0, 42.0, 26.0, 27.0, 19.0, 20.0, 66.0,
                                    29.0, 22.0, 25.0, 40.0, 79.0, 3.0, 99.0, 2.0, 67.0, 5.0, 39.0, 69.0, 70.0, 65.0,
                                    58.0, 68.0])
        thickness = st.number_input("Enter the Thickness")
        width = st.number_input("Enter the Width")
        product_ref = st.selectbox("Choose Product_ref",
                                   [1670798778, 1668701718, 628377, 640665, 611993, 1668701376, 164141591, 1671863738,
                                    1332077137, 640405, 1693867550, 1665572374, 1282007633, 1668701698, 628117,
                                    1690738206, 628112, 640400, 1671876026, 164336407, 1665572032, 164337175, 611728,
                                    1721130331, 1693867563, 611733, 1690738219, 1722207579, 929423819, 1668701725,
                                    1665584320, 1665584642])
        selling_price = st.number_input("Enter the Selling Price")
        newtestdata1 = [quantity_tons, country, item_type, application, thickness, width, product_ref, selling_price]

        # Add the form_submit_button widget to submit the form
        submitted = st.form_submit_button("Submit")

    # Display the collected inputs when the form is submitted
    if submitted:

        st.write('Great!,Model is Under Training....')

        # Make predictions on the new test data
        X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        #  Random Forest classifier with best parameters
        model = RandomForestClassifier(n_estimators=200, random_state=42, min_samples_split=2, max_depth=None)
        result_rf1 = model.fit(X_train, y_train)

        new_test_data1 = np.array([newtestdata1])

        new_test_data_scale1 = scaler.transform(new_test_data1)
        new_predicted = result_rf1.predict(new_test_data_scale1)

        st.subheader("Prediction Completed...")

        if new_predicted == 0:
            st.markdown("The Predicted Outcome is 0 ")
            st.markdown("OOPS!!! ,That's LOST, They are not likely to be a Customer!!!")
        else:
            st.markdown("The Predicted Outcome is 1")
            st.markdown("Hurry!!! ,That's WON, They are likely to be a Customer!!!")






if options == "About":
    st.title("Industrial Copper Modelling")
    st.subheader("Description")
    st.markdown("- Copper, being a crucial material in various manufacturing sectors, requires accurate sales predictions and pricing decisions to optimize revenue and profitability.")
    st.markdown("- The Industrial Copper Modeling project focuses on applying machine learning techniques to improve decision-making processes in the copper industry.")
    st.markdown("- Additionally, the project will develop a lead classification model to evaluate and classify leads based on their likelihood to become customers.")
    st.subheader("Aim Of This Project")
    st.markdown("#### Sales and Pricing Prediction")
    st.markdown("- The first aspect of the project involves building a machine learning regression model to predict the selling price of copper based on various factors such as quantity, location, demand, and market trends.")
    st.markdown("- The goal is to create a model that can accurately forecast copper prices, enabling businesses to make data-driven pricing decisions.")
    st.markdown("#### Lead Classification")
    st.markdown("- The second aspect of the project revolves around developing a machine learning classification model to evaluate and classify leads based on their potential to become customers.")
    st.markdown("- By analyzing historical data and lead characteristics, the model will predict the likelihood of a lead resulting in a successful sale (WON) or an unsuccessful one (LOST).")

if options =="WorkFlow":
    st.header("Process Flow")


    st.subheader("Data Understanding")
    st.markdown("- Identify the types of variables (continuous, categorical) and their distributions. Convert rubbish values in 'Material_Reference' starting with '00000' into null.")
    st.markdown("- Treat reference columns as categorical variables.Consider dropping the 'INDEX' column as it may not be useful.")
    st.subheader("Data Preprocessing")
    st.markdown("- Handle missing values using mean/median/mode.Treat outliers using IQR or Isolation Forest from the sklearn library.")
    st.markdown("- Identify skewness in the dataset and treat it with appropriate data transformations, such as log transformation or boxcox transformation, to handle high skewness in continuous variables.Encode categorical variables using suitable techniques, such as one-hot encoding, label encoding, or ordinal encoding.")
    st.subheader("Exploratory Data Analysis (EDA)")
    st.markdown("- Visualize outliers and skewness before and after treatment using Seaborn's boxplot, distplot, and histplot.")
    st.subheader("Feature Engineering")
    st.markdown("- Engineer new features if applicable, such as aggregating or transforming existing features to create more informative representations of the data.Drop highly correlated columns using a Seaborn heatmap.")
    st.subheader("Model Building and Evaluation")
    st.markdown("- Split the dataset into training and testing/validation sets.Train and evaluate different regression models, such as ExtraTreesRegressor or XGBRegressor, using appropriate evaluation metrics.")
    st.markdown("- Train and evaluate different classification models, such as ExtraTreesClassifier, XGBClassifier, or Logistic Regression, using appropriate evaluation metrics such as accuracy, precision, recall, F1 score, and AUC curve.")
    st.markdown("- Optimize model hyperparameters using techniques such as cross-validation and grid search to find the best-performing model.")
    st.subheader("Model GUI (Streamlit)")
    st.markdown("- Create an interactive web page using Streamlit.")
    st.markdown("- Include task input (Regression or Classification). Provide input fields for each column value except 'Selling_Price' for the regression model and except 'Status' for the classification model.")
    st.markdown("- Perform the same feature engineering, scaling, and transformation steps used for training the ML models.")
    st.markdown("- Display the output with the predicted Selling_Price or Status (Won/Lost).")


if options == "Selling Price Prediction":
    with st.form("Enter the Input Features"):
        quantity_tons = st.number_input("Enter the Quantity in Tons")
        country = st.selectbox("Choose a country",
                               [28.0, 25.0, 30.0, 32.0, 38.0, 78.0, 27.0, 77.0, 113.0, 79.0, 26.0, 39.0, 40.0, 84.0,
                                80.0, 89.0, 107.0])
        status = st.number_input("Enter the Status (0-lost),(1-Won)")
        item_type = st.selectbox(
            "Choose a item-type \n '5.0 for W', \n '6.0 for WI', \n '3.0 for S', \n '1.0 for Others',\n  '2.0 for PL', \n '0.0 for IPL', \n '4.0 for SLAWR'",
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        application = st.selectbox("choose a application",
                                   [10.0, 41.0, 28.0, 59.0, 15.0, 4.0, 38.0, 56.0, 42.0, 26.0, 27.0, 19.0, 20.0, 66.0,
                                    29.0, 22.0, 25.0, 40.0, 79.0, 3.0, 99.0, 2.0, 67.0, 5.0, 39.0, 69.0, 70.0, 65.0,
                                    58.0, 68.0])
        thickness = st.number_input("Enter the Thickness")
        width = st.number_input("Enter the Width")
        product_ref = st.selectbox("Choose Product_ref",
                                   [1670798778, 1668701718, 628377, 640665, 611993, 1668701376, 164141591, 1671863738,
                                    1332077137, 640405, 1693867550, 1665572374, 1282007633, 1668701698, 628117,
                                    1690738206, 628112, 640400, 1671876026, 164336407, 1665572032, 164337175, 611728,
                                    1721130331, 1693867563, 611733, 1690738219, 1722207579, 929423819, 1668701725,
                                    1665584320, 1665584642])

        newtestdata_reg = [quantity_tons, country, status, item_type, application, thickness, width, product_ref]

        # Add the form_submit_button widget to submit the form
        submitted = st.form_submit_button("Submit")

        # Display the collected inputs when the form is submitted
    if submitted:
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(Xr, yr, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        scaler.fit(X_train_reg)
        X_train_reg = scaler.transform(X_train_reg)

        # Create a Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1, n_jobs=-1, random_state=42)
        # Fit the model to the training data
        result_rf = model.fit(X_train_reg, y_train_reg)

        new_test_data_reg = np.array([newtestdata_reg])
        new_test_data_scale = scaler.transform(new_test_data_reg)
        result_rf = model.fit(X_train_reg, y_train_reg)


        new_predicted_reg = result_rf.predict(new_test_data_scale)
        st.subheader("Prediction from New Data")
        st.write("Predicted Selling price of Copper : ", new_predicted_reg)

if options =="Outcomes":
    st.header("Outcomes  Of This Projects")
    st.markdown("- By the end of the project,a fully functional machine learning model and Streamlit web application is available.")
    st.markdown(" - The web application will provide valuable insights and predictions to users,allowing them to make informed decisions regarding copper pricing and lead prioritization.")
