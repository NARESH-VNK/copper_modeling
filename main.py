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


    col1, col2, col3,col4,col5,col6,col7,col8 = st.columns(8)
    with col1:
        quantity_tons = st.number_input("Enter the Quantity in Tons")
    with col2:
        country = st.selectbox("Choose a country",[28.0,25.0, 30.0, 32.0, 38.0, 78.0, 27.0, 77.0, 113.0, 79.0, 26.0, 39.0, 40.0, 84.0, 80.0, 89.0, 107.0])
    with col3:
        item_type = st.selectbox("Choose a item-type \n '5.0 for W', \n '6.0 for WI', \n '3.0 for S', \n '1.0 for Others',\n  '2.0 for PL', \n '0.0 for IPL', \n '4.0 for SLAWR'",[0.0,1.0,2.0,3.0,4.0,5.0,6.0] )
    with col4:
        application = st.selectbox("choose a application",[10.0, 41.0, 28.0, 59.0, 15.0, 4.0, 38.0, 56.0, 42.0, 26.0, 27.0, 19.0, 20.0, 66.0, 29.0, 22.0, 25.0, 40.0, 79.0, 3.0, 99.0, 2.0, 67.0, 5.0, 39.0, 69.0, 70.0, 65.0, 58.0, 68.0])
    with col5:
        thickness = st.number_input("Enter the Thickness")
    with col6:
        width = st.number_input("Enter the Width")
    with col7:
        product_ref = st.selectbox("Choose Product_ref",[1670798778, 1668701718, 628377, 640665, 611993,1668701376, 164141591, 1671863738, 1332077137, 640405, 1693867550, 1665572374, 1282007633, 1668701698, 628117, 1690738206, 628112, 640400, 1671876026, 164336407, 1665572032, 164337175, 611728, 1721130331, 1693867563, 611733, 1690738219, 1722207579, 929423819, 1668701725, 1665584320, 1665584642])
    with col8:
        selling_price = st.number_input("Enter the Selling Price")



    fitting = st.checkbox('Fit into Model')

    if fitting:
        st.write('Great!')
        # Make predictions on the new test data
        X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        #  Random Forest classifier with best parameters
        model = RandomForestClassifier(n_estimators=200, random_state=42, min_samples_split=2, max_depth=None)
        result_rf1 = model.fit(X_train, y_train)

        newtestdata1 = [quantity_tons, country, item_type, application, thickness, width, product_ref, selling_price]
        new_test_data1 = np.array([newtestdata1])

        new_test_data_scale1 = scaler.transform(new_test_data1)
        new_predicted = result_rf1.predict(new_test_data_scale1)


        if st.button("Predict"):

            if new_predicted == 0:
                st.markdown("The Predicted Outcome is 0 ")
                st.markdown("OOPS!!! ,That's LOST, They are not likely to be a Customer!!!")
            else:
                st.markdown("The Predicted Outcome is 1")
                st.markdown("Hurry!!! ,That's WON, They are likely to be a Customer!!!")
#
# scaler = StandardScaler()
# new_test_data = np.array([newtestdata])
# scaler.fit(X_train)
# X_train= scaler.transform(X_train)
# new_test_data_scale = scaler.transform(new_test_data)
#
# #  Random Forest classifier with best parameters
# model = RandomForestClassifier(n_estimators=200, random_state=42,min_samples_split = 2,max_depth= None)
# result_rf = model.fit(X_train, y_train)
#
# new_predicted= result_rf.predict(new_test_data_scale)


if options == "About":
    st.title("Copper Modelling")


if options == "Selling Price Prediction":
    col1, col2, col3,col4,col5,col6,col7,col8= st.columns(8)

    with col1:
        quantity_tons = st.number_input("Enter the Quantity in Tons")
    with col2:
        country = st.selectbox("Choose a country",[28.0,25.0, 30.0, 32.0, 38.0, 78.0, 27.0, 77.0, 113.0, 79.0, 26.0, 39.0, 40.0, 84.0, 80.0, 89.0, 107.0])
    with col3:
        status = st.number_input("Enter the Status (0-lost),(1-Won)")
    with col4:
        item_type = st.selectbox("Choose a item-type \n '5.0 for W', \n '6.0 for WI', \n '3.0 for S', \n '1.0 for Others',\n  '2.0 for PL', \n '0.0 for IPL', \n '4.0 for SLAWR'",[0.0,1.0,2.0,3.0,4.0,5.0,6.0] )
    with col5:
        application = st.selectbox("choose a application",[10.0, 41.0, 28.0, 59.0, 15.0, 4.0, 38.0, 56.0, 42.0, 26.0, 27.0, 19.0, 20.0, 66.0, 29.0, 22.0, 25.0, 40.0, 79.0, 3.0, 99.0, 2.0, 67.0, 5.0, 39.0, 69.0, 70.0, 65.0, 58.0, 68.0] )
    with col6:
        thickness = st.number_input("Enter the Thickness")
    with col7:
        width = st.number_input("Enter the Width")
    with col8:
        product_ref = st.selectbox("Choose Product_ref",[1670798778, 1668701718, 628377, 640665, 611993,1668701376, 164141591, 1671863738, 1332077137, 640405, 1693867550, 1665572374, 1282007633, 1668701698, 628117, 1690738206, 628112, 640400, 1671876026, 164336407, 1665572032, 164337175, 611728, 1721130331, 1693867563, 611733, 1690738219, 1722207579, 929423819, 1668701725, 1665584320, 1665584642])

    newtestdata_reg = [quantity_tons, country, status,item_type, application, thickness, width, product_ref]

    fitting = st.checkbox('Fit into Model')

    if fitting:
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(Xr, yr, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        scaler.fit(X_train_reg)
        X_train_reg = scaler.transform(X_train_reg)

        # Create a Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1, n_jobs=-1, random_state=42)
        # Fit the model to the training data
        model.fit(X_train_reg, y_train_reg)

        new_test_data_reg = np.array([newtestdata_reg])
        X_train_reg = scaler.transform(X_train_reg)
        new_test_data_scale = scaler.transform(new_test_data_reg)

        #  Random Forest classifier with best parameters
        model = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,n_jobs=-1,random_state=42)

        result_rf = model.fit(X_train_reg, y_train_reg)

        if st.button("Click to Predict"):
            new_predicted_reg = result_rf.predict(new_test_data_scale)
            st.write("Predicted Selling price of Copper : ",new_predicted_reg)

