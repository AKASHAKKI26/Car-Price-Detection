import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    data=pd.read_csv("CarPrice_Assignment.csv")
    return data
data=load_data()
st.title("🚗 Car Price Detection")
st.dataframe(data.head())
data_encoded = pd.get_dummies(data[["CarName","fueltype"]], drop_first=True)
st.file_uploader("Upload your dataset",type="csv")
x = data_encoded
y=data["price"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
print(data.columns)
option1=st.selectbox("Select Car",data["CarName"].unique())
option2=st.selectbox("Select Fuel Type",data["fueltype"].unique())
user_input=pd.DataFrame({
    "CarName":[option1],
    "fueltype":[option2]
})
user_input = pd.get_dummies(user_input)
user_input = user_input.reindex(columns=x.columns, fill_value=0)
if st.button("Predict Price"):
    y_pred=model.predict(user_input)

    st.success(y_pred)
