import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# make containers
header=st.container()
datasets=st.container()
features =st.container()
model_training=st.container()

with header:
    st.title("kashti ki app")
    st.text("In this project we will work on kashti data")

with datasets:
    st.header("Kashti doob gai")
    st.text("In this project we will work on Titanic data set")
    df=sns.load_dataset("titanic")
    df=df.dropna()
    df.head(10)
    st.subheader("How much persons were there")
    st.bar_chart(df["sex"].value_counts())

    #other chart
    st.subheader("class k hisab sy farq")
    st.bar_chart(df["class"].value_counts())
    #barplot
    st.bar_chart(df["age"].sample(10)) #or head(10)


with features:
    st.header("There is your app")
    st.text("we will add quite a lot features")
    st.markdown("1. **Feature 01:** This will tell us about any thing")
    st.markdown("2. **Feature 02:** This will tell us about further info")


with model_training:
    st.header("Kashti walon ka kia bana? model training")
    # making columns
    input,display=st.columns(2)
    # first column we will have selection points
    max_depth=input.slider("How many peaople do you know",min_value=10,max_value=100,value=20,step=5)

#n_estimators
n_estimator=input.selectbox("how many tree should be in RF",options=[50,100,150,200,"NO LIMIT"])

# list of features
input.write(df.columns)
#input features from the users
input_features=input.text_input("which feature should be used?")


#machine learning model
model=RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimator)

#by applying condition on number of trees
if n_estimator=="NO LIMIT":
    model=RandomForestRegressor(max_depth=max_depth)
else:
    model=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimator)
    
X=df[[input_features]]
y=df[["fare"]]

#fit our model
model.fit(X,y)
pred=model.predict(y)


#display metrices
display.subheader("Mean Absolute error of the model is: ")
display.write(mean_absolute_error(y,pred))
display.subheader("Mean square error of the model is: ")
display.write(mean_squared_error(y,pred))
display.subheader("R2 of the model is: ")
display.write(r2_score(y,pred))