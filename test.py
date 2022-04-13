import streamlit as st
import seaborn as sns
st.header("How are you")
st.text("How's Ramadan going")
st.header("So how's the job going in scroching sun")

df=sns.load_dataset("iris")
st.write(df[["species","sepal_length","petal_length"]].head(10))
st.bar_chart(df["sepal_length"])
st.line_chart(df["sepal_length"])
