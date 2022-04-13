#import libraries
import streamlit as st
import plotly.express as px
import pandas as pd

#import datasets
st.title("plotly and streamlit ko milla k app bnana")
df=px.data.gapminder()
st.write(df)
#st.write(df.head())
#suumary stats
st.write(df.describe())

#data management
year_option=df["year"].unique().tolist()
year=st.selectbox("which year should we plot?",year_option,0)
df=df[df["year"]==year]

#plotting
fig=px.scatter(df,x="gdpPercap",y="lifeExp",size="pop",color="country",hover_name="country",
log_x=True,size_max=55,range_x=[100,100000],range_y=[20,90])
st.write(fig)

fig=px.scatter(df,x="gdpPercap",y="lifeExp",size="pop",color="continent",hover_name="continent",
log_x=True,size_max=55,range_x=[100,100000],range_y=[20,90])
st.write(fig)


