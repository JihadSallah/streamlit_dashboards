#import libraries
from cProfile import Profile
from distutils.command.upload import upload
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
#webapp ka title
st.markdown(''' 
# **Exploratory Data Analysis web Application**
This app is developed by codanics youtube channel called "EDA App"
''')
#how to upload file from PC
with st.sidebar.header("upload your dataset(.csv)"):
    uploaded_file=st.sidebar.file_uploader("upload your file",type=["csv"])
    df=sns.load_dataset("titanic")
    st.sidebar.markdown("[Eample CSV file](https://raw.githubusercontent.com/AammarTufail/machinelearning_ka_chilla/master/Sastaticket_datasets/sastaticket_train.csv)")
#profiling report for pandas
if uploaded_file is not None:
    @st.cache
    def load_csv():  
        csv=pd.read_csv(uploaded_file)
        return csv
    df=load_csv()
    pr=ProfileReport(df,explorative=True)
    st.header("**Input DF**")
    st.write(df)
    st.write('...')
    st.header("**Profiling Report with Pandas**")
    st_profile_report(pr)
else:
    st.info("Awaiting for CSV file")
    if st.button("Press to use example data"):
        # example dataset
        @st.cache
        def load_data():
            a=pd.DataFrame(np.random.rand(100,5),
            columns=["age","banana","codanics","dutchland","Ear"])
            return a
        df=load_data()
        pr=ProfileReport(df,explorative=True)
        st.header("**Input DF**")
        st.write(df)
        st.write('...')
        st.header("**Profiling Report with Pandas**")
        st_profile_report(pr)



