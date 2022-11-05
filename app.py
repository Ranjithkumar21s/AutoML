import os
import streamlit as st
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup,compare_models,pull,save_model


with st.sidebar:
    st.image('https://developer.apple.com/assets/elements/icons/create-ml/create-ml-96x96_2x.png')
    st.title('Automated Meachine Learning Application')
    choice = st.radio('Navigator',['Upload','Data Proecessing','Machine Learning','Download'])
    st.text('Welcome to AutoML')

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv",index_col=None)

if choice == 'Upload':
    st.title("Upload your File")
    file = st.file_uploader("Upload Your CSV File Here!")
    if file:
        df = pd.read_csv(file,index_col=None)
        df.to_csv("sourcedata.csv",index=None)
        st.dataframe(df)

if choice == 'Data Proecessing':
    st.title("Auto Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == 'Machine Learning':
    st.title("Machine Learning!!!!")
    target = st.selectbox("Select The Target Variable",df.columns)
    if st.button("Train Models"):
        setup(df,target=target)
        setup_df = pull()
        st.info("ML Model setting")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("ML Result:")
        st.dataframe(compare_df)
        best_model
        save_model(best_model,'best_model')
if choice == 'Download':
    with ("best_model.pkl",'rb') as f:
        st.download_button("Download the file!",f,"best_model.pkl")
