import streamlit as st
from streamlit_pandas_profiling import st_profile_report

st.write("""
# Exploratory data anlysis via pandas profiling

Pandas profiling gives gives you a quick and easy overview of the data. Missing values, correlations, value distributions etc. are all tools that pandas profiling covers.

""")
if 'df' not in st.session_state:
    st.info('Go back to Home page and upload a dataset')
    st.stop()
else:
    df = st.session_state.df
    generate_EDA = st.button("Generate pandas profiling report!")
    if generate_EDA:
        pr = df.profile_report()
        st_profile_report(pr)

