# Import Libraries
import streamlit as st
import numpy as np
import pandas as pd

# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning App',
    layout='wide',
    page_icon="📊")

# import data function
@st.cache
def import_data(data):
    df = pd.read_excel(data, index_col='date')

    df.index = pd.to_datetime(df.index)
    columns = ['Temperature (°C)', 'Relative Humidity (%)', 'Current Q235 (nA)', 'Current Q345 (nA)', 'Current Q345-W (nA)', 'Electrical Quantity Q235 (C)']
    df.columns = columns

    df['Electrical Quantity Q345 (C)'] = df['Current Q345 (nA)'].cumsum()
    df['Electrical Quantity Q345 (C)'] = df['Electrical Quantity Q345 (C)'] * 0.00000006

    df['Electrical Quantity Q345-W (C)'] = df['Current Q345-W (nA)'].cumsum()
    df['Electrical Quantity Q345-W (C)'] = df['Electrical Quantity Q345-W (C)'] * 0.00000006

    #scale down the data where the threshold of rain (1000 nA) is reached by a factor or 0.2
    df['Current Q235 (nA)'] = np.where(df['Current Q235 (nA)'] >= 1000, df['Current Q235 (nA)']*0.2, df['Current Q235 (nA)'])
    df['Current Q345 (nA)'] = np.where(df['Current Q345 (nA)'] >= 1000, df['Current Q345 (nA)'] * 0.2, df['Current Q345 (nA)'])
    df['Current Q345-W (nA)'] = np.where(df['Current Q345-W (nA)'] >= 1000, df['Current Q345-W (nA)'] * 0.2, df['Current Q345-W (nA)'])

    return df
#---------------------------------#

st.write("""
# The Machine Learning App

In this implementation, A *LSTM deep learning network* from the Keras library is used to build a supervised machine learning regression model that is able to predict the ACM current from temperature and relative humidity input data. 

Try adjusting the hyperparameters!
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input excel file", type=["xlsx"])
    st.session_state['uploaded_file'] = uploaded_file

if 'uploaded_file' in st.session_state and st.session_state['uploaded_file'] is not None:
    st.session_state['df'] = import_data(uploaded_file)

check_timeframe = st.checkbox("Should the data be resampled to another timeframe? (might be needed to reduce computational time)")
timeframe = False
if check_timeframe:
    timeframe = st.select_slider('What timeframe should the data be resampled to?', options=['10min', '30min', 'H', '6H', 'D'])
# Sidebar - Specify parameter settings

# Main panel
# Displays the dataset
st.subheader('Glimpse of dataset')

if 'df' in st.session_state:
    # resample the data to an hourly frame for better GPU processing time
    if timeframe is not False:
        df_1 = st.session_state.df.resample(timeframe).mean()
        df_1.interpolate(inplace=True)
        st.session_state['df'] = df_1
    st.write(st.session_state.df.head())

else:
    st.info('Awaiting for excel file to be uploaded.')

