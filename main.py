
# Import Libraries
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from numpy import array
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
# from keras.layers import TimeDistributed
# from keras.layers.convolutional import Conv1D
# from keras.layers.convolutional import MaxPooling1D
# from keras.layers import ConvLSTM2D
# from keras.callbacks import EarlyStopping
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import deeplearning

models = ('Vanilla LSTM', 'Stacked LSTM', 'Bidirectional LSTM')
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning App',
    layout='wide',
    page_icon="📊")

# import data function
@st.cache
def import_data(data, timeframe):
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


    # resample the data to an hourly frame for better GPU processing time
    if timeframe is not False:
        df = df.resample(timeframe).mean()
        df.interpolate(inplace = True)
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

# Sidebar - Specify the deep learning lSTM model
with st.sidebar.header('2. Select model'):
    model_selection = st.sidebar.selectbox('Select the deep learning model of choice', models)

check_timeframe = st.checkbox("Should the data be resampled to another timeframe? (might be needed to reduce computational time)")
timeframe = False
if check_timeframe:
    timeframe = st.select_slider('What timeframe should the data be resampled to?', options=['10min', '30min', 'H', '6H', 'D'])
# Sidebar - Specify parameter settings
with st.sidebar.header('3. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    if uploaded_file is not None:
        df = import_data(uploaded_file, timeframe)
        columns = df.columns
        feature_columns = st.sidebar.multiselect('Select the input features', columns)
        columns_1 = columns.drop(feature_columns)
        target_column = st.sidebar.selectbox('Select the target variable', columns_1)
        # columns_2 = columns_1.drop(target_column)
        # index_column = st.sidebar.selectbox("Select the index column (if time series the dat column)", columns_2)

with st.sidebar.subheader('3.1. Learning Parameters'):
    parameter_n_steps = st.sidebar.slider('Number of lagged time steps per output value (n_steps)', 1, 10, 6, 1)
    parameter_nodes = st.sidebar.slider('Number of nodes in the LSTM layer', 10, 100, 50, 5)
    parameter_epochs = st.sidebar.slider('Number of epochs used during training', 10, 100, 30, 5)
    parameter_activation = st.sidebar.select_slider('Activation function used in the LSTM layers', options=['relu', 'tanh', 'sigmoid', 'softmax'])
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])

# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df.head())
    built_model = st.button("Build your model!")
    pandas_profile = st.checkbox("Show exploratory data analysis report")
    if pandas_profile:
        pr = df.profile_report()
        st_profile_report(pr)

    if built_model and feature_columns is not None and target_column is not None:
        deeplearning.vanilla_LSTM(df, model_selection, feature_columns, target_column, parameter_n_steps, split_size, parameter_epochs, parameter_nodes)
else:
    st.info('Awaiting for excel file to be uploaded.')
     
