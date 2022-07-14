import pandas as pd
from pycaret.time_series import *
import matplotlib.pyplot as plt
import streamlit as st
# https://nbviewer.org/github/pycaret/pycaret/blob/36e132eaf6b59835ec166bdd45fde115bb16ad91/time_series_101.ipynb

st.write("""
# The Machine Learning App

In this implementation, A *LSTM deep learning network* from the Keras library is used to build a supervised machine learning regression model that is able to predict the ACM current from temperature and relative humidity input data.

Try adjusting the hyperparameters!
""")
@st.cache
def import_the_data():
    df = pd.read_excel('data.xlsx', index_col='date')

    df.index = pd.to_datetime(df.index)
    columns = ['Temperature (°C)', 'Relative Humidity (%)', 'Current Q235 (nA)', 'Current Q345 (nA)',
               'Current Q345-W (nA)', 'Electrical Quantity Q235 (C)']
    df.columns = columns
    df = df.resample('D').mean()
    df.interpolate(inplace=True)
    print(df.head())
    return df

# train_size = 300
# train_x, train_y = X[:train_size], X[:train_size]
# test_x, test_y = y[train_size:len(X)], y[train_size:len(y)]
df = import_the_data()
data = df[['Temperature (°C)', 'Relative Humidity (%)', 'Current Q235 (nA)']]
st.write(data.head())
setup(data = data, target = 'Current Q235 (nA)', fh = 80, enforce_exogenous=True)
best = compare_models(sort='smape', verbose = True)
df = pull()
plot_model(plot='train_test_split', display_format='streamlit')
plot_model(plot='cv', display_format='streamlit')
st.write(df)
plot_model(best, plot='forecast', display_format='streamlit')
final_best = finalize_model(best)
X = data[['Temperature (°C)', 'Relative Humidity (%)']]
predict_model(final_best, X=X[len(X)-80:len(X)], fh=80)
st.write(final_best)

