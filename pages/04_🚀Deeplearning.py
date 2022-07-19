import netron
import os
import pandas as pd
# Import Libraries
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from numpy import array
from sklearn.preprocessing import MinMaxScaler
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

models = ('Vanilla LSTM', 'Stacked LSTM', 'Bidirectional LSTM')

# function that splits the data in the right way for a LSTM variant to be read
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# preproces funtion that takes in the data from the import function and deos the scaling and test train split
def data_preproces(data, n_steps, train_test_split, variables):
    # only predict for Current of Q235-This needs to be an optino that can be chosen in the future
    values = data[variables].values

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # define X and y
    X, y = split_sequences(scaled, n_steps)

    # split the data into training and testing
    train_size = int(len(X) * train_test_split)
    test_size = len(X) - train_size
    train_x, train_y = X[:train_size], y[:train_size]
    test_x, test_y = X[train_size:len(X)], y[train_size:len(y)]
    n_features = train_x.shape[2]
    return train_x, train_y, test_x, test_y, scaler, n_features


# error function
def error_function(inv_y, inv_yhat):
    error_rmse = mean_squared_error(
        y_true=inv_y,
        y_pred=inv_yhat,
        squared=False
    )
    r2 = r2_score(inv_y, inv_yhat)
    return r2, error_rmse


# Function that predicts and rescales the data
def predict_rescale(test_x, test_y, model, scaler, n_steps):
    # predict the values
    yhat = model.predict(test_x, verbose=0)
    # rescale the forecasted values
    rescaling = n_steps - 1
    test_X = test_x[:, rescaling:]
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    test_y = test_y.reshape((len(test_y), 1))
    inv_yhat = concatenate((test_X[:, :], yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -1]

    # Make sure that no negative currents are present
    inv_yhat[inv_yhat < 0] = 0

    # invert scaling for actual
    inv_y = concatenate((test_X[:, :], test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -1]
    return inv_y, inv_yhat


# plotting function
def plot(df, train_x, n_steps, inv_yhat, model_name, target_feature):
    # Plot
    index_y = train_x.shape[0] + n_steps - 1
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df[target_feature].iloc[index_y:].index, y=df[target_feature].iloc[index_y:], name='actual', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=df[target_feature].iloc[index_y:].index, y=inv_yhat, name='predicted', mode='lines+markers'))
    fig.update_layout(
        title=model_name,
        title_x=0.5,
        xaxis_title='Date',
        yaxis_title=target_feature
    )
    st.plotly_chart(fig)


def fitting_plot(train_x, train_y, epochs, test_x, test_y, model):
    # fit model, epoch is the maount of iterables
    # Make a clalback class that checks if the loss doesnt decrease anymore to stop the epoch counter
    history = model.fit(train_x, train_y, epochs=epochs, validation_data=(test_x, test_y), verbose=0)
    # summarize history for accuracy
    # Plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(y=history.history['loss'], name='train', mode='lines+markers'))
    fig.add_trace(go.Scatter(y=history.history['val_loss'], name='test', mode='lines+markers'))
    fig.update_layout(
        title='Model loss',
        title_x=0.5,
        xaxis_title='Epoch',
        yaxis_title='Loss'
    )
    st.plotly_chart(fig)
    return model


# ---------------------------------#
# Model building
def LSTM_models(df, model_selection, input_features, target_feature, n_steps=6, train_test_split=0.8, epochs=30, nodes=50):
    # get the values from the data preproces function
    input_features.append(target_feature)
    train_x, train_y, test_x, test_y, scaler, n_features = data_preproces(df, n_steps, train_test_split / 100, input_features)

    # Show the train and testing data on the app
    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(train_x.shape)
    st.write('Test set')
    st.info(test_x.shape)

    # Show the features and the output
    st.markdown('**1.3. Variable details**:')
    st.write('Input features')
    st.info(input_features[:-1])
    st.write('Target variable')
    st.info(target_feature)

    # check what model is selected
    if model_selection == models[0]:
        model_name = models[0]
        # define model. simple lstm with 50 hidden LSTM units. Dense means that all of the nodes are connected with each other
        model = Sequential()
        model.add(LSTM(nodes, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
    elif model_selection == models[1]:
        model_name = models[1]
        # define model.
        model = Sequential()
        model.add(LSTM(nodes, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        model.add(LSTM(nodes, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
    elif model_selection == models[2]:
        model_name = models[2]
        # define model.
        model = Sequential()
        model.add(Bidirectional(LSTM(nodes, activation='relu'), input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

    # information about the training of the model
    st.subheader('2. Model training')
    # call the fit and history plotter
    model = fitting_plot(train_x, train_y, epochs, test_x, test_y, model)

    # call the predicter and rescaler
    inv_y, inv_yhat = predict_rescale(test_x, test_y, model, scaler, n_steps)

    # call Test error  function
    r2, error_rmse = error_function(inv_y, inv_yhat)

    st.subheader('3. Model Performance test set')
    st.markdown('**3.1. Test set $R^2$ and RMSE**')
    st.write('Coefficient of determination ($R^2$):')
    st.info(round(r2, ndigits=2))

    st.write('Error (RMSE):')
    st.info(round(error_rmse))

    # call the plotting function
    st.markdown('**4.2. Actual vs predicted ACM values**')
    plot(df, train_x, n_steps, inv_yhat, model_name, target_feature)

    return model

# User interface
st.write("""
# Deep learning models

In this implementation, A *LSTM deep learning network* from the Keras library is used to build a supervised machine learning regression model that is able to predict the ACM current from temperature and relative humidity input data. 

Try adjusting the hyperparameters!
""")

if 'df' not in st.session_state:
    st.info('Go back to Home page and upload a dataset')
    st.stop()
else:
    df = st.session_state.df

with st.sidebar.header('2. LSTM model'):
    model_selection = st.sidebar.selectbox('Select the deep learning model of choice', models)
    username = os.getlogin()
    file_name = 'C:/Users/' + username + '/Downloads/model.h5'
    download_model = st.sidebar.checkbox(f'Do you wish to download and the model see its architecture after building (path is {file_name})')

with st.sidebar.header('3. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    if df is not None:
        columns = df.columns
        feature_columns = st.sidebar.multiselect('Select the input features', columns)
        columns_1 = columns.drop(feature_columns)
        target_column = st.sidebar.selectbox('Select the target variable', columns_1)
        # columns_2 = columns_1.drop(target_column)
        # index_column = st.sidebar.selectbox("Select the index column (if time series the dat column)", columns_2)

with st.sidebar.subheader('3.1 Specific learning Parameters'):
    parameter_n_steps = st.sidebar.slider('Number of lagged time steps per output value (n_steps)', 1, 10, 6, 1)
    parameter_nodes = st.sidebar.slider('Number of nodes in the LSTM layer', 10, 100, 50, 5)
    parameter_epochs = st.sidebar.slider('Number of epochs used during training', 10, 100, 30, 5)
    parameter_activation = st.sidebar.select_slider('Activation function used in the LSTM layers', options=['relu', 'tanh', 'sigmoid', 'softmax'])
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])

# show the dataset the user has chosen based on the uploaded data and the selected columns
st.subheader('Data information')
col1, col2 = st.columns(2)
col1.markdown('**Input feature(s)**')
col2.markdown('**Target variables(s)**')
col1.write(df[feature_columns].head())
col2.write(df[target_column].head())

# show the train test split distribution
st.markdown('**Train test split**')
train_size = int(len(df) * split_size/100)
df_train= df[:train_size]
df_test = df[train_size:len(df)]
print(len(df_train))
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=df_train.index, y=df_train[target_column], name='Train (original)', mode='lines+markers'))
fig.add_trace(go.Scatter(x=df_test.index, y=df_test[target_column], name='Test (original)', mode='lines+markers'))
fig.update_layout(
    title='Train/test split',
    title_x=0.5,
    yaxis_title=target_column
)
st.plotly_chart(fig)



built_model = st.button("Build (and download) your model!")
if built_model and feature_columns is not None and target_column:
    model = LSTM_models(df, model_selection, feature_columns, target_column, parameter_n_steps, split_size, parameter_epochs, parameter_nodes)
    # download the model doesnt work yet
    if download_model:
        model.save(file_name)
        netron.start(file_name)

