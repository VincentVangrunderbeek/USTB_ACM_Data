
# Import Libraries
import streamlit as st
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5

from numpy import array
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# from keras.layers import Bidirectional
# from keras.layers import TimeDistributed
# from keras.layers.convolutional import Conv1D
# from keras.layers.convolutional import MaxPooling1D
# from keras.layers import ConvLSTM2D
# from keras.callbacks import EarlyStopping
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

models = ('Vanilla LSTM', 'Stacked LSTM', 'Bidirectional LSTM')
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning App',
    layout='wide',
    page_icon="📈")

# import data function
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
    
    # resample the data to an hourly frame for better GPU processing time
    df_mean = df.resample(timeframe).mean()
    df_mean.interpolate(inplace = True)
    
    return df_mean

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
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

#preproces funtion that takes in the data from the import function and deos the scaling and test train split
def data_preproces(data, n_steps, train_test_split):
    
    #only predict for Current of Q235-This needs to be an optino that cna be chosen in the future
    values = data[['Temperature (°C)', 'Relative Humidity (%)', 'Current Q235 (nA)']].values
    
    #scale the data 
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    #define X and y
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
                    y_true = inv_y,
                    y_pred = inv_yhat,
                    squared = False
                )
    r2 = r2_score(inv_y, inv_yhat)
    return r2, error_rmse

# Function that predicts and rescales the data
def predict_rescale(test_x, test_y, model, scaler, n_steps):

    #predict the values
    yhat = model.predict(test_x, verbose = 0)
    #rescale the forecasted values
    rescaling = n_steps - 1
    test_X = test_x[:, rescaling:]
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    test_y = test_y.reshape((len(test_y), 1))
    inv_yhat = concatenate((test_X[:,:], yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,-1] 
    
    #Make sure that no negative currents are present
    inv_yhat[inv_yhat < 0] = 0
    
    # invert scaling for actual
    inv_y = concatenate((test_X[:,:], test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,-1]

    return inv_y, inv_yhat

#plotting function   
def plot(df, train_x, n_steps, inv_yhat, model_name):    
    # Plot
    index_y = train_x.shape[0]+n_steps-1
    fig, ax = plt.subplots(figsize=(15, 12))
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.title(model_name, fontsize = 18)
    plt.plot(df['Current Q235 (nA)'].iloc[index_y:], label='Actual')
    plt.plot(df['Current Q235 (nA)'].iloc[index_y:].index, inv_yhat, label='Predicted')
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Galvanic current (nA)', fontsize=18)
    ax.legend(fontsize=12)
    st.pyplot(fig)
    

def fitting_plot(train_x, train_y, epochs, test_x, test_y, model):
    # fit model, epoch is the maount of iterables
    # Make a clalback class that checks if the loss doesnt decrease anymore to stop the epoch counter
    history = model.fit(train_x, train_y, epochs=epochs, validation_data=(test_x, test_y), verbose = 0)
    # summarize history for accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend(['train', 'test'], loc='upper right')
    st.pyplot(fig)
    
    return model

#---------------------------------#
# Model building
def vanilla_LSTM(df, n_steps=6, train_test_split=0.8, epochs=30, nodes=50):
    #get the values from the data preproces function
    train_x, train_y, test_x, test_y, scaler, n_features = data_preproces(df, n_steps, train_test_split/100)
    
    # Show the train and testing data on the app
    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(train_x.shape)
    st.write('Test set')
    st.info(test_x.shape)
    
    #Show the features and the output
    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(df.columns[0:2]))
    st.write('Y variable')
    st.info(df.columns[2])
    
    model_name = 'Vanilla LSTM'
    # define model. simple lstm with 50 hidden LSTM units. Dense means that all of the nodes are connected with each other
    model = Sequential()
    model.add(LSTM(nodes, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    st.subheader('2. Model training')
    # call the fit and history plotter
    model = fitting_plot(train_x, train_y, epochs, test_x, test_y, model)
    
    # call the predicter and rescaler
    inv_y, inv_yhat = predict_rescale(test_x, test_y, model, scaler, n_steps)
    
    # call Test error  function
    r2, error_rmse = error_function(inv_y, inv_yhat)
#     error = 'R-squared = ' + str(round(r2, ndigits=2)) + ', RMSE = ' + str(round(error_rmse))
    
    
    st.subheader('3. Model Performance test set')
    st.markdown('**3.1. Test set $R^2$ and RMSE**')
    st.write('Coefficient of determination ($R^2$):')
    st.info( round(r2, ndigits=2) )

    st.write('Error (RMSE):')
    st.info( round(error_rmse) )
    
    # call the plotting function
    st.markdown('**3.2. Actual vs predicted ACM values**')
    plot(df, train_x, n_steps, inv_yhat, model_name)
        
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

# Sidebar - Specify parameter settings
with st.sidebar.header('3. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    timeframe = st.sidebar.select_slider('What timeframe should it be resampled to (standard 1 min)?', options = ['30min', 'H', '6H', 'D'])
    
with st.sidebar.subheader('3.1. Learning Parameters'):
    parameter_n_steps = st.sidebar.slider('Number of lagged time steps per output value (n_steps)', 1, 10, 6, 1)
    parameter_nodes = st.sidebar.slider('Number of nodes in the LSTM layer', 10, 100, 50, 5)
    parameter_epochs = st.sidebar.slider('Number of epochs used during training', 10, 100, 30, 5)
    parameter_activation = st.sidebar.select_slider('Activation function used in the LSTM layers', options = ['relu', 'tanh', 'sigmoid', 'softmax'])
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])

# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = import_data(uploaded_file, timeframe)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    vanilla_LSTM(df, parameter_n_steps, split_size, parameter_epochs, parameter_nodes)
else:
    st.info('Awaiting for excel file to be uploaded.')
     
