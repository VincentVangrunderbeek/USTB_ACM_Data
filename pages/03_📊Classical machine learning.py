from pycaret.time_series import *
import streamlit as st
import plotly.graph_objects as go
# ---------------------------------#

st.write("""
# Classical statistical and machine learning models

In this implementation, the Pycaret open source library is used to analyze a wide variety of classical statistical and machine learning time series models.

Try adjusting the hyperparameters!
""")

#---------------------------------#

# Sidebar - Specify parameter settings

if 'df' not in st.session_state:
    st.info('Go back to Home page and upload a dataset')
    st.stop()
else:
    df = st.session_state.df

with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 100, 80, 5)
    cv_fold = st.sidebar.slider('cross validation folds', 1, 10, 3, 1)

    # call the pandas dataframe from the Home page
    if df is not None:
        columns = df.columns
        feature_columns = st.sidebar.multiselect('Select the input features', columns)
        columns_1 = columns.drop(feature_columns)
        target_column = st.sidebar.selectbox('Select the target variable', columns_1)
    # columns_2 = columns_1.drop(target_column)
    # index_column = st.sidebar.selectbox("Select the index column (if time series the dat column)", columns_2)

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

def pycaret_models(df, input_features, target_feature, split_size, fold):
    input_features.append(target_feature)
    data = df[input_features]
    horizon = len(data)-len(data)*split_size/100
    setup(data=data, target=target_feature, fh=int(horizon), profile=True)
    plot_model(plot='cv', display_format='streamlit')
    best = compare_models(sort='smape', verbose = True)
    st.write(best)
    df = pull()
    st.write(df)
    tuned_model = tune_model(best)
    st.write(tuned_model)
    plot_model(best, plot='forecast', display_format='streamlit')
    final_best = finalize_model(best)
    X = data[['Temperature (°C)', 'Relative Humidity (%)']]
    predict_model(final_best, X=X[len(X)-80:len(X)], fh=80)
    st.write(final_best)

# model_check = st.sidebar.checkbox('Do you wish to train only a select few model types?')
#         if model_check:
#             model_type = st.sidebar.multiselect('Specify your desired algorithm (s)', options=models())

built_model = st.button("Build your model!")

if built_model and feature_columns is not None and target_column:
    pycaret_models(df, feature_columns, target_column, split_size, cv_fold)
