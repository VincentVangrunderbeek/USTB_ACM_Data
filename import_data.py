# Import Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file = 'data.xlsx'
# import function that reads the Excel file of interest
def import_data(data):
    df = pd.read_excel(data, index_col='date')

    df.index = pd.to_datetime(df.index)
    columns = ['Temperature (°C)', 'Relative Humidity (%)', 'Current Q235 (nA)', 'Current Q345 (nA)',
               'Current Q345-W (nA)', 'Electrical Quantity Q235 (C)']
    df.columns = columns

    df['Electrical Quantity Q345 (C)'] = df['Current Q345 (nA)'].cumsum()
    df['Electrical Quantity Q345 (C)'] = df['Electrical Quantity Q345 (C)'] * 0.00000006

    df['Electrical Quantity Q345-W (C)'] = df['Current Q345-W (nA)'].cumsum()
    df['Electrical Quantity Q345-W (C)'] = df['Electrical Quantity Q345-W (C)'] * 0.00000006

    # scale down the data where the threshold of rain (1000 nA) is reached by a factor or 0.2
    df['Current Q235 (nA)'] = np.where(df['Current Q235 (nA)'] >= 1000, df['Current Q235 (nA)'] * 0.2,
                                       df['Current Q235 (nA)'])

    # resample the data to an hourly frame for better GPU processing time
    df_mean = df.resample('H').mean()
    df_mean.interpolate(inplace=True)

    return df_mean


df = import_data(file)
print(df.head())