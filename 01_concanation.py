#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
#-----------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#-----------------------------------------------------------------------------------------#

df = pd.read_csv('data_02/581301_wind_speed.csv')
# df = pd.read_csv('data/TEMPERATURE.csv')
# df = pd.read_csv('data/WIND_SPEED.csv')
df = df.sort_values(df.columns[1])
for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
scaler = MinMaxScaler(feature_range=(0, 1))
df.iloc[:, 2:] = scaler.fit_transform(df.iloc[:, 2:])
years = df.iloc[:, 1].astype(str) 

#-----------------------------------------------------------------------------------------#

U.plot(df.iloc[:, 2], years)

#-----------------------------------------------------------------------------------------#