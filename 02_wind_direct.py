#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
#-----------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#-----------------------------------------------------------------------------------------#

df = pd.read_csv('data/WIND_DIRECTION.csv')
data = df.iloc[4:83, 2:15]
df_sorted = data.sort_values(by=data.columns[0])
wind_dir_columns = [data.columns[1], data.columns[2]] 
encoded_df = pd.DataFrame(index=df_sorted.index)

#-----------------------------------------------------------------------------------------#

for col in wind_dir_columns:
    df_sorted[col] = df_sorted[col].astype(str)  
    encoded = U.encode_directions(df_sorted[col])  
    encoded_df = encoded_df.add(encoded, fill_value=0)  
result = df_sorted.join(encoded_df)
years = df_sorted.iloc[:, 0].astype(str) 

#-----------------------------------------------------------------------------------------#

U.plot(result, years)

#-----------------------------------------------------------------------------------------#