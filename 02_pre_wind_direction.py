#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
#-----------------------------------------------------------------------------------------#
import pandas as pd
import os
import numpy as np
#-----------------------------------------------------------------------------------------#

csv = 'data_03/ALLTHAILAND/CSV FILE/winddirection.csv'
folder_name = 'preprocessing'
save_csv = 'wind_direction.csv'
start_row = 4
end_row = 3926
start_col = 3
end_col = 14

#-----------------------------------------------------------------------------------------#

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

#-----------------------------------------------------------------------------------------#

df_original = pd.read_csv(csv)
df = df_original.copy()

#-----------------------------------------------------------------------------------------#

df['zip_code'] = df.iloc[start_row:, 1].str.split('-').str[0]
df['zip_code'] = df['zip_code'].fillna('0')  
df['zip_code'] = df['zip_code'].astype(int)

#-----------------------------------------------------------------------------------------#

df, direction_to_int_mapping = U.encode_wind_directions(df, start_col, end_col)
dict = {
    'zip_code': df['zip_code'],
    'year':     df.iloc[start_row:end_row, 2],
    'Jan':      df.iloc[start_row:end_row, 3],
    'Feb':      df.iloc[start_row:end_row, 4],
    'Mar':      df.iloc[start_row:end_row, 5],
    'Apr':      df.iloc[start_row:end_row, 6],
    'May':      df.iloc[start_row:end_row, 7],
    'Jun':      df.iloc[start_row:end_row, 8],
    'Jul':      df.iloc[start_row:end_row, 9],
    'Aug':      df.iloc[start_row:end_row, 10],
    'Sep':      df.iloc[start_row:end_row, 11],
    'Oct':      df.iloc[start_row:end_row, 12],
    'Nov':      df.iloc[start_row:end_row, 13],
    'Dec':      df.iloc[start_row:end_row, 14]  
}

#-----------------------------------------------------------------------------------------#

df_new = pd.DataFrame(dict)
df_new = df_new.iloc[4:3924+1]
csv_file_path = os.path.join(folder_name, save_csv)
df_new.to_csv(csv_file_path, index=False)
print(f"CSV file saved at: {csv_file_path}")

#-----------------------------------------------------------------------------------------#
