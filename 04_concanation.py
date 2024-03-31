#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
#-----------------------------------------------------------------------------------------#
import pandas as pd
import os
from functools import reduce
#-----------------------------------------------------------------------------------------#

file_paths = {
    'rain': 'preprocessing/rain.csv',
    'temperature': 'preprocessing/temperature.csv',
    'wind_direction': 'preprocessing/wind_direction.csv',
    'windspeed': 'preprocessing/windspeed.csv'
}

#-----------------------------------------------------------------------------------------#

# Step 1: Determine the overall min/max year span for each zip_code
zip_code_year_spans = {}
for path in file_paths.values():
    df = pd.read_csv(path)
    for zip_code in df['zip_code'].unique():
        zip_years = df[df['zip_code'] == zip_code]['year']
        if zip_code not in zip_code_year_spans:
            zip_code_year_spans[zip_code] = [zip_years.min(), zip_years.max()]
        else:
            zip_code_year_spans[zip_code][0] = min(zip_code_year_spans[zip_code][0], zip_years.min())
            zip_code_year_spans[zip_code][1] = max(zip_code_year_spans[zip_code][1], zip_years.max())

#-----------------------------------------------------------------------------------------#

# Step 2: Pad DataFrames with missing years for each zip_code with -999
dfs = {}
for key, path in file_paths.items():
    df = pd.read_csv(path)
    padded_rows = []
    for zip_code, (min_year, max_year) in zip_code_year_spans.items():
        for year in range(min_year, max_year + 1):
            if not ((df['zip_code'] == zip_code) & (df['year'] == year)).any():
                # If year for zip_code is missing, create a row with -999 for data columns
                new_row = {'zip_code': zip_code, 'year': year}
                new_row.update({col: -999 for col in df.columns if col not in ['zip_code', 'year']})
                padded_rows.append(new_row)
    if padded_rows:
        df = pd.concat([df, pd.DataFrame(padded_rows)], ignore_index=True)
    rename_cols = {col: col + '_' + key for col in df.columns if col not in ['zip_code', 'year']}
    df.rename(columns=rename_cols, inplace=True)
    dfs[key] = df

#-----------------------------------------------------------------------------------------#

# Merge all DataFrames on 'zip_code' and 'year'
final_df = reduce(lambda left, right: pd.merge(left, right, on=['zip_code', 'year'], how='outer'), dfs.values())

#-----------------------------------------------------------------------------------------#

# Save the merged DataFrame
final_csv_path = 'preprocessing/merged_data.csv'
final_df.to_csv(final_csv_path, index=False)
print(f"Data successfully merged and saved to {final_csv_path}")

#-----------------------------------------------------------------------------------------#