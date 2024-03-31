import pandas as pd

# Load the anomaly data
data = pd.read_csv('data_03/ONI.txt', sep='\s+')

# Initialize the default label
data['label'] = 1

# Assign labels based on the anomaly values
data.loc[data['ANOM'] > 0.5, 'label'] = 2  # Positive anomalies
data.loc[data['ANOM'] < -0.5, 'label'] = 3  # Negative anomalies, correctly labeled as 2

# Group by year to determine the label for each year
# The label for the year can be the max label observed in that year to ensure priority is given to significant anomalies
yearly_labels = data.groupby('YR')['label'].max().reset_index()

# Merge the yearly label into final_df based on the year
final_df = pd.read_csv('preprocessing/merged_data.csv')  # Assuming final_df needs to be read again
final_df = final_df.merge(yearly_labels, left_on='year', right_on='YR', how='left')

# Replace NaN labels with -999 for years without a matching anomaly label
final_df['label'] = final_df['label'].fillna(-999).astype(int)

# Clean up final_df by dropping any unnecessary columns (e.g., 'YR' if added by the merge)
final_df.drop(columns=['YR'], inplace=True, errors='ignore')

# Save the updated DataFrame
final_csv_path = 'preprocessing/merged_data_with_labels.csv'
final_df.to_csv(final_csv_path, index=False)
print(f"Data successfully merged with labels and saved to {final_csv_path}")
