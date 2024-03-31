import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#-----------------------------------------------------------------------------------------#
import matplotlib
params = {
	'savefig.dpi': 300,  
	'figure.dpi' : 300,
	'axes.labelsize':12,  
	'axes.titlesize':12,
	'axes.titleweight': 'bold',
	'legend.fontsize': 10,
	'xtick.labelsize':10,
	'ytick.labelsize':10,
	'font.serif': 'Times New Roman'
}
matplotlib.rcParams.update(params)
#-----------------------------------------------------------------------------------------#

def plot(one_column, years):
    plt.figure()
    num_row = len(one_column)
    axis_x = np.linspace(start=0, stop=num_row - 1, num=num_row)  # Create a linear space for x-axis
    plt.scatter(axis_x, one_column, s=50, alpha=0.75)
    plt.plot(axis_x, one_column, alpha=0.75)  # Added to connect points
    tick_interval = max(len(axis_x) // 10, 1)  # Adjust the divisor to control the number of ticks
    ticks_to_use = axis_x[::tick_interval]
    labels_to_use = years[::tick_interval]
    plt.xticks(ticks=ticks_to_use, labels=labels_to_use, rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Normalized Data')
    plt.title('One Month')
    plt.show()

def encode_directions(series):
    return series.str.get_dummies(sep=',')

def normalize_columns_exclude_outliers(df, start_col, end_col):
    for col in df.columns[start_col:end_col+1]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        temp = df[col].replace(-999, np.nan)
        normalized = (temp - temp.min()) / (temp.max() - temp.min())
        df[col] = normalized.fillna(-999)
    return df

def encode_wind_directions(df, start_col, end_col):
    unique_directions = set()
    for col in df.columns[start_col:end_col+1]:
        directions = df[col].astype(str).unique()
        for dir in directions:
            if dir.lower() == 'nan' or dir == '':
                continue
            if "," in dir:  # Handle combinations by sorting
                dir = ",".join(sorted(dir.split(",")))
            unique_directions.add(dir)
    direction_to_int = {'nan': -999, '': -999}
    for i, dir in enumerate(sorted(unique_directions), start=1):
        direction_to_int[dir] = i
    for col in df.columns[start_col:end_col+1]:
        df[col] = df[col].astype(str).apply(lambda x: ",".join(sorted(x.split(","))) if "," in x else x)
        df[col] = df[col].map(direction_to_int).fillna(-999).astype(int)
    return df, direction_to_int

def plot_ONI(data, output_folder):
    plt.figure(figsize=(10, 5))
    plt.plot(data['ANOM'], label='ONI Anomaly', color='black')
    plt.fill_between(data.index, 0.5, data['ANOM'], where=(data['ANOM'] > 0.5),
                     color='red', alpha=0.3, label='Anomaly > 0.5')
    plt.fill_between(data.index, -0.5, data['ANOM'], where=(data['ANOM'] < -0.5),
                     color='blue', alpha=0.3, label='Anomaly < -0.5')
    plt.axhline(0.5, color='red', linestyle='--', lw=1)
    plt.axhline(0.0, color='grey', linestyle='--', lw=1)
    plt.axhline(-0.5, color='blue', linestyle='--', lw=1)
    years = data['YR'].unique()
    decade_ticks = years[years % 10 == 0]  # Select years that are multiples of 10
    plt.xticks(ticks=[data[data['YR'] == year].index[0] for year in decade_ticks], 
               labels=decade_ticks)
    plt.xlabel('Year')
    plt.ylabel('ONI Anomaly')
    plt.title('Oceanic Niño Index (ONI)')
    plt.legend()
    plt.savefig(output_folder + "/ONI.png", format='png', bbox_inches='tight', pad_inches=0, transparent=False)
    plt.show()

def plot_box(df_metrics):
    metrics = ['Precision', 'Recall', 'F1_Score', 'Accuracy']
    data_to_plot = [df_metrics[metric].values for metric in metrics]
    positions = np.arange(1, len(metrics) + 1)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.4,
                    patch_artist=True, boxprops=dict(facecolor="skyblue"), flierprops=dict(markerfacecolor='g', marker='D'))
    ax.set_xticks(positions)
    ax.set_xticklabels(metrics)
    plt.title('TabNet Model Metrics')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.subplots_adjust(wspace=0.3)
    plt.savefig('figures/tabnet_boxplots.png', format='png', bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.show()