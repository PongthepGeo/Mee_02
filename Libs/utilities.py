import matplotlib.pyplot as plt
import numpy as np
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