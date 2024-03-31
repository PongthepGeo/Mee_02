#-----------------------------------------------------------------------------------------#
import matplotlib
import rasterio
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from PIL import Image
import geopandas as gpd
from skimage import exposure, io
from collections import defaultdict
import pandas as pd
#-----------------------------------------------------------------------------------------#
params = {
	'savefig.dpi': 300,  
	'figure.dpi' : 300,
	'axes.labelsize':12,  
	'axes.titlesize':12,
	'axes.titleweight': 'bold',
	'legend.fontsize': 10,
	'xtick.labelsize':10,
	'ytick.labelsize':10,
	'font.family': 'serif',
	'font.serif': 'Times New Roman'
}
matplotlib.rcParams.update(params)
#-----------------------------------------------------------------------------------------#

class DEMPlotter:
    def __init__(self, dem_file):
        self.dem_file = dem_file

    def format_degree(self, value, pos):
        """Helper function to format tick labels with degree symbols."""
        return f'{value:.2f}°'

    def plot(self, save_plot=False, save_image_folder="image_out", output_filename="DEM.svg"):
        with rasterio.open(self.dem_file) as dem:
            elevation = dem.read(1)
            extent = [dem.bounds.left, dem.bounds.right, dem.bounds.bottom, dem.bounds.top]

        # Custom colormap with discrete colors
        elevationPalette = [
            'A0522D', '8B4513', 'A52A2A',  # Brown shades for lower elevations
            '00FF66', '00FF00', '66FF00', 'CCFF00',
            '0000FF', '0066FF', '00CCFF', 'FFFFFF'  # Blue shades and white for higher elevations
        ]
        elevation_colors = [mcolors.to_rgb(f"#{c}") for c in elevationPalette]
        num_colors = len(elevationPalette)
        cmap = mcolors.ListedColormap(elevation_colors)
        bounds = np.linspace(elevation.min(), elevation.max(), num_colors)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.figure()
        plt.imshow(elevation, cmap=cmap, norm=norm, extent=extent)
        plt.colorbar(label='Elevation', boundaries=bounds, ticks=bounds)
        plt.title('Digital Elevation Model (DEM)')
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.format_degree))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.format_degree))
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        if save_plot:
            plt.savefig(f'{save_image_folder}/{output_filename}', format='svg',
                        bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()

#-----------------------------------------------------------------------------------------#

class Data_Point_Plotter:
    def __init__(self, dem_file, csv_file):
        self.dem_file = dem_file
        self.csv_file = csv_file

    def format_degree(self, value, pos):
        """Helper function to format tick labels with degree symbols."""
        return f'{value:.2f}°'

    def plot(self, save_plot=False, save_image_folder="image_out", output_filename="DEM.svg"):
        # Read the DEM file
        with rasterio.open(self.dem_file) as dem:
            elevation = dem.read(1)
            extent = [dem.bounds.left, dem.bounds.right, dem.bounds.bottom, dem.bounds.top]
            # Create a mask for non-zero elevation values
            mask = elevation > 0
            # Mask the elevation data
            elevation_masked = np.ma.masked_where(~mask, elevation)
        
        # Read the CSV file
        df = pd.read_csv(self.csv_file)
        # Sum cases for the same lat long
        grouped = df.groupby(['x', 'y'])['Cases'].sum().reset_index()
        
        # Set up the plot
        fig, ax = plt.subplots()
        # Display masked DEM in the background with partial transparency
        ax.imshow(elevation_masked, extent=extent, cmap='gray', alpha=0.5)
        
        # Plot the squares for data points
        sc = ax.scatter(grouped['x'], grouped['y'], c=grouped['Cases'], cmap='Reds', alpha=0.6,
            marker='o', s=50)
        
        plt.colorbar(sc, label='Number of Cases')
        ax.set_title('Case Distribution on DEM')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        if save_plot:
            fig.savefig(f'{save_image_folder}/{output_filename}', format='svg',
                        bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()

#-----------------------------------------------------------------------------------------#