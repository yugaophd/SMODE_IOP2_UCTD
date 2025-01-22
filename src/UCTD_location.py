# %%
# make a scatter plot of the UCTD locations

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cftime
from datetime import datetime, timedelta
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import xarray as xr
import glob
import os

os.chdir('/Users/yugao/Desktop/projects/SMODE_IOP2_UCTD/src')

# %%    
# Load the data
campaign = 'IOP2'
data_dir = f'/Users/yugao/Desktop/projects/SMODE_IOP2_UCTD/data/external/{campaign}/'    
data_files = glob.glob(data_dir + f'S_MODE_{campaign}_*L2*.nc')
data_files.sort()
data_files

# %%
# Step 1: Determine global vmin and vmax
global_min = float('inf')
global_max = float('-inf')
for file in data_files:
    print(f"Processing file for min/max: {file}")
    ds = xr.open_dataset(file, decode_times=False)
    time_values = ds['time'].values
    global_min = min(global_min, time_values.min())
    global_max = max(global_max, time_values.max())

print(f"Global min time: {global_min}, Global max time: {global_max}")

# Set up color normalization and colormap
vmin = global_min
vmax = global_max
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = cm.viridis

# %%
# Step 2: Create the plot
fig, ax = plt.subplots(
    2, 1,
    figsize=(10, 10),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

for file in data_files:
    print(f"Plotting file: {file}")
    ds = xr.open_dataset(file, decode_times=False)
    
    time_values = ds['time'].values  # Keep as numeric values
    latitude = ds['latitude'].values
    longitude = ds['longitude'].values
    
    # Scatter plot for both panels
    ax[0].scatter(longitude, latitude, c=time_values, cmap=cmap, norm=norm, s=10, transform=ccrs.PlateCarree())
    ax[1].scatter(longitude, latitude, c=time_values, cmap=cmap, norm=norm, s=10, transform=ccrs.PlateCarree())

# Add map features
for a in ax:
    a.add_feature(cfeature.COASTLINE)
    a.add_feature(cfeature.BORDERS, linestyle=':')
    a.add_feature(cfeature.LAND, facecolor='lightgray')
    
    # Add gridlines
    gl = a.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False  # Turn off top labels
    gl.ylabels_right = False  # Turn off right labels
    gl.xlabel_style = {'size': 8, 'color': 'black'}
    gl.ylabel_style = {'size': 8, 'color': 'black'}

# Adjust plot extents
ax[0].set_extent([-126, -123.5, 36.5, 37.5], crs=ccrs.PlateCarree())
ax[1].set_extent([-126, -122, 36.5, 39], crs=ccrs.PlateCarree())

# Add a single color bar for both panels
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.95, 0.01, 0.02, 0.5])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
cbar.set_label('Date')

# Format color bar ticks as dates
# Convert vmin to a datetime object for the base date
time_units = ds['time'].attrs['units']  # Use the units from the dataset
calendar = ds['time'].attrs.get('calendar', 'gregorian')
base_date = cftime.num2date(vmin, time_units, calendar)

# Create ticks and labels
date_ticks = np.linspace(vmin, vmax, num=11)  # Adjust the number of ticks as needed
cbar.set_ticks(date_ticks)
cbar.set_ticklabels([(base_date + timedelta(days=int(tick - vmin))).strftime('%Y-%m-%d') for tick in date_ticks])

# Add titles
ax[0].set_title('Zoomed Location of UCTD Casts')
ax[1].set_title('Location of UCTD Casts')

plt.show()
plt.savefig(f'../img/UCTD_location_{campaign}.png')

# %%
