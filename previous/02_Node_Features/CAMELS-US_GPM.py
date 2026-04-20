#%% Setting Up
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import networkx as nx
import rioxarray as rxr

import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

from shapely.geometry import Point
from shapely.geometry import Polygon

import glob
import os
import itertools
import tqdm
import gc
import time
import pickle

from joblib import Parallel, delayed

import rioxarray as rxr

import configparser
cfg = configparser.ConfigParser()
cfg.optionxform = str
cfg.read('/home/sarth/rootdir/datadir/assets/defaults.ini')
cfg = {s: dict(cfg.items(s)) for s in cfg.sections()}
PATHS = cfg['PATHS']

print("Setting up...")

#%% Region-Specific: CAMELS-US
DIRNAME = '03min_GloFAS_CAMELS-US'
SAVE_PATH = os.path.join(PATHS['devp_datasets'], DIRNAME)
resolution = 0.05
lon_360_180 = lambda x: (x + 180) % 360 - 180 # convert 0-360 to -180-180
lon_180_360 = lambda x: x % 360 # convert -180-180 to 0-360
region_bounds = {
    'minx': -130,
    'miny': 20,
    'maxx': -65,
    'maxy': 50
}
camels_attributes_graph = pd.read_csv(os.path.join(SAVE_PATH, 'graph_attributes.csv'), index_col=0)
camels_attributes_graph.index = camels_attributes_graph.index.map(lambda x: str(x).zfill(8))
camels_attributes_graph['huc_02'] = camels_attributes_graph['huc_02'].map(lambda x: str(x).zfill(2))
camels_graph = camels_attributes_graph.copy()
camels_graph = camels_graph[camels_graph['area_percent_difference'] < 10]
camels_graph = camels_graph[camels_graph['num_nodes'] > 1]
print(f"Number of CAMELS-US catmt's: {len(camels_graph)}")
del camels_attributes_graph

#%% GloFAS Grid
os.makedirs(os.path.join(SAVE_PATH, "graph_features"), exist_ok = True)

ldd = xr.open_dataset(os.path.join(PATHS['gis_ldd'], 'GloFAS_03min', 'ldd.nc'))
ldd = ldd['ldd']
ldd = ldd.sel(
    lat = slice(region_bounds['maxy'], region_bounds['miny']), 
    lon = slice(region_bounds['minx'], region_bounds['maxx'])
)

lons = ldd['lon'].values
lats = ldd['lat'].values

ds_grid = xr.Dataset({
    'lat': (['lat'], lats),
    'lon': (['lon'], lons),
})

# Round the lat lon values to 3 decimal places in ds_grid
ds_grid['lat'] = ds_grid['lat'].round(3)
ds_grid['lon'] = ds_grid['lon'].round(3)

#%% GPM Data

def first_run(gpm_product, year):
    regridder_weights = os.path.join(PATHS['assets'], 'regridder', 'regridder_gpm_to_glofas_03min_US.nc')

    tif_files = sorted(glob.glob(f"{PATHS['GPM']}/{gpm_product}/{year}/*.tif")) # YYYY-MM-DD format
    tif_files = [f for f in tif_files if not ('-02-29' in f)] # Exclude leap days
    print(f"Number of tif files: {len(tif_files)}")

    tif_file = tif_files[0]
    ds = rxr.open_rasterio(tif_file, masked=True).squeeze()
    ds = ds.sel(
        x = slice(region_bounds['minx'], region_bounds['maxx']), 
        y = slice(region_bounds['miny'], region_bounds['maxy'])
    )
    ds = ds.drop_vars(['band', 'spatial_ref'], errors='ignore')
    ds = ds.rename({'x': 'lon', 'y': 'lat'})

    # Print the shape of the dataset
    print(f"Shape of the dataset: {ds.shape}")

    if not os.path.exists(regridder_weights):
        regridder = xe.Regridder(
            ds, 
            ds_grid, 
            'bilinear', 
            reuse_weights=False, 
            # filename = regridder_weights
        )
        regridder.to_netcdf(regridder_weights)
    else:
        regridder = xe.Regridder(
            ds, 
            ds_grid, 
            'bilinear', 
            reuse_weights=True, 
            filename = regridder_weights
        )

    ds_regrided = regridder(ds)
    print(f"Shape of the regridded dataset: {ds_regrided.shape}")
    ds.close()
    ds_regrided.close()

    del regridder, ds, ds_regrided
    gc.collect()

    return None

def yearly_run(year, gpm_product, verbose=False):
    regridder_weights = os.path.join(PATHS['assets'], 'regridder', 'regridder_gpm_to_glofas_03min_US.nc')

    tif_files = sorted(glob.glob(f"{PATHS['GPM']}/{gpm_product}/{year}/*.tif")) # YYYY-MM-DD format
    tif_files = [f for f in tif_files if not ('-02-29' in f)] # Exclude leap days
    if verbose: print(f"Number of tif files: {len(tif_files)}")

    def load_tif(tif_file):
        da = rxr.open_rasterio(tif_file, masked=True).squeeze()
        da = da.sel(
            x = slice(region_bounds['minx'], region_bounds['maxx']), 
            y = slice(region_bounds['miny'], region_bounds['maxy'])
        )
        da = da.drop_vars(['band', 'spatial_ref'], errors='ignore')
        da = da.rename({'x': 'lon', 'y': 'lat'})
        return da
    
    ds_list = Parallel(n_jobs=-1)(
        delayed(load_tif)(tif_file) for tif_file in tif_files
    )
    ds = xr.concat(ds_list, dim='time')
    del ds_list
    gc.collect()

    if verbose: print(f"Shape of the dataset: {ds.shape}")

    if not os.path.exists(regridder_weights):
        regridder = xe.Regridder(
            ds, 
            ds_grid, 
            'bilinear', 
            reuse_weights=False, 
            # filename = regridder_weights
        )
        regridder.to_netcdf(regridder_weights)
    else:
        regridder = xe.Regridder(
            ds, 
            ds_grid, 
            'bilinear', 
            reuse_weights=True, 
            filename = regridder_weights
        )

    ds_regrided = regridder(ds)
    if verbose: print(f"Shape of the regridded dataset: {ds_regrided.shape}")
    ds.close()
    del regridder, ds

    dates = [os.path.basename(f).split('.')[0] for f in tif_files]
    dates = pd.to_datetime(dates, format='%Y-%m-%d')

    ds_regrided['time'] = ('time', dates)

    # Set DataArray ds_regrided to float32
    ds_regrided = ds_regrided.astype('float32')

    ds_regrided.load()

    def process(idx, row):
        huc, gauge_id = row['huc_02'], row.name
        nodes_coords = pd.read_csv(os.path.join(SAVE_PATH, 'graph_files', huc, gauge_id, 'nodes_coords.csv'), index_col = 0)
        data = pd.DataFrame(index = dates, columns = nodes_coords.index.astype(str))
        for node_idx, node_row in nodes_coords.iterrows():
            lat, lon = node_row['lat'], node_row['lon']
            ds_window_loc = ds_regrided.sel(lat = lat, lon = lon, method = 'nearest')
            data.loc[:, str(node_idx)] = ds_window_loc.values
        assert not data.isna().any().any(), f"NaN values found in data for {huc} {gauge_id}"
        data = data.astype('float32')
        os.makedirs(os.path.join(SAVE_PATH, "graph_features", huc, gauge_id, 'dynamic', 'GPM', gpm_product), exist_ok=True)
        data.to_csv(os.path.join(SAVE_PATH, "graph_features", huc, gauge_id, 'dynamic', 'GPM', gpm_product, f"{year}.csv"))

    with Parallel(n_jobs = 8, verbose = 0) as parallel:
        _ = parallel(delayed(process)(idx, row) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc=f"{gpm_product} {year}"))
        # _ = parallel(delayed(process)(idx, row) for idx, row in camels_graph.iterrows())

    ds_regrided.close()
    del ds_regrided
    gc.collect()

    return None

#%% Early Run
start_year = 1998
end_year = 2022
gpm_product = 'Early_Run'

first_run(gpm_product, start_year)
for year in range(start_year, end_year + 1):
    try: yearly_run(year, gpm_product, verbose=False)
    except: print(f"Failed to process {year} for {gpm_product}. Skipping...")

#%% Late Run
start_year = 1998
end_year = 2022
gpm_product = 'Late_Run'

first_run(gpm_product, start_year)
for year in range(start_year, end_year + 1):
    try: yearly_run(year, gpm_product, verbose=False)
    except: print(f"Failed to process {year} for {gpm_product}. Skipping...")

#%% Final Run
start_year = 1998
end_year = 2022
gpm_product = 'Final_Run'

first_run(gpm_product, start_year)
for year in range(start_year, end_year + 1):
    try: yearly_run(year, gpm_product, verbose=False)
    except: print(f"Failed to process {year} for {gpm_product}. Skipping...")