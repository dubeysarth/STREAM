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

#%% Region-Specific: CAMELS-IND
DIRNAME = '03min_GloFAS_CAMELS-IND'
SAVE_PATH = os.path.join(PATHS['devp_datasets'], DIRNAME)
resolution = 0.05
lon_360_180 = lambda x: (x + 180) % 360 - 180 # convert 0-360 to -180-180
lon_180_360 = lambda x: x % 360 # convert -180-180 to 0-360
region_bounds = {
    'minx': 66,
    'miny': 5,
    'maxx': 100,
    'maxy': 30
}
camels_attributes_graph = pd.read_csv(os.path.join(SAVE_PATH, 'graph_attributes.csv'), index_col=0)
camels_attributes_graph.index = camels_attributes_graph.index.map(lambda x: str(x).zfill(5))
camels_attributes_graph['huc_02'] = camels_attributes_graph['huc_02'].map(lambda x: str(x).zfill(2))
camels_graph = camels_attributes_graph.copy()
camels_graph = camels_graph[camels_graph['ghi_area'] <= 30000]
camels_graph = camels_graph[camels_graph['area_percent_difference'] < 10]
camels_graph = camels_graph[camels_graph['num_nodes'] > 1]
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

#%% ECMWF HRES Data
hres = xr.open_zarr(os.path.join(PATHS['WeatherBench2'], 'ECMWF_HRES_2016_2022.zarr'), consolidated=True, chunks="auto")

hres['prediction_timedelta'] = hres['prediction_timedelta'].astype('float32')  # Ensure the dtype is float32
hres['prediction_timedelta'] = hres['prediction_timedelta'] / (1e9 * 3600)  # Convert to hours
hres['prediction_timedelta'] = hres['prediction_timedelta'].astype('int32')  # Ensure the dtype is int32

hres['longitude'] = lon_360_180(hres['longitude'])
hres = hres.sortby(['longitude', 'latitude', 'prediction_timedelta'])
hres = hres.rename({'longitude': 'lon', 'latitude': 'lat'})
hres = hres.sel(
    lat = slice(region_bounds['miny'], region_bounds['maxy']), 
    lon = slice(region_bounds['minx'], region_bounds['maxx'])
)

hres = hres.sel(time=hres['time'][~((hres['time'].dt.month == 2) & (hres['time'].dt.day == 29))])

lead_times = hres['prediction_timedelta'].values[1:]

var_names = list(hres.data_vars)

for var_name in var_names:
    print(f"Processing {var_name}")
    for lead_time in lead_times:
        subset = hres[var_name].sel(prediction_timedelta=lead_time)
        # subset = subset.compute()
        start_time = time.time()
        subset.load()
        end_time = time.time()
        print(f"Loading took {(end_time - start_time) / 60:.2f} minutes")

        regridder_weights = os.path.join(PATHS['assets'], 'regridder', 'regridder_ecmwf_hres_to_glofas_03min_IND.nc')

        if not os.path.exists(regridder_weights):
            regridder = xe.Regridder(
                subset, 
                ds_grid, 
                'bilinear', 
                reuse_weights=False, 
                # filename = regridder_weights
            )
            regridder.to_netcdf(regridder_weights)
        else:
            regridder = xe.Regridder(
                subset, 
                ds_grid, 
                'bilinear', 
                reuse_weights=True, 
                filename = regridder_weights
            )

        ds_regrided = regridder(subset)
        subset.close()
        del regridder, subset
        start_time = time.time()
        ds_regrided.load()
        end_time = time.time()
        print(f"Regridding took {(end_time - start_time) / 60:.2f} minutes")

        def process(idx, row):
            huc, gauge_id = row['huc_02'], row.name
            nodes_coords = pd.read_csv(os.path.join(SAVE_PATH, 'graph_files', huc, gauge_id, 'nodes_coords.csv'), index_col = 0)
            data = pd.DataFrame(index=ds_regrided['time'].values, columns=nodes_coords.index.astype(str))
            for node_idx, node_row in nodes_coords.iterrows():
                lat, lon = node_row['lat'], node_row['lon']
                ds_window_loc = ds_regrided.sel(lat=lat, lon=lon, method='nearest')
                data.loc[:, str(node_idx)] = ds_window_loc.values
            assert not data.isna().any().any(), f"NaN values found in data for {huc} {gauge_id}"
            data = data.astype('float32')
            os.makedirs(os.path.join(SAVE_PATH, "graph_features", huc, gauge_id, 'dynamic', 'ECMWF_HRES', var_name), exist_ok=True)
            data.to_csv(os.path.join(SAVE_PATH, "graph_features", huc, gauge_id, 'dynamic', 'ECMWF_HRES', var_name, f"{lead_time}hrs.csv"))

        with Parallel(n_jobs = 8, verbose = 0) as parallel:
            _ = parallel(delayed(process)(idx, row) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc=f"{lead_time}hrs"))
            # _ = parallel(delayed(process)(idx, row) for idx, row in camels_graph.iterrows())

        ds_regrided.close()
        del ds_regrided
        gc.collect()