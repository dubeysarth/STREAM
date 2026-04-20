import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import networkx as nx

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

import configparser
cfg = configparser.ConfigParser()
cfg.optionxform = str
cfg.read('/home/sarth/rootdir/assets/global.ini')
cfg = {s: dict(cfg.items(s)) for s in cfg.sections()}
PATHS = cfg['PATHS']

DIRNAME = '15min_MERIT-Plus_CAMELS-US'
SAVE_PATH = os.path.join(PATHS['devp_datasets'], DIRNAME)
resolution = 0.25
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
camels_graph.sort_values(ascending=True, by = 'huc_02').groupby('huc_02').size()

del camels_attributes_graph

os.makedirs(os.path.join(SAVE_PATH, "graph_features"), exist_ok = True)

ldd = xr.open_dataset(os.path.join(PATHS['gis_ldd'], 'MERIT-Plus_15min', 'ldd.nc'))
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

dates = pd.date_range('1980-01-01', '2020-12-31', freq='D')
dates = dates[~((dates.month == 2) & (dates.day == 29))]
print(f"Number of dates: {len(dates)}")

# var_names = ['prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp', 'dayl']
# var_names = ['prcp', 'tmin', 'tmax', 'dayl']
var_names = ['srad', 'swe', 'vp']

def process(idx, row, var_name):
    huc, gauge_id = row['huc_02'], row.name
    nodes_coords = pd.read_csv(os.path.join(PATHS['devp_datasets'], DIRNAME, 'graph_files', huc, gauge_id, 'nodes_coords.csv'), index_col = 0)
    data = pd.DataFrame(index = dates, columns = nodes_coords.index)
    os.makedirs(os.path.join(PATHS['devp_datasets'], DIRNAME, 'graph_features', huc, gauge_id, 'dynamic'), exist_ok = True)
    os.makedirs(os.path.join(PATHS['devp_datasets'], DIRNAME, 'graph_features', huc, gauge_id, 'dynamic', 'Daymet'), exist_ok = True)
    data.to_csv(os.path.join(PATHS['devp_datasets'], DIRNAME, 'graph_features', huc, gauge_id, 'dynamic', 'Daymet', f"{var_name}.csv"))

for var_name in var_names:
    print(var_name)
    with Parallel(n_jobs = 8, verbose = 0) as parallel:
        _ = parallel(delayed(process)(idx, row, var_name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)))

missing_dates = ['1980-12-31', '1984-12-31', '1988-12-31', '1992-12-31',
               '1996-12-31', '2000-12-31', '2004-12-31', '2008-12-31',
               '2012-12-31', '2016-12-31', '2020-12-31']
missing_dates = [date + 'T12:00:00' for date in missing_dates]
missing_dates = np.array(missing_dates, dtype = 'datetime64')
ds_missing_dates = xr.DataArray(
    np.nan*np.zeros((len(missing_dates), len(lats), len(lons))),
    coords = [missing_dates, lats, lons],
    dims = ['time', 'lat', 'lon']
)

for var_name in itertools.islice(var_names, 0, None, 1):
    print(var_name)
    ds = xr.open_mfdataset(os.path.join(PATHS['Daymet'], var_name, f"*.nc"), combine='by_coords')
    ds = ds[var_name]
    ds = ds.rename({'x': 'lon', 'y': 'lat'})
    ds = ds.sel(time=~((ds['time.month'] == 2) & (ds['time.day'] == 29)))
    ds = ds.sel(
        lat = slice(region_bounds['maxy'], region_bounds['miny']), 
        lon = slice(region_bounds['minx'], region_bounds['maxx'])
    )

    if os.path.exists(os.path.join(PATHS['Assets'], 'regridder_daymet_to_merit-plus_15min.nc')):
        regridder = xe.Regridder(
            ds, 
            ds_grid, 
            'bilinear', 
            reuse_weights=True, 
            filename = os.path.join(PATHS['Assets'], 'regridder_daymet_to_merit-plus_15min.nc')
        )
    else:
        regridder = xe.Regridder(
            ds, 
            ds_grid, 
            'bilinear', 
            reuse_weights=False
        )
        regridder.to_netcdf(os.path.join(PATHS['Assets'], 'regridder_daymet_to_merit-plus_15min.nc'))
    
    ds_regrided = regridder(ds)

    # Concatenate missing dates
    ds_regrided = xr.concat([ds_regrided, ds_missing_dates], dim = 'time')
    ds_regrided = ds_regrided.sortby('time')
    ds.close()
    # Print length of time
    print(f"timesteps: {len(ds_regrided['time'])}")

    for start_year in range(1980, 2020+1, 5):
        start_date = f"{start_year}-01-01"
        end_date = f"{min(start_year+4,2020)}-12-31"
        ds_window = ds_regrided.sel(time = slice(start_date, end_date)).copy()
        start_time = time.time()
        ds_window.load()
        end_time = time.time()
        print(start_date, end_date, f"Time: {(end_time - start_time)/60:.2f} mins")
    
        def process(idx, row):
            huc, gauge_id = row['huc_02'], row.name
            nodes_coords = pd.read_csv(os.path.join(PATHS['devp_datasets'], DIRNAME, 'graph_files', huc, gauge_id, 'nodes_coords.csv'), index_col = 0)
            data = pd.read_csv(os.path.join(PATHS['devp_datasets'], DIRNAME, 'graph_features', huc, gauge_id, 'dynamic', 'Daymet', f"{var_name}.csv"), index_col = 0, parse_dates = True)
            for node_idx, node_row in nodes_coords.iterrows():
                lat, lon = node_row['lat'], node_row['lon']
                ds_window_loc = ds_window.sel(lat = lat, lon = lon, method = 'nearest')
                data.loc[start_date:end_date, str(node_idx)] = ds_window_loc.values
            data.to_csv(os.path.join(PATHS['devp_datasets'], DIRNAME, 'graph_features', huc, gauge_id, 'dynamic', 'Daymet', f"{var_name}.csv"))
            return None
        
        with Parallel(n_jobs = 8, verbose = 0) as parallel:
            _ = parallel(delayed(process)(idx, row) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)))

        ds_window.close()
        del ds_window
        gc.collect()

    ds_regrided.close()
    del ds, ds_regrided
    gc.collect()

for idx, row in tqdm.tqdm(camels_graph.iterrows()):
    huc, gauge_id = row['huc_02'], row.name
    nodes_coords = pd.read_csv(os.path.join(PATHS['devp_datasets'], DIRNAME, 'graph_files', huc, gauge_id, 'nodes_coords.csv'), index_col = 0)
    for var_name in var_names:
        data = pd.read_csv(os.path.join(PATHS['devp_datasets'], DIRNAME, 'graph_features', huc, gauge_id, 'dynamic', 'Daymet', f"{var_name}.csv"), index_col = 0, parse_dates = True)
        # Fill the NaN values with a window of 15 days centered around the missing value
        for col in data.columns:
            data[col] = data[col].fillna(data[col].rolling(15, min_periods = 1, center = True).mean())
        data.to_csv(os.path.join(PATHS['devp_datasets'], DIRNAME, 'graph_features', huc, gauge_id, 'dynamic', 'Daymet', f"{var_name}.csv"))