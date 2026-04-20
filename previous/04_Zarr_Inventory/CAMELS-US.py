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

# camels_attributes_graph = pd.read_csv(os.path.join(SAVE_PATH, 'graph_attributes.csv'), index_col=0)
# camels_attributes_graph.index = camels_attributes_graph.index.map(lambda x: str(x).zfill(8))
# camels_attributes_graph['huc_02'] = camels_attributes_graph['huc_02'].map(lambda x: str(x).zfill(2))
# camels_graph = camels_attributes_graph.copy()
# camels_graph = camels_graph[camels_graph['area_percent_difference'] < 10]
# camels_graph = camels_graph[camels_graph['num_nodes'] > 1]
# print(f"Number of CAMELS-US catmt's: {len(camels_graph)}")
# del camels_attributes_graph

camels_graph = pd.read_csv(os.path.join(SAVE_PATH, 'nested_gauges', 'graph_attributes_with_nesting.csv'), index_col=0)
camels_graph.index = camels_graph.index.map(lambda x: str(x).zfill(8))
camels_graph['huc_02'] = camels_graph['huc_02'].map(lambda x: str(x).zfill(2))
# camels_graph = camels_graph[camels_graph['nesting'].isin(['not_nested', 'nested_downstream'])]
# camels_graph = camels_graph.reset_index()
print(f"Number of catmt's with nesting: {len(camels_graph)}")

def idx_to_map(ds, var_name):
    lats = ds.lat.values
    lons = ds.lon.values
    catmt_var_map = xr.DataArray(
        np.zeros((len(lats), len(lons)), dtype = np.float32)*np.nan,
        dims = ['lat', 'lon'],
        coords = {'lat': lats, 'lon': lons}
    )
    for idx in ds.idx.values:
        lat, lon = ds['idx2lat'].sel(idx = idx).values, ds['idx2lon'].sel(idx = idx).values
        catmt_var_map.loc[dict(lat = lat, lon = lon)] = ds[var_name].sel(idx = idx).values
    return catmt_var_map

START_DATE = pd.Timestamp('1998-01-01')
END_DATE = pd.Timestamp('2022-12-31')

camels_graph = camels_graph[camels_graph['huc_02'] != '05']

#%%
def compile_spatial_encoding_zarr(huc, gauge_id):
    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))

    catmt[f'encoding_transformed_longitude'] = xr.DataArray(
        np.zeros((len(catmt.idx)), dtype = np.float32)*np.nan, 
        dims = ['idx'], 
        coords = {'idx': catmt.idx}
    )
    catmt[f'encoding_transformed_latitude'] = xr.DataArray(
        np.zeros((len(catmt.idx)), dtype = np.float32)*np.nan, 
        dims = ['idx'], 
        coords = {'idx': catmt.idx}
    )

    # lon: -180 to 180; lat: -60 to 90
    lon_transform = lambda x: np.sin(2 * np.pi * (x+180) / 360)
    lat_transform = lambda x: (x - (-60))/(90 - (-60))

    nodes_coords = pd.read_csv(os.path.join(SAVE_PATH, 'graph_files', huc, gauge_id, 'nodes_coords.csv'), index_col = 0)
    for node_idx, node_row in nodes_coords.iterrows():
        lat, lon = node_row['lat'], node_row['lon']
        lon_transformed = lon_transform(lon)
        lat_transformed = lat_transform(lat)
        catmt['encoding_transformed_longitude'].loc[dict(idx = node_idx)] = lon_transformed
        catmt['encoding_transformed_latitude'].loc[dict(idx = node_idx)] = lat_transformed

    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_spatial_encoding_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_spatial_encoding_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph))
    )

#%%
def compile_LDD_uparea_zarr(huc, gauge_id):
    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
    var_name = 'uparea'
    catmt[f'static_uparea'] = xr.DataArray(
        np.zeros((len(catmt.idx)), dtype = np.float32)*np.nan, 
        dims = ['idx'], 
        coords = {'idx': catmt.idx}
    )
    data = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, 'uparea.csv'), index_col = 0, parse_dates = True)
    for idx in data.columns:
        catmt[f'static_uparea'].loc[dict(idx = int(idx))] = data[idx].values[0]
    del data, idx
    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_LDD_uparea_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_LDD_uparea_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph))
    )

#%%
def compile_terrain_static_zarr(huc, gauge_id):
    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
    var_names = ['elv', 'slope_percentage', 'slope_riserun', 'slope_degrees', 'slope_radians', 'aspect', 'curvature', 'planform_curvature', 'profile_curvature', 'upa', 'wth']
    metrics = ['mean', 'std', '50%']
    for var_name in var_names:
        data = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, 'static', 'MERIT-Hydro', f"{var_name}.csv"), index_col = 0, parse_dates = True)
        for metric in metrics:
            catmt[f'static_terrain_{var_name}_{metric}'] = xr.DataArray(
                np.zeros((len(catmt.idx)), dtype = np.float32)*np.nan, 
                dims = ['idx'], 
                coords = {'idx': catmt.idx}
            )
            for idx in data.columns:
                catmt[f'static_terrain_{var_name}_{metric}'].loc[dict(idx = int(idx))] = data.loc[metric, idx]
        del data, idx
    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_terrain_static_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_terrain_static_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph))
    )

#%%
def compile_USGS_outlet_zarr(huc, gauge_id):
    warnings.filterwarnings('ignore')

    data_START_DATE = pd.Timestamp('1998-01-01')
    data_END_DATE = pd.Timestamp('2020-12-31')

    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
    var_name = 'Q_mm'
    data = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, "USGS.csv"), index_col = 0, parse_dates = True)
    data = data.loc[data_START_DATE:data_END_DATE]
    data = data[var_name].values
    data = data.astype(np.float32)
    catmt[f'outlet_USGS_{var_name}'] = xr.DataArray(
        np.zeros((len(catmt.time)), dtype = np.float32)*np.nan,
        dims = ['time'], 
        coords = {'time': catmt.time}
    )
    catmt[f'outlet_USGS_{var_name}'].loc[dict(time=slice(data_START_DATE, data_END_DATE))] = data
    del data

    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_USGS_outlet_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_USGS_outlet_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph))
    )

#%%
def compile_ERA5Land_zarr(huc, gauge_id):

    warnings.filterwarnings('ignore')

    data_START_DATE = pd.Timestamp('1998-01-01')
    data_END_DATE = pd.Timestamp('2019-12-31')

    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
    var_names = [
        'total_precipitation_sum',
        'temperature_2m_min',
        'temperature_2m_max',
        'total_evaporation_sum',
        'potential_evaporation_sum',
        'surface_net_solar_radiation_sum',
        'surface_net_thermal_radiation_sum',
        'surface_pressure',
        'u_component_of_wind_10m',
        'v_component_of_wind_10m',
        'snowfall_sum',
        'snowmelt_sum',
        'snow_depth',
        'snow_cover',
        'dewpoint_temperature_2m_min',
        'dewpoint_temperature_2m_max',
        'leaf_area_index_high_vegetation',
        'leaf_area_index_low_vegetation',
        'runoff_sum',
        'surface_runoff_sum',
        'sub_surface_runoff_sum',
        'volumetric_soil_water_layer_1',
        'volumetric_soil_water_layer_2',
        'volumetric_soil_water_layer_3',
        'volumetric_soil_water_layer_4',
    ]
    for var_name in var_names:
        catmt[f'dynamic_ERA5-Land_{var_name}'] = xr.DataArray(
            np.zeros((len(catmt.time), len(catmt.idx)), dtype = np.float32)*np.nan, 
            dims = ['time', 'idx'], 
            coords = {'time': catmt.time, 'idx': catmt.idx}
        )
        data = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, 'dynamic', 'ERA5-Land', f"{var_name}.csv"), index_col = 0, parse_dates = True)
        data = data.loc[data_START_DATE:data_END_DATE]
        for idx in data.columns:
            catmt[f'dynamic_ERA5-Land_{var_name}'].loc[dict(time=slice(data_START_DATE, data_END_DATE), idx=int(idx))] = data[idx].values
        del data, idx

    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_ERA5Land_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_ERA5Land_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph))
    )

#%%
def compile_GPM_zarr(huc, gauge_id):

    warnings.filterwarnings('ignore')

    data_START_DATE = pd.Timestamp('1999-01-01')
    data_END_DATE = pd.Timestamp('2020-12-31')

    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
    sources = [
        'Early_Run',
        'Late_Run',
        'Final_Run'
    ]
    for source in sources:
        catmt[f'dynamic_GPM_{source}'] = xr.DataArray(
            np.zeros((len(catmt.time), len(catmt.idx)), dtype = np.float32)*np.nan, 
            dims = ['time', 'idx'], 
            coords = {'time': catmt.time, 'idx': catmt.idx}
        )
        # data = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, 'dynamic', 'GPM', source, f"*.csv"), index_col = 0, parse_dates = True)
        data_files = sorted(glob.glob(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, 'dynamic', 'GPM', source, "*.csv")))
        data = pd.concat([pd.read_csv(file, index_col=0, parse_dates=True) for file in data_files], axis=0)
        data = data.loc[data_START_DATE:data_END_DATE]
        for idx in data.columns:
            catmt[f'dynamic_GPM_{source}'].loc[dict(time=slice(data_START_DATE, data_END_DATE), idx=int(idx))] = data[idx].values
        del data, idx

    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_GPM_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_GPM_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph))
    )

#%%
def compile_ECMWF_HRES_zarr(huc, gauge_id):

    warnings.filterwarnings('ignore')

    data_START_DATE = pd.Timestamp('2016-01-01')
    data_END_DATE = pd.Timestamp('2020-12-31')

    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
    var_names = [
        '2m_temperature',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        'total_precipitation_24hr',
        'surface_pressure',
    ]
    leads = [24*x for x in range(1, 10+1)]
    catmt = catmt.assign_coords({'lead': leads})
    for var_name in var_names:
        catmt[f'dynamic_HRES_{var_name}'] = xr.DataArray(
            np.zeros((len(catmt.time), len(catmt.idx), len(catmt.lead)), dtype=np.float32) * np.nan, 
            dims=['time', 'idx', 'lead'], 
            coords={'time': catmt.time, 'idx': catmt.idx, 'lead': catmt.lead}
        )
        for lead in leads:
            data = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, 'dynamic', 'ECMWF_HRES', var_name, f"{lead}hrs.csv"), index_col = 0, parse_dates = True)
            data = data.loc[data_START_DATE:data_END_DATE]
            for idx in data.columns:
                catmt[f'dynamic_HRES_{var_name}'].loc[dict(time=slice(data_START_DATE, data_END_DATE), idx=int(idx), lead=lead)] = data[idx].values
            del data, idx

    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_ECMWF_HRES_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_ECMWF_HRES_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph))
    )
