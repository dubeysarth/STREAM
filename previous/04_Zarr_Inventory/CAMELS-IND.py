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
# camels_attributes_graph = pd.read_csv(os.path.join(SAVE_PATH, 'graph_attributes.csv'), index_col=0)
# camels_attributes_graph.index = camels_attributes_graph.index.map(lambda x: str(x).zfill(5))
# camels_attributes_graph['huc_02'] = camels_attributes_graph['huc_02'].map(lambda x: str(x).zfill(2))
# camels_graph = camels_attributes_graph.copy()
# camels_graph = camels_graph[camels_graph['ghi_area'] <= 30000]
# camels_graph = camels_graph[camels_graph['area_percent_difference'] < 10]
# camels_graph = camels_graph[camels_graph['num_nodes'] > 1]
# camels_graph = camels_graph.rename(columns = {'ghi_lon': 'gauge_lon', 'ghi_lat': 'gauge_lat'})
# print(f"Number of catmt's: {len(camels_graph)}")
# del camels_attributes_graph

camels_graph = pd.read_csv(os.path.join(SAVE_PATH, 'nested_gauges', 'graph_attributes_with_nesting.csv'), index_col=0)
camels_graph.index = camels_graph.index.map(lambda x: str(x).zfill(5))
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

#%%
START_DATE = pd.Timestamp('1998-01-01')
END_DATE = pd.Timestamp('2022-12-31')
# camels_graph = camels_graph[camels_graph['huc_02'] == '08']

#%% Init inventory
def init_zarr(huc = '10', gauge_id = '06452000', res_arcmins = 3, round_up_places = 3, uparea = None):
    res_degrees = res_arcmins / 60
    nodes_coords = pd.read_csv(os.path.join(SAVE_PATH, 'graph_files', huc, gauge_id, 'nodes_coords.csv'), index_col = 0)
    edges = pd.read_csv(os.path.join(SAVE_PATH, 'graph_files', huc, gauge_id, 'edges.csv'), index_col = 0)
    edges_idx = edges.loc[:,['from_idx', 'to_idx']].values.tolist()

    lats = np.arange(max(nodes_coords['lat']), min(nodes_coords['lat']) - res_degrees / 2, -res_degrees).astype(np.float32)
    lats = [round(x, round_up_places) for x in lats]

    lons = np.arange(min(nodes_coords['lon']), max(nodes_coords['lon']) + res_degrees / 2, res_degrees).astype(np.float32)
    lons = [round(x, round_up_places) for x in lons]

    mask = xr.DataArray(np.zeros((len(lats), len(lons)), dtype = bool), coords = [lats, lons], dims = ['lat', 'lon'])
    for _, row in nodes_coords.iterrows():
        mask.loc[row['lat'], row['lon']] = True

    catmt = xr.Dataset()
    catmt['mask'] = mask

    catmt.attrs['watershed'] = f"CAMELS-US/HUC:{huc}/GaugeID:{gauge_id}"
    catmt.attrs['gauge_idx'] = [0]
    catmt.attrs['resolution(degrees)'] = res_degrees
    catmt.attrs['resolution(arcmins)'] = res_arcmins
    catmt.attrs['edges_idx'] = edges_idx
    catmt.attrs['drainage_area_km2'] = uparea

    catmt['idx2lat'] = xr.DataArray(nodes_coords['lat'].values.astype(np.float32), dims = 'idx')
    catmt['idx2lon'] = xr.DataArray(nodes_coords['lon'].values.astype(np.float32), dims='idx')
    catmt = catmt.assign_coords(idx = nodes_coords.index.values.astype(np.int32))

    catmt = catmt.assign_coords(time = pd.date_range(start = '1998-01-01', end = '2022-12-31', freq = 'D'))
    catmt = catmt.sel(time=~((catmt.time.dt.month == 2) & (catmt.time.dt.day == 29)))

    catmt = catmt.assign_coords(lead = np.arange(1, 10+1, dtype = np.int32))

    os.makedirs(os.path.join(SAVE_PATH, 'inventory', huc), exist_ok = True)
    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'w')

for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Initializing Zarr'):
    huc, gauge_id = row['huc_02'], row.name
    uparea = row['ghi_area']
    init_zarr(huc, gauge_id, res_arcmins = 3, round_up_places = 3, uparea = uparea)

#%% ERA5
def compile_ERA5_zarr(huc, gauge_id):

    warnings.filterwarnings('ignore')

    data_START_DATE = pd.Timestamp('1998-01-01')
    data_END_DATE = pd.Timestamp('2020-12-31')

    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
    var_names = [
        'total_precipitation',
        'surface_net_solar_radiation',
        'surface_net_thermal_radiation',
        'evaporation',
        'potential_evaporation',
        'runoff',
        'surface_runoff',
        'sub_surface_runoff'
    ]
    for var_name in var_names:
        catmt[f'dynamic_ERA5_{var_name}'] = xr.DataArray(
            np.zeros((len(catmt.time), len(catmt.idx)), dtype = np.float32)*np.nan, 
            dims = ['time', 'idx'], 
            coords = {'time': catmt.time, 'idx': catmt.idx}
        )
        data = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, 'dynamic', 'ERA5', f"{var_name}.csv"), index_col = 0, parse_dates = True)
        data = data.loc[data_START_DATE:data_END_DATE]
        data = data * 24
        for idx in data.columns:
            catmt[f'dynamic_ERA5_{var_name}'].loc[dict(time=slice(data_START_DATE, data_END_DATE), idx=int(idx))] = data[idx].values
        del data, idx
    
    var_names = [
        '2m_temperature',
        'surface_pressure',
        '2m_dewpoint_temperature',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        'snowfall',
        'snow_depth',
        'snowmelt',
        'volumetric_soil_water_layer_1',
        'volumetric_soil_water_layer_2',
        'volumetric_soil_water_layer_3',
        'volumetric_soil_water_layer_4'
    ]
    for var_name in var_names:
        catmt[f'dynamic_ERA5_{var_name}'] = xr.DataArray(
            np.zeros((len(catmt.time), len(catmt.idx)), dtype = np.float32)*np.nan, 
            dims = ['time', 'idx'], 
            coords = {'time': catmt.time, 'idx': catmt.idx}
        )
        data = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, 'dynamic', 'ERA5', f"{var_name}.csv"), index_col = 0, parse_dates = True)
        data = data.loc[data_START_DATE:data_END_DATE]
        for idx in data.columns:
            catmt[f'dynamic_ERA5_{var_name}'].loc[dict(time=slice(data_START_DATE, data_END_DATE), idx=int(idx))] = data[idx].values
        del data, idx

    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_ERA5_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_ERA5_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Compiling ERA5 Zarr')
    )

def compile_ERA5_static_continuous_zarr(huc, gauge_id):
    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
    var_names = ['high_vegetation_cover','low_vegetation_cover']
    for var_name in var_names:
        catmt[f'static_ERA5_{var_name}'] = xr.DataArray(
            np.zeros((len(catmt.idx)), dtype = np.float32)*np.nan, 
            dims = ['idx'], 
            coords = {'idx': catmt.idx}
        )
        data = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, 'static', 'ERA5', f"static_{var_name}.csv"), index_col = 0, parse_dates = True)
        for idx in data.columns:
            catmt[f'static_ERA5_{var_name}'].loc[dict(idx = int(idx))] = data[idx].values[0]
        del data, idx
    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_ERA5_static_continuous_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_ERA5_static_continuous_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Compiling ERA5 Static Continuous Zarr')
    )

# Categorical variables
def compile_ERA5_static_categorical_zarr(huc, gauge_id):
    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
    var_names = ['soil_type', 'type_of_high_vegetation', 'type_of_low_vegetation']
    var_classes = [
        [0,1,2,3,4,5,6,7],
        [0,3,4,5,6,18,19],
        [0,1,2,7,9,10,11,13,16,17,20]
    ]
    var_num_classes = [len(x) for x in var_classes]
    for var_name, var_class in zip(var_names, var_classes):
        # Store as a string, flip the bit to get the class
        catmt[f'static_ERA5_{var_name}'] = xr.DataArray(
            np.zeros((len(catmt.idx)), dtype = int),
            dims = ['idx'], 
            coords = {'idx': catmt.idx}
        )
        data = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, 'static', 'ERA5', f"static_{var_name}.csv"), index_col = 0, parse_dates = True)
        for idx in data.columns:
            node_class = data[idx].values[0]
            node_class = node_class if node_class in var_class else var_class[0]
            # node_class_idx = np.where(np.isin(var_class, node_class))[0][0]
            # node_class_bool = np.zeros(len(var_class), dtype = np.int32)
            # node_class_bool[node_class_idx] = 1
            # node_class_bool_str = ''.join([str(x) for x in node_class_bool])
            # catmt[f'static_ERA5_{var_name}'].loc[dict(idx = int(idx))] = node_class_bool_str
            catmt[f'static_ERA5_{var_name}'].loc[dict(idx = int(idx))] = int(node_class)
        del data, idx
    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_ERA5_static_categorical_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_ERA5_static_categorical_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Compiling ERA5 Static Categorical Zarr')
    )

#%% GLEAM4
def compile_GLEAM4_zarr(huc, gauge_id):
    warnings.filterwarnings('ignore')

    data_START_DATE = pd.Timestamp('1998-01-01')
    data_END_DATE = pd.Timestamp('2020-12-31')

    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
    var_names = ['Ep', 'SMrz', 'SMs'] # ['Ep', 'SMrz', 'SMs', 'Eb', 'Ei', 'Es', 'Et', 'Ew', 'S', 'H']
    for var_name in var_names:
        catmt[f'dynamic_GLEAM4_{var_name}'] = xr.DataArray(
            np.zeros((len(catmt.time), len(catmt.idx)), dtype = np.float32)*np.nan, 
            dims = ['time', 'idx'], 
            coords = {'time': catmt.time, 'idx': catmt.idx}
        )
        data = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, 'dynamic', 'GLEAM4', f"{var_name}.csv"), index_col = 0, parse_dates = True)
        data = data.loc[data_START_DATE:data_END_DATE]
        for idx in data.columns:
            catmt[f'dynamic_GLEAM4_{var_name}'].loc[dict(time=slice(data_START_DATE, data_END_DATE), idx=int(idx))] = data[idx].values
        del data, idx

    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_GLEAM4_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_GLEAM4_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Compiling GLEAM4 Zarr')
    )

#%% GloFAS
def compile_GloFAS_dynamic_zarr(huc, gauge_id):
    warnings.filterwarnings('ignore')

    data_START_DATE = pd.Timestamp('1998-01-01')
    data_END_DATE = pd.Timestamp('2020-12-31')
    
    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
    var_names = ['discharge_mm', 'runoff_water_equivalent', 'snow_depth_water_equivalent', 'soil_wetness_index']
    for var_name in var_names:
        catmt[f'dynamic_GloFAS_{var_name}'] = xr.DataArray(
            np.zeros((len(catmt.time), len(catmt.idx)), dtype = np.float32)*np.nan, 
            dims = ['time', 'idx'], 
            coords = {'time': catmt.time, 'idx': catmt.idx}
        )
        data = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, 'dynamic', 'GloFAS', f"{var_name}.csv"), index_col = 0, parse_dates = True)
        data = data.loc[data_START_DATE:data_END_DATE]
        for idx in data.columns:
            catmt[f'dynamic_GloFAS_{var_name}'].loc[dict(time=slice(data_START_DATE, data_END_DATE), idx=int(idx))] = data[idx].values
        del data, idx

    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_GloFAS_dynamic_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_GloFAS_dynamic_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Compiling GloFAS Dynamic Zarr')
    )

def compile_GloFAS_static_zarr(huc, gauge_id):
    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
    var_names = ['chanbnkf', 'chanflpn', 'changrad', 'chanlength', 'chanman', 'chans', 'chanbw', 'fracforest', 'fracirrigated', 'fracrice', 'fracsealed', 'fracwater', 'fracother', 'cellarea_km2']
    for var_name in var_names:
        catmt[f'static_GloFAS_{var_name}'] = xr.DataArray(
            np.zeros((len(catmt.idx)), dtype = np.float32)*np.nan, 
            dims = ['idx'], 
            coords = {'idx': catmt.idx}
        )
        data = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, 'static', 'GloFAS', f"{var_name}.csv"), index_col = 0, parse_dates = True)
        for idx in data.columns:
            catmt[f'static_GloFAS_{var_name}'].loc[dict(idx = int(idx))] = data[idx].values[0]
        del data, idx
    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_GloFAS_static_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_GloFAS_static_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Compiling GloFAS Static Zarr')
    )

#%%
def compile_HWSD_static_zarr(huc, gauge_id):
    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
    var_names = ['S_CLAY', 'S_GRAVEL', 'S_SAND', 'S_SILT', 'T_CLAY', 'T_GRAVEL', 'T_SAND', 'T_SILT']
    for var_name in var_names:
        catmt[f'static_HWSD_{var_name}'] = xr.DataArray(
            np.zeros((len(catmt.idx)), dtype = np.float32)*np.nan, 
            dims = ['idx'], 
            coords = {'idx': catmt.idx}
        )
        data = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, 'static', 'HWSD', f"{var_name}.csv"), index_col = 0, parse_dates = True)
        for idx in data.columns:
            catmt[f'static_HWSD_{var_name}'].loc[dict(idx = int(idx))] = data[idx].values[0]
        del data, idx
    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_HWSD_static_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_HWSD_static_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Compiling HWSD Static Zarr')
    )

#%%
def compile_solar_insolation_zarr(huc, gauge_id):
    warnings.filterwarnings('ignore')

    data_START_DATE = pd.Timestamp('1998-01-01')
    data_END_DATE = pd.Timestamp('2020-12-31')

    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))

    catmt[f'encoding_solar_insolation'] = xr.DataArray(
        np.zeros((len(catmt.time), len(catmt.idx)), dtype = np.float32)*np.nan, 
        dims = ['time', 'idx'], 
        coords = {'time': catmt.time, 'idx': catmt.idx}
    )
    data = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, "solar_insolation.csv"), index_col = 0, parse_dates = True)
    data = data.loc[data_START_DATE:data_END_DATE]
    for idx in data.columns:
        catmt[f'encoding_solar_insolation'].loc[dict(time=slice(data_START_DATE, data_END_DATE), idx=int(idx))] = data[idx].values
    del data, idx

    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_solar_insolation_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_solar_insolation_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Compiling Solar Insolation Zarr')
    )

#%%
def compile_time_encodings_zarr(huc, gauge_id):
    warnings.filterwarnings('ignore')

    data_START_DATE = pd.Timestamp('1998-01-01')
    data_END_DATE = pd.Timestamp('2020-12-31')

    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
    var_names = ['sine_month', 'sine_weekofyear', 'sine_dayofyear']
    data_all_vars = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, "time_encodings.csv"), index_col = 0, parse_dates = True)
    data_all_vars = data_all_vars.loc[data_START_DATE:data_END_DATE]
    for var_name in var_names:
        data = data_all_vars[var_name].values
        data = data.astype(np.float32)
        catmt[f'encoding_{var_name}'] = xr.DataArray(
            np.zeros((len(catmt.time)), dtype = np.float32)*np.nan,
            dims = ['time'], 
            coords = {'time': catmt.time}
        )
        catmt[f'encoding_{var_name}'].loc[dict(time=slice(data_START_DATE, data_END_DATE))] = data
        del data

    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_time_encodings_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_time_encodings_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Compiling Time Encodings Zarr')
    )

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
        delayed(compile_spatial_encoding_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Compiling Spatial Encoding Zarr')
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
        delayed(compile_LDD_uparea_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Compiling LDD Uparea Zarr')
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
        delayed(compile_terrain_static_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Compiling Terrain Static Zarr')
    )

#%%
def compile_IndiaWRIS_outlet_zarr(huc, gauge_id):
    warnings.filterwarnings('ignore')

    data_START_DATE = pd.Timestamp('1998-01-01')
    data_END_DATE = pd.Timestamp('2020-12-31')

    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
    var_name = 'Q_mm'
    data = pd.read_csv(os.path.join(SAVE_PATH, 'graph_features', huc, gauge_id, "IndiaWRIS.csv"), index_col = 0, parse_dates = True)
    data = data.loc[data_START_DATE:data_END_DATE]
    data = data[var_name].values
    data = data.astype(np.float32)
    catmt[f'outlet_IndiaWRIS_{var_name}'] = xr.DataArray(
        np.zeros((len(catmt.time)), dtype = np.float32)*np.nan,
        dims = ['time'], 
        coords = {'time': catmt.time}
    )
    catmt[f'outlet_IndiaWRIS_{var_name}'].loc[dict(time=slice(data_START_DATE, data_END_DATE))] = data
    del data

    catmt.to_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'), mode = 'a')

# for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph)):
#     huc, gauge_id = row['huc_02'], row.name
#     compile_IndiaWRIS_outlet_zarr(huc, gauge_id)

with Parallel(n_jobs=8, verbose=10) as parallel:
    parallel(
        delayed(compile_IndiaWRIS_outlet_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Compiling IndiaWRIS Outlet Zarr')
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
        delayed(compile_ERA5Land_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Compiling ERA5-Land Zarr')
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
        delayed(compile_GPM_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Compiling GPM Zarr')
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
        delayed(compile_ECMWF_HRES_zarr)(row['huc_02'], row.name) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph), desc='Compiling ECMWF HRES Zarr')
    )

#%%
row = camels_graph.iloc[0]
huc, gauge_id = row['huc_02'], row.name
catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
catmt = catmt[sorted(catmt.data_vars)]
print(f"Number of data variables in {huc} - {gauge_id}: {len(catmt.data_vars)}")