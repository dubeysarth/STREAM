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

camels_graph = pd.read_csv(os.path.join(SAVE_PATH, 'nested_gauges', 'graph_attributes_with_nesting.csv'), index_col=0)
camels_graph.index = camels_graph.index.map(lambda x: str(x).zfill(5))
camels_graph['huc_02'] = camels_graph['huc_02'].map(lambda x: str(x).zfill(2))
# camels_graph = camels_graph[camels_graph['nesting'].isin(['not_nested', 'nested_downstream'])]
camels_graph = camels_graph.reset_index()
print(f"Number of catmt's with nesting: {len(camels_graph)}")

#%%
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

#%%
def process_catmt(huc, gauge_id):
    warnings.filterwarnings('ignore')
    catmt = xr.open_zarr(os.path.join(SAVE_PATH, 'inventory', huc, f'{gauge_id}.zarr'))
    catmt = catmt[sorted(catmt.data_vars)]

    MONTHLY_PATH = os.path.join(SAVE_PATH, 'monthly_inventory')

    #########################
    sum_var_names = []
    # ERA5
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
    var_names = [f"dynamic_ERA5_{var_name}" for var_name in var_names]
    sum_var_names.extend(var_names)

    # # Daymet
    # var_names = [
    #     'prcp',
    # ]
    # var_names = [f"dynamic_Daymet_{var_name}" for var_name in var_names]
    # sum_var_names.extend(var_names)

    # ERA5-Land
    var_names = [
        'total_precipitation_sum', # SUM
        'total_evaporation_sum', # SUM
        'potential_evaporation_sum', # SUM
        'surface_net_solar_radiation_sum', # SUM
        'surface_net_thermal_radiation_sum', # SUM
        'snowfall_sum', # SUM
        'snowmelt_sum', # SUM
        'runoff_sum', # SUM
        'surface_runoff_sum', # SUM
        'sub_surface_runoff_sum', # SUM
    ]
    var_names = [f"dynamic_ERA5-Land_{var_name}" for var_name in var_names]
    sum_var_names.extend(var_names)

    # GLEAM4
    var_names = [
        'Ep', # SUM
    ]
    var_names = [f"dynamic_GLEAM4_{var_name}" for var_name in var_names]
    sum_var_names.extend(var_names)

    # GPM
    var_names = [
        'Early_Run', # SUM
        'Late_Run', # SUM
        'Final_Run', # SUM
    ]
    var_names = [f"dynamic_GPM_{var_name}" for var_name in var_names]
    sum_var_names.extend(var_names)

    # GloFAS
    var_names = [
        'discharge_mm', # SUM
        'runoff_water_equivalent', # SUM
    ]
    var_names = [f"dynamic_GloFAS_{var_name}" for var_name in var_names]
    sum_var_names.extend(var_names)

    # IndiaWRIS
    var_names = [
        'outlet_IndiaWRIS_Q_mm'
    ]
    # var_names = [f"dynamic_USGS_{var_name}" for var_name in var_names]
    sum_var_names.extend(var_names)

    catmt_sum = catmt[sum_var_names].copy()

    # Resample to monthly sum at month-start
    catmt_sum = catmt_sum.resample(time='1MS').sum(dim='time')
    #########################

    #########################
    mean_var_names = []

    # ERA5
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
    var_names = [f"dynamic_ERA5_{var_name}" for var_name in var_names]
    mean_var_names.extend(var_names)

    # # Daymet
    # var_names = [
    #     'srad', # MEAN
    #     'swe', # MEAN
    #     'tmax', # MAX
    #     'tmin', # MIN
    #     'vp', # MEAN
    #     'dayl', # MEAN
    # ]
    # var_names = [f"dynamic_Daymet_{var_name}" for var_name in var_names]
    # mean_var_names.extend(var_names)

    # ERA5-Land
    var_names = [
        'temperature_2m_min', # MIN
        'temperature_2m_max', # MAX
        'surface_pressure', # MEAN
        'u_component_of_wind_10m', # MEAN
        'v_component_of_wind_10m', # MEAN
        'snow_depth', # MEAN
        'snow_cover', # MEAN
        'dewpoint_temperature_2m_min', # MIN
        'dewpoint_temperature_2m_max', # MAX
        'leaf_area_index_high_vegetation', # MEAN
        'leaf_area_index_low_vegetation', # MEAN
        'volumetric_soil_water_layer_1', # MEAN
        'volumetric_soil_water_layer_2', # MEAN
        'volumetric_soil_water_layer_3', # MEAN
        'volumetric_soil_water_layer_4', # MEAN
    ]
    var_names = [f"dynamic_ERA5-Land_{var_name}" for var_name in var_names]
    mean_var_names.extend(var_names)

    # GLEAM4
    var_names = [
        'SMs', # MEAN
        'SMrz' # MEAN
    ]
    var_names = [f"dynamic_GLEAM4_{var_name}" for var_name in var_names]
    mean_var_names.extend(var_names)

    # GPM

    # GloFAS
    var_names = [
        'snow_depth_water_equivalent', # MEAN
        'soil_wetness_index', # MEAN
    ]
    var_names = [f"dynamic_GloFAS_{var_name}" for var_name in var_names]
    mean_var_names.extend(var_names)

    # Encodings
    var_names = [
        'encoding_solar_insolation', # MEAN
        'encoding_sine_dayofyear',
        'encoding_sine_weekofyear',
        'encoding_sine_month'
        
        
    ]
    mean_var_names.extend(var_names)

    catmt_mean = catmt[mean_var_names].copy()

    # Resample to monthly mean at month-start
    catmt_mean = catmt_mean.resample(time='1MS').mean(dim='time')
    #########################

    #########################
    catmt_non_dynamic = catmt[[var for var in catmt.data_vars if not (var in sum_var_names or var in mean_var_names or var.startswith('dynamic_HRES'))]]
    catmt_non_dynamic = catmt_non_dynamic[sorted(catmt_non_dynamic.data_vars)]
    #########################

    catmt_monthly = xr.merge([catmt_sum, catmt_mean, catmt_non_dynamic])
    catmt_monthly = catmt_monthly[sorted(catmt_monthly.data_vars)]

    # Save to zarr
    os.makedirs(os.path.join(MONTHLY_PATH, huc), exist_ok=True)
    catmt_monthly.to_zarr(os.path.join(MONTHLY_PATH, huc, f'{gauge_id}.zarr'), mode='w', consolidated=True)

    # Clean up
    catmt.close()
    del catmt, catmt_sum, catmt_mean, catmt_non_dynamic, catmt_monthly
    gc.collect()

#%%
with Parallel(n_jobs=16, verbose=10) as parallel:
    parallel(
        delayed(process_catmt)(row['huc_02'], row['gauge_id']) for idx, row in tqdm.tqdm(camels_graph.iterrows(), total=len(camels_graph))
    )