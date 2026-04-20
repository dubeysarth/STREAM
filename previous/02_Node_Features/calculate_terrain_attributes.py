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
camels_graph.sort_values(ascending=True, by = 'huc_02').groupby('huc_02').size()

del camels_attributes_graph

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

import richdem as rd
def calculate_terrain_attributes(dem):
    slope_percentage = rd.TerrainAttribute(dem, attrib='slope_percentage')
    slope_riserun = rd.TerrainAttribute(dem, attrib='slope_riserun')
    slope_degrees = rd.TerrainAttribute(dem, attrib='slope_degrees')
    slope_radians = rd.TerrainAttribute(dem, attrib='slope_radians')
    aspect = rd.TerrainAttribute(dem, attrib='aspect')
    curvature = rd.TerrainAttribute(dem, attrib='curvature')
    planform_curvature = rd.TerrainAttribute(dem, attrib='planform_curvature')
    profile_curvature = rd.TerrainAttribute(dem, attrib='profile_curvature')
    results = {
        'slope_percentage': slope_percentage,
        'slope_riserun': slope_riserun,
        'slope_degrees': slope_degrees,
        'slope_radians': slope_radians,
        'aspect': aspect,
        'curvature': curvature,
        'planform_curvature': planform_curvature,
        'profile_curvature': profile_curvature
    }
    return results

tiles_paths = sorted(glob.glob(os.path.join(PATHS['MERIT-Hydro'], 'elv', '**', '*.tif'), recursive=True))
valid_tiles = ['n00w120', 'n00w090', 'n30w120', 'n30w090', 'n30w150', 'n00e060']
tiles_paths = [tile for tile in tiles_paths if not os.path.basename(os.path.dirname(tile)).split('_')[-1] in valid_tiles]
tiles_filenames = [os.path.basename(tile) for tile in tiles_paths]
tiles_names = [tile.split('_')[0] for tile in tiles_filenames]
print(len(tiles_paths))

import sys
import os
import tqdm

# for tiles_path, tiles_filename, tiles_name in itertools.islice(zip(tiles_paths, tiles_filenames, tiles_names),0, None, 1):
for tile_idx in range(0, len(tiles_paths), 1):
    tiles_path = tiles_paths[tile_idx]
    tiles_filename = tiles_filenames[tile_idx]
    tiles_name = tiles_names[tile_idx]
    print(tile_idx, tiles_filename)
    MainTile = os.path.basename(os.path.dirname(tiles_path)).split('_')[-1]
    dem = rd.LoadGDAL(tiles_path)
    terrain_attributes = calculate_terrain_attributes(dem)
    for key, value in terrain_attributes.items():
        dirname = os.path.join(PATHS['MERIT-Hydro'], key, f"{key}_{MainTile}")
        os.makedirs(dirname, exist_ok = True)
        save_path = os.path.join(dirname, f"{tiles_name}_{key}.tif")
        rd.SaveGDAL(save_path, value)