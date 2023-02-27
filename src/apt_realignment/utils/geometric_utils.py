import logging
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
import shapely


import zipfile

def extract_zip_files(zip_path,target_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_path)

def download_osm_building_footprint(county, state, country='USA', out_path='results'):
    place_name = "{} {} {}".format(county, state, country)
    tags = {"building": True}
    try:
        import osmnx as ox
    except ImportError as err:
        raise err
    gdf = ox.geometries_from_place(place_name, tags)
    gdf['building_geometry'] = gdf['geometry'].apply(
        lambda x: x if isinstance(x, shapely.geometry.polygon.Polygon) else None)
    bfp_gdf = gdf[['building_geometry']]
    bfp_gdf.dropna(inplace=True)
    bfp_gdf.reset_index(inplace=True)
    bfp_gdf = bfp_gdf[['osmid', 'building_geometry']]
    bfp_gdf = gpd.GeoDataFrame(bfp_gdf, geometry='building_geometry', crs="EPSG:4326")
    if out_path:
        bfp_gdf.to_file(out_path,driver='ESRI Shapefile')
    return bfp_gdf


def merge_osm_msft_data(msft_geojson, osm_geojson):
    osm_gdf = read_vector_data(osm_geojson)
    osm_bounds = box(*osm_gdf.total_bounds)

    msft_df = read_vector_data(msft_geojson)
    msft_df = msft_df.reset_index()

    osm_in_msft_df = gpd.sjoin(osm_gdf, msft_df, op='intersects', how='left')
    merged_data = osm_in_msft_df['geometry']

    def check_within_bounds(x):
        geo_polygon = None
        if osm_bounds.contains(x):
            geo_polygon = x
        return geo_polygon

    msft_geometries = msft_df.loc[~msft_df['index'].isin(osm_in_msft_df['index_right'].values)]['geometry']
    final_geometries = msft_geometries.apply(lambda x: check_within_bounds(x))
    final_gdf = merged_data.append(final_geometries[final_geometries.notna()])
    return final_gdf

def download_osm_building_footprint(place_name, state, country='USA', out_path='results'):
    tags = {"building": True}
    try:
        import osmnx as ox
    except ImportError as err:
        raise err
    gdf = ox.geometries_from_place(place_name, tags)
    gdf['building_geometry'] = gdf['geometry'].apply(
        lambda x: x if isinstance(x, shapely.geometry.polygon.Polygon) else None)
    bfp_gdf = gdf[['building_geometry']]
    bfp_gdf.dropna(inplace=True)
    bfp_gdf.reset_index(inplace=True)
    bfp_gdf = bfp_gdf[['osmid', 'building_geometry']]
    bfp_gdf = gpd.GeoDataFrame(bfp_gdf, geometry='building_geometry', crs="EPSG:4326")
    return bfp_gdf


def df_2_geovector(data_frame, vector, prefix='geometry', outdir='artifact', vector_format='shp'):
    gdf = gpd.GeoDataFrame(data_frame[vector], crs="epsg:4326", geometry=vector)
    gdf.to_crs("epsg:4326")
    out_dir = os.path.join(outdir, vector)
    if not os.path.isdir(out_dir):
        try:
            os.mkdir(outdir)
        except OSError as err:
            logging.error("{}".format(err))

    if vector_format == 'shp':
        gdf.to_file(os.path.join(out_dir, "{}_{}.shp".format(prefix, vector)), driver='ESRI Shapefile')
    else:
        gdf.to_file(os.path.join(out_dir, "{}_{}.geojson".format(prefix, vector)), driver="GeoJSON")
    return


def save_geodataframe(apt_df, column_name='updated_geometries',ext='pkl', out_dir='results', filename='APT_data'):
    gdf = gpd.GeoDataFrame(apt_df, geometry=column_name, crs="EPSG:4326")
    pd_dataframe = pd.DataFrame(gdf)
    os.makedirs(out_dir, exist_ok=True)
    if ext =='pkl':
        pd_dataframe.to_pickle(os.path.join(out_dir, filename + '.pkl'))
    if ext=='shp':
        gdf.to_file(driver='ESRI Shapefile', filename=os.path.join(out_dir, filename + '.shp'))
    return os.path.join(out_dir, filename + '.shp')


def geojson2shpfile(geojson_file, crs='epsg:4326', verbose=0):
    logger = create_logger()
    if verbose:
        logger.info("Reading {} ...".format(os.path.basename(geojson_file)))
    srcDS = gpd.read_file(geojson_file)
    if verbose:
        logger.info("converting to {}".format(crs))
    srcDS = srcDS.to_crs(crs)
    srcDS.to_file(geojson_file.split('.')[0] + '.shp', driver="ESRI Shapefile")
    return 0


def read_vector_data(vector_file):
    if not os.path.isfile(vector_file):
        logging.error("{} file not found".format(vector_file))
        return
    vector_df = gpd.read_file(vector_file)
    vector_df = gpd.GeoDataFrame(vector_df, crs="EPSG:4326", geometry='geometry')
    vector_df = vector_df.to_crs("epsg:4326")
    return vector_df


def create_logger():
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:- %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("APT_Realignment")
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def prep_polygons_asarr(gs):
    def get_pts(poly):
        if isinstance(poly, shapely.geometry.Polygon):
            coords = np.array(poly.exterior.coords)
        elif isinstance(poly, shapely.geometry.MultiPolygon):
            coords = np.concatenate([get_pts(sp) for sp in poly.geoms])
        return coords

    return [get_pts(poly) for poly in gs]


def get_nearest_poly(pt, polys):
    polys = prep_polygons_asarr(polys)
    dists = np.array([np.abs(np.linalg.norm(poly - pt, axis=1)).min() for poly in polys])
    return dists.argmin()
