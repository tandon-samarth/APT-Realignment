import geopandas as gpd
import logging
import numpy as np
import os
import os.path as osp
import pandas as pd
import shapely
from utils.geometric_utils import read_vector_data


def bfp_parcel_overlap(bfp_path, parcel_path):
    """
    Function gets spatial-join/Intersection between Building Footprints and Land Parcel
    data
    bfp_path= str(BFP file path)
    parcel_path = str(Parcel data file path)
    """
    bfp_df = read_vector_data(bfp_path)
    land_parcel_df = read_vector_data(parcel_path)
    building_within_parcel_df = gpd.sjoin(land_parcel_df, bfp_df, op='intersects', how='left')
    building_within_parcel_df = building_within_parcel_df.dropna()  # drop columns with no Buildings

    def __get_buildingfootprint(val):
        return bfp_df['geometry'].loc[val]

    def __get_building_roi(data: gpd.GeoSeries):
        building_polygon = data['building_geometry']
        parcel_polygon = data['geometry']
        building_roi = None
        try:
            if building_polygon == np.nan:
                building_roi = parcel_polygon
            if building_polygon.area > parcel_polygon.area:
                building_roi = parcel_polygon.intersection(building_polygon)
            else:
                building_roi = building_polygon
        except shapely.geos.TopologicalError as err:
            logging.error("{} for {}".format(err, building_polygon))
        return building_roi

    building_within_parcel_df['building_geometry'] = building_within_parcel_df['index_right'].apply(
        lambda x: __get_buildingfootprint(x))
    del bfp_df

    building_within_parcel_df['building_roi'] = building_within_parcel_df.apply(
        lambda x: __get_building_roi(x), axis=1)
    building_within_parcel_df = building_within_parcel_df.drop(['index_right'], axis=1)
    building_within_parcel_df = building_within_parcel_df.dropna()
    return building_within_parcel_df


def get_buildings_within_parcel(data: gpd.GeoSeries, count=None):
    """
    The function finds out number of buildings within a Parcel for ex.
    One building within a Parcel (Row house) or multiple buildings within a Parcel(society)
    data: Dataframe BFP intersection Parcel
    count: BFP count in Parcel
    return: dataframe with n buildings within parcel where n is count
    """
    building_within_parcel_count = data.groupby('PRCLDMPID')['geometry'].count()

    if count == 1:
        parcel_ids_with_one_building = list(building_within_parcel_count[building_within_parcel_count == 1].keys())
        filtered_dataframe = data[data['PRCLDMPID'].isin(parcel_ids_with_one_building)]

    elif count == 2:
        parcel_ids_with_two_buildings = list(building_within_parcel_count[building_within_parcel_count == 2].keys())
        filtered_dataframe = data[data['PRCLDMPID'].isin(parcel_ids_with_two_buildings)]

    else:
        parcel_ids_with_n_buildings = list(building_within_parcel_count[building_within_parcel_count > 2].keys())
        filtered_dataframe = data[data['PRCLDMPID'].isin(parcel_ids_with_n_buildings)]
    return filtered_dataframe

def get_apt_within_parcel(apt_dataframe,bfp_parcel_dataframe):
    apt_df_columns = list(apt_dataframe.columns)
    grouped_df = gpd.sjoin(bfp_parcel_dataframe, apt_dataframe, op='contains', how='inner')

    def _get_apt_point(val):
        return apt_dataframe['apt_geometry'].loc[val]

    grouped_df['apt_geometry'] = grouped_df['index_right'].apply(lambda x: _get_apt_point(x))
    grouped_df = grouped_df.drop(['index_right'], axis=1)
    return grouped_df



