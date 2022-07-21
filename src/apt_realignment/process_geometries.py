import logging
import os
import warnings

import geopandas as gpd
import shapely
from pandarallel import pandarallel
from shapely.geometry import Point
from utils.haversine_distance import get_distance
from utils.geometric_utils import read_vector_data

warnings.filterwarnings("ignore")


class ProcessGeometricData:
    def __init__(self, parcel_shapefile, building_shapefile, apt_shape_file, output_path=None):
        self.outpath = output_path

        self._land_parcels = parcel_shapefile
        self._building_footprints = building_shapefile
        self._anchor_point = apt_shape_file

    def process_dataframe(self, bfp_count_per_parcel=1):
        bfp_intersection_parcel_df = self.get_bfp_parcel_ovelap()
        bfp_intersection_parcel_df['building_centroid'] = bfp_intersection_parcel_df['building_roi'].parallel_apply(
            lambda x: x.centroid)
        df_parcel_single_bfp = self.get_buildings_within_parcel(bfp_intersection_parcel_df, count=bfp_count_per_parcel)
        processed_df = self.get_parcel_anchorpoints(df_parcel_single_bfp)

        processed_df['APT_on_Building_footprint'] = processed_df.apply(lambda x: self.apt_at_rooftop(x),axis=1)
        processed_df['APT_Centroid_distance'] = processed_df.apply(lambda x: self.get_apt_to_bfp_distance(x),axis=1)

        return processed_df

    def get_nearest_bfp_to_apt(self,data):
        anchor_point = data['APT']
        building_polygon = data['building_roi']



    def get_apt_to_bfp_distance(self,data):
        anchor_point = data['APT']
        bfp_centroid = data['building_centroid']
        return get_distance(anchor_point, bfp_centroid)

    def get_bfp_parcel_ovelap(self):
        land_parcel_df = read_vector_data(self._land_parcels)
        footprint_df = read_vector_data(self._building_footprints)
        building_within_parcel_df = gpd.sjoin(land_parcel_df, footprint_df, op='intersects', how='left')
        building_within_parcel_df = building_within_parcel_df.dropna()  # drop columns with no Buildings

        def _get_buildingfootprint(val):
            return footprint_df['geometry'].loc[val]

        def get_building_roi(data: gpd.GeoSeries):
            building_polygon = data['building_geometry']
            parcel_polygon = data['geometry']
            building_roi = None
            try:
                if building_polygon.area > parcel_polygon.area:
                    building_roi = parcel_polygon.intersection(building_polygon)
                else:
                    building_roi = building_polygon
            except:
                logging.error("error for {},{}".format(building_polygon, parcel_polygon))
            return building_roi

        building_within_parcel_df['building_geometry'] = building_within_parcel_df['index_right'].parallel_apply(
            lambda x: _get_buildingfootprint(x))
        building_within_parcel_df['building_roi'] = building_within_parcel_df.parallel_apply(
            lambda x: get_building_roi(x), axis=1)
        building_within_parcel_df = building_within_parcel_df.drop(['index_right', 'capture_dates_range', 'release'],
                                                                   axis=1)
        building_within_parcel_df = building_within_parcel_df.dropna()
        return building_within_parcel_df

    def get_buildings_within_parcel(self, data: gpd.GeoSeries, count=None):
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

    def get_parcel_anchorpoints(self, input_dataframe: gpd.GeoSeries):
        anchorpoint_df = read_vector_data(self._anchor_point)
        # Read anchor points - Parcel centroid or Address building
        anchorpoint_df = anchorpoint_df.loc[(anchorpoint_df['AnchorPoin'] == 'AddressParcelCentroid') | (
                anchorpoint_df['AnchorPoin'] == 'AddressBuilding')]
        # find spatial join of input_dataframe with anchorpoint
        grouped_df = gpd.sjoin(input_dataframe, anchorpoint_df, op='contains', how='inner')

        def _get_apt_point(val):
            return anchorpoint_df['geometry'].loc[val]

        grouped_df['APT'] = grouped_df['index_right'].parallel_apply(lambda x: _get_apt_point(x))
        grouped_df = grouped_df.drop(['index_right'], axis=1)
        return grouped_df

    def apt_at_rooftop(self, data: gpd.GeoSeries):
        building_roi = data['building_roi']
        anchor_point = data['APT']
        return self.check_point_on_polygon(bfp_polygon=building_roi, apt_point=anchor_point)

    @staticmethod
    def check_point_on_polygon(bfp_polygon, apt_point):
        flag = False
        if bfp_polygon.contains(apt_point):
            flag = True
        return flag