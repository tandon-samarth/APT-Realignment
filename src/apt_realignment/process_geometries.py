import logging
import os
import warnings
import os.path as osp
import geopandas as gpd
import pandas as pd
import numpy as np
from utils.haversine_distance import get_distance
from utils.geometric_utils import read_vector_data, create_logger, get_nearest_poly

from pyproj import Geod

geod = Geod(ellps="WGS84")
warnings.filterwarnings("ignore")


class ProcessGeometricData:
    def __init__(self, parcel_shapefile, building_shapefile, apt_shape_file, output_path=None, meta_info=False):
        self.out_path = output_path

        self.__land_parcels = parcel_shapefile
        self.__building_footprints = building_shapefile
        self.__anchor_points_data = apt_shape_file

        self.apt_df_columns = list()
        self.__meta_data__ = meta_info
        self.__logger = create_logger()

    def process_dataframe(self, bfp_count_per_parcel=1, save_df=True, filename='APT_realigned'):
        self.__logger.info("Processing APT's with {} BFP-Count Per Parcel".format(bfp_count_per_parcel))

        bfp_intersection_parcel_df = self.get_bfp_parcel_overlap()

        df_parcel_within_bfp = self.get_buildings_within_parcel(bfp_intersection_parcel_df, count=bfp_count_per_parcel)
        # read Anchor points Data
        process_df = self.get_parcel_anchorpoints(df_parcel_within_bfp)

        if bfp_count_per_parcel == 2:
            process_df = process_df.groupby(['PRCLDMPID'], as_index=False).apply(
                lambda x: pd.Series(self.process_multi_mapping(x)))

        req_columns = self.apt_df_columns + ['PRCLDMPID','updated_geometries']
        process_df['updated_geometries'] = process_df['building_roi'].apply(lambda x: x.centroid)

        print(process_df.keys)

        filter_df = process_df[req_columns]
        filter_df = filter_df.drop(['geometry'],axis=1)

        filter_df['updated_lat'] = filter_df['updated_geometries'].apply(lambda z: z.y)
        filter_df['updated_lon'] = filter_df['updated_geometries'].apply(lambda z: z.x)

        if self.__meta_data__:
            process_df['APT_on_Building_footprint'] = process_df.apply(lambda x: self.apt_at_rooftop(x), axis=1)
            process_df['APT_to_Centroid_distance'] = process_df.apply(lambda x: self.get_apt_to_bfp_distance(x),
                                                                      axis=1)
        if save_df:
            self.__logger.info("saving Realigned matrix..")
            result_path = osp.join(self.out_path, 'APT_realign_{}_bfp2parcel'.format(bfp_count_per_parcel))
            os.makedirs(result_path, exist_ok=True)
            geo_dataframe = gpd.GeoDataFrame(filter_df, geometry='updated_geometries', crs="EPSG:4326")

            pd_dataframe = pd.DataFrame(geo_dataframe)
            pd_dataframe.to_pickle(os.path.join(result_path, filename + '.pkl'))
            self.__logger.info("File saved at {}".format(os.path.join(result_path, filename + '.pkl')))
        return process_df

    def get_apt_to_bfp_distance(self, data):
        anchor_point = data['APT']
        bfp_centroid = data['updated_APT']
        return get_distance(anchor_point, bfp_centroid)

    def get_bfp_parcel_overlap(self):
        self.__logger.info("Processing Land Parcel data and Building Footprints ")
        if not osp.isfile(self.__land_parcels) and osp.isfile(self.__building_footprints):
            self.__logger("File path not correct ")
            raise
        land_parcel_df = read_vector_data(self.__land_parcels)
        footprint_df = read_vector_data(self.__building_footprints)
        building_within_parcel_df = gpd.sjoin(land_parcel_df, footprint_df, op='intersects', how='left')
        building_within_parcel_df = building_within_parcel_df.dropna()  # drop columns with no Buildings

        def __get_buildingfootprint(val):
            return footprint_df['geometry'].loc[val]

        def __get_building_roi(data: gpd.GeoSeries):
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

        building_within_parcel_df['building_geometry'] = building_within_parcel_df['index_right'].apply(
            lambda x: __get_buildingfootprint(x))
        building_within_parcel_df['building_roi'] = building_within_parcel_df.apply(
            lambda x: __get_building_roi(x), axis=1)
        building_within_parcel_df = building_within_parcel_df.drop(['index_right', 'capture_dates_range', 'release'],
                                                                   axis=1)
        building_within_parcel_df = building_within_parcel_df.dropna()
        return building_within_parcel_df

    def get_buildings_within_parcel(self, data: gpd.GeoSeries, count=None):
        self.__logger.info("Acquiring BFP's within land Parcels ".format(count))
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
        self.__logger.info("Processing Anchor-Points data over Parcel-Building Geo-Dataframe")
        if not osp.isfile(self.__anchor_points_data):
            self.__logger("{} Path not found".format(self.__anchor_points_data))
        anchorpoint_df = read_vector_data(self.__anchor_points_data)
        self.apt_df_columns = list(anchorpoint_df.columns)
        # find spatial join of input_dataframe with anchorpoint
        grouped_df = gpd.sjoin(input_dataframe, anchorpoint_df, op='contains', how='inner')

        def _get_apt_point(val):
            return anchorpoint_df['geometry'].loc[val]

        grouped_df['APT'] = grouped_df['index_right'].apply(lambda x: _get_apt_point(x))
        grouped_df = grouped_df.drop(['index_right'], axis=1)
        return grouped_df

    def apt_at_rooftop(self, data: gpd.GeoSeries):
        building_roi = data['building_roi']
        anchor_point = data['APT']
        return self.check_point_on_polygon(bfp_polygon=building_roi, apt_point=anchor_point)

    def process_multi_mapping(self, x, area_thresh=150):
        print("processing multi-mapping")
        ret = dict()
        req_columns = ['PRCLDMPID', 'building_roi', 'APT'] + self.apt_df_columns
        building_polygons = list(x['building_roi'][:2])
        for cols in req_columns:
            ret[cols] = x.iloc[0][cols]
        geo_area = list(x['building_roi'].apply(lambda poly: abs(geod.geometry_area_perimeter(poly)[0])))[:2]
        area_diff = geo_area[0] - geo_area[1]
        if area_diff > 0:
            if area_diff > area_thresh:
                ret['building_roi'] = building_polygons[0]
            else:
                ret['building_roi'] = (
                    list(x['building_roi'][:2])[get_nearest_poly(list(x['APT'])[0], building_polygons)])
        elif area_diff <= 0:
            if np.abs(area_diff) > area_thresh:
                ret['building_roi'] = building_polygons[1]
            else:
                ret['building_roi'] = (
                    list(x['building_roi'][:2])[get_nearest_poly(list(x['APT'])[0], building_polygons)])
        return ret

    @staticmethod
    def check_point_on_polygon(bfp_polygon, apt_point):
        flag = False
        if bfp_polygon.contains(apt_point):
            flag = True
        return flag


if __name__ == '__main__':
    data_path = "/mnt/c/Users/tandon/OneDrive - TomTom/Desktop/tomtom/Workspace/01_Rooftop_accuracy/BFP_Analysis_USA/data/data"
    state = "California"
    city = "SantaClara"
    apt_data_path = osp.join(data_path, state, "APT_2022_06_010_nam_usa_uca.shp")
    parcel_path = osp.join(data_path, state, city, "Parcels_06085/Parcels_06085.shp")
    building_geojson = osp.join(data_path, state, 'California.geojson')

    apt_preprocess = ProcessGeometricData(parcel_shapefile=parcel_path, building_shapefile=building_geojson,
                                          apt_shape_file=apt_data_path)
    processed_df = apt_preprocess.process_dataframe(bfp_count_per_parcel=2)
