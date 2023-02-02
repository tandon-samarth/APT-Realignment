import argparse

from glob import glob
import geopandas as gpd
import logging
import numpy as np
import os
import os.path as osp
import pandas as pd

import shapely.geos
from pyproj import Geod
import time
import warnings

warnings.filterwarnings("ignore")

from utils.geometric_utils import read_vector_data, create_logger, get_nearest_poly, save_geodataframe, \
    extract_zip_files
from utils.haversine_distance import get_distance

geod = Geod(ellps="WGS84")
logger = create_logger()


class ProcessGeometricData:
    def __init__(self, building_shapefile, apt_shape_file, output_path=None, meta_info=False):

        self.out_path = output_path

        self.__land_parcels = None
        self.__building_footprints = building_shapefile

        self.apt_df_columns = list()
        self.__meta_data__ = meta_info

        logger.info("Reading {}".format(osp.basename(apt_shape_file)))
        self.anchorpoint_df = read_vector_data(apt_shape_file)

    def process_dataframe(self, parcel_shapefile, complexity=1, save_df=True, filename='APT_realigned'):
        self.__land_parcels = parcel_shapefile
        logger.info("Starting Process with complexity upto {} BFP-Count Per Parcel".format(complexity))
        bfp_intersection_parcel_df = self.get_bfp_parcel_overlap()

        s_time = time.time()
        for bfp_count_per_parcel in range(1, complexity + 1):
            logger.info("Processing with complexity {} BFP-Count Per Parcel".format(bfp_count_per_parcel))
            df_parcel_within_bfp = self.get_buildings_within_parcel(bfp_intersection_parcel_df,
                                                                    count=bfp_count_per_parcel)
            process_df = self.get_parcel_anchorpoints(df_parcel_within_bfp)  # read Anchor points Data

            if bfp_count_per_parcel >= 2:
                process_df = process_df.groupby(['PRCLDMPID'], as_index=False).apply(
                    lambda x: pd.Series(self.multi_map(x, bfp_count=bfp_count_per_parcel)))

            process_df['updated_geometries'] = process_df['building_roi'].apply(lambda x: x.centroid)
            process_df['apt_bfp_dist'] = process_df.apply(lambda x: self.get_apt_to_bfp_distance(x), axis=1)
            req_columns = self.apt_df_columns + ['PRCLDMPID', 'updated_geometries', 'apt_bfp_dist']

            filter_df = process_df[req_columns]
            filter_df = filter_df.drop(['geometry'], axis=1)

            if self.__meta_data__:
                process_df['APT_on_Building_footprint'] = process_df.apply(lambda x: self.apt_at_rooftop(x), axis=1)
                process_df['apt_bfp_dist'] = process_df.apply(lambda x: self.get_apt_to_bfp_distance(x), axis=1)

            if save_df:
                logger.info("saving {} data points ".format(filter_df.shape[0]))
                dirname = "updated_geometries_bfp-count_" + str(bfp_count_per_parcel) + '_' + \
                          osp.basename(self.__land_parcels).split('.')[0]
                result_path = osp.join(self.out_path, dirname)
                os.makedirs(result_path, exist_ok=True)
                save_geodataframe(filter_df, shp=True, out_dir=result_path, filename=filename)

                logger.info(
                    "File saved at {}".format(os.path.join(result_path, filename + str(complexity) + '.pkl')))

            del df_parcel_within_bfp
            del process_df
            del filter_df

        time_delta = round((time.time() - s_time) / 60, 2)
        logger.info("Total time to complete the process: {} min.".format(time_delta))
        return

    def get_apt_to_bfp_distance(self, data):
        anchor_point = data['APT']
        bfp_centroid = data['updated_geometries']
        return get_distance(anchor_point, bfp_centroid)

    def get_bfp_parcel_overlap(self):
        logger.info("Reading Building Footprint {}".format(osp.basename(self.__building_footprints)))
        footprint_df = read_vector_data(self.__building_footprints)
        logger.info("reading Land Parcels {}".format(osp.basename(self.__land_parcels)))
        land_parcel_df = read_vector_data(self.__land_parcels)

        logger.info("Processing Land Parcel data and Building Footprints ")
        building_within_parcel_df = gpd.sjoin(land_parcel_df, footprint_df, op='intersects', how='left')
        building_within_parcel_df = building_within_parcel_df.dropna()  # drop columns with no Buildings

        def __get_buildingfootprint(val):
            return footprint_df['geometry'].loc[val]

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
        del footprint_df

        building_within_parcel_df['building_roi'] = building_within_parcel_df.apply(
            lambda x: __get_building_roi(x), axis=1)
        building_within_parcel_df = building_within_parcel_df.drop(['index_right'], axis=1)
        building_within_parcel_df = building_within_parcel_df.dropna()

        return building_within_parcel_df

    def get_buildings_within_parcel(self, data: gpd.GeoSeries, count=None):
        logger.info("Acquiring BFP's within land Parcels ".format(count))
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
        self.apt_df_columns = list(self.anchorpoint_df.columns)
        logger.info("Processing Anchor-Points and Parcel-Building Geo-Dataframe")
        # find spatial join of input_dataframe with anchorpoint
        grouped_df = gpd.sjoin(input_dataframe, self.anchorpoint_df, op='contains', how='inner')

        def _get_apt_point(val):
            return self.anchorpoint_df['geometry'].loc[val]

        grouped_df['APT'] = grouped_df['index_right'].apply(lambda x: _get_apt_point(x))
        grouped_df = grouped_df.drop(['index_right'], axis=1)
        return grouped_df

    def apt_at_rooftop(self, data: gpd.GeoSeries):
        building_roi = data['building_roi']
        anchor_point = data['APT']
        return self.check_point_on_polygon(bfp_polygon=building_roi, apt_point=anchor_point)

    def multi_map(self, x, area_thresh=150, bfp_count=2):
        ret = dict()
        req_columns = ['PRCLDMPID', 'building_roi', 'APT'] + self.apt_df_columns
        for cols in req_columns:
            ret[cols] = x.iloc[0][cols]

        if bfp_count == 2:
            building_polygons = list(x['building_roi'][:2])
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
        if bfp_count > 2:
            building_polygons = list(x['building_roi'][:3])
            mnr_apt = list(x['APT'])[0]
            point_on_polygon = False

            for polygon in building_polygons:
                if self.check_point_on_polygon(polygon, mnr_apt):
                    point_on_polygon = True
            if not point_on_polygon:
                ret['building_roi'] = (list(x['building_roi'][:3])[get_nearest_poly(mnr_apt, building_polygons)])
        return ret

    @staticmethod
    def check_point_on_polygon(bfp_polygon, apt_point):
        flag = False
        if bfp_polygon.contains(apt_point):
            flag = True
        return flag

    @staticmethod
    def merge_data(apt_realignment_dir_path):
        data = []
        pkl_files = glob(apt_realignment_dir_path + '/*/*.pkl')
        for pkl in pkl_files:
            df = pd.read_pickle(pkl)
            df = df.reset_index(drop=True)
            data.append(df)
        merge_df = pd.concat(data)
        merge_df.to_pickle(os.path.join(apt_realignment_dir_path, 'FinalUpdated_APT.pkl'))


def main(args):
    parcel_path = osp.join(args.path, 'parcel_data')
    parcels_data = [parcels for parcels in os.listdir(parcel_path) if
                    not osp.isdir(osp.join(parcel_path, parcels.split('.')[0]))]
    logger.info("number of Parcel data found {}:".format(len(parcels_data)))
    apt_preprocess = ProcessGeometricData(building_shapefile=osp.join(args.path, args.msft_bfp),
                                          apt_shape_file=osp.join(args.path, args.schema),
                                          output_path=osp.join(args.path, args.out)
                                          )
    for parcels in parcels_data:
        if parcels.endswith('.zip'):
            target_path = osp.join(parcel_path, parcels.split('.')[0])
            logger.info("processing :{}".format(parcels))
            try:
                extract_zip_files(osp.join(parcel_path, parcels), target_path=target_path)
                parcel_shp = osp.join(target_path, parcels.replace('.zip', '.shp'))
                apt_preprocess.process_dataframe(parcel_shapefile=parcel_shp, complexity=2)
            except IOError as err:
                logging.info("{}! for parcel {} ".format(err, parcels))
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     'Create Realigned Matrix of APTs using MNR APT databse, DMP Parcels '
                                     'and Building footprint')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to the data')
    parser.add_argument('-s', '--schema', type=str, required=True, help='name of the scema')
    parser.add_argument('-m', '--msft_bfp', type=str, required=True, help='name of the BFP file in geojson/shp')
    parser.add_argument('-o', '--out', type=str, default='Apt_realignment_MSFT', help='output path')
    args = parser.parse_args()
    main(args)
