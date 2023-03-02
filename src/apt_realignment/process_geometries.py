import argparse
import logging
import os
import os.path as osp
import warnings
from glob import glob

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
import time

warnings.filterwarnings("ignore")

from utils.geometric_utils import create_logger, get_nearest_poly, save_geodataframe, extract_zip_files
from utils.haversine_distance import get_distance
from utils.extract_mnr_data import ExtractMNRData
from spartial_preprocessing import process_parcel_bfp_data as process_bfp_parcel
from spartial_preprocessing import process_apt_bfp_data as process_bfp_apt

geod = Geod(ellps="WGS84")
logger = create_logger()


class ImproveAPAScore:
    def __init__(self, building_shapefile, mnr_apt_data, output_path=None):

        self.out_path = output_path
        self.__land_parcels = None
        self.__building_footprints = building_shapefile
        self.anchorpoint_df = mnr_apt_data
        self.apt_df_columns = list(self.anchorpoint_df.columns)
        self.req_columns = ['feat_id', 'PRCLDMPID', 'iso_script', 'iso_lang_code', 'postal_code', 'house_number',
                            'state_province_code', 'country_code', 'street_name', 'locality', 'prefix', 'suffix',
                            'predir', 'postdir', 'sn_body', 'apt_geometry']

    def process_with_bfp_parcel(self, parcel_shapefile, bfp_per_parcel=1, save_df=True, filename='APT_realigned'):
        self.__land_parcels = parcel_shapefile
        logger.info("Processing APT's with Parcel data {} BFP-Count Per Parcel".format(bfp_per_parcel))
        bfp_intersection_parcel_df = process_bfp_parcel.bfp_parcel_overlap(bfp_path=self.__building_footprints,
                                                                           parcel_path=parcel_shapefile)
        df_bfp_within_parcels = process_bfp_parcel.get_buildings_within_parcel(bfp_intersection_parcel_df,
                                                                               count=bfp_per_parcel)
        process_df = process_bfp_parcel.get_apt_within_parcel(self.anchorpoint_df, df_bfp_within_parcels)
        del df_bfp_within_parcels
        if bfp_per_parcel >= 2:
            process_df = process_df.groupby(['PRCLDMPID'], as_index=False).apply(
                lambda x: pd.Series(self.process_apt_with_multi_bfp(x, bfp_count=bfp_per_parcel)))

        process_df['APT_bfp_parcel'] = process_df['building_roi'].apply(lambda x: x.centroid)  # change to bfp edge
        process_df['apt_bfp_dist'] = process_df.apply(lambda x: self.get_apt_to_bfp_distance(x), axis=1)
        process_df['apt_on_bfp'] = process_df.apply(lambda x: self.apt_at_rooftop(x), axis=1)

        cols = self.req_columns + ['apt_bfp_dist', 'apt_on_bfp', 'APT_bfp_parcel']
        filter_df = process_df[cols]
        del process_df

        if save_df:
            logger.info("saving {} data points ".format(filter_df.shape[0]))
            dirname = "{}_updated_apt_{}_bfp".format(osp.basename(parcel_shapefile).split('.')[0], str(bfp_per_parcel))
            result_path = osp.join(self.out_path, dirname)
            save_geodataframe(filter_df, column_name='APT_bfp_parcel'
                              , out_dir=result_path, filename='updated_anchor_points')
            logger.info("File saved as {}.pkl".format())
            del filter_df
        return 0

    def process_with_bfp(self, thresh_distance=10, max_distance=50,out_fname="updated_apts_wih_bfp"):
        query_table_df = self.anchorpoint_df  # gpd.GeoDataFrame(self.anchorpoint_df[['feat_id', 'apt_geometry']], geometry='apt_geometry', crs=4326)
        query_table_df = query_table_df.to_crs(3857)
        # query_table_df = query_table_df.sample(1000)  ## create sample
        logger.info("Extracting Nearest Building footprints(in meters) from Database..")

        nearest_bfp_df = process_bfp_apt.get_nearest_building_to_apt(anchor_point_gdf=query_table_df)
        del query_table_df
        logger.info("Nearest BFPs acquired. Processing for multiple scenarios..")

        # Remove outliers AnchorPoint to BFP distance above 50m
        outlier_index = nearest_bfp_df.loc[nearest_bfp_df['apt_distance'] > max_distance].index
        nearest_bfp_df.drop(outlier_index, inplace=True)

        apt_on_bfp_df = process_bfp_apt.process_apt_on_bfp(nearest_bfp_df)
        apt_on_bfp_df = pd.merge(self.anchorpoint_df, apt_on_bfp_df, on='feat_id')
        apt_on_bfp_df['updated_APT_with_BFP'] = apt_on_bfp_df['apt_geometry'].apply(lambda x: x)

        # Append both dataframes
        apt_not_on_bfp = nearest_bfp_df.loc[nearest_bfp_df['apt_intersects'] == False]
        del nearest_bfp_df

        apt_near_bfp_df = process_bfp_apt.process_apt_within_bfp(apt_not_on_bfp,
                                                                 thresh_distance=thresh_distance)  # meters
        apt_near_bfp_df = pd.merge(self.anchorpoint_df, apt_near_bfp_df, on='feat_id')  # Merge apt not on bfp

        # merge data frames with APT on BFP
        apts_on_bfp_df = apt_on_bfp_df.append(apt_near_bfp_df, ignore_index=True)
        del apt_near_bfp_df
        del apt_on_bfp_df

        # Anchor Points in range 10-50m
        apt_far_from_bfp = apt_not_on_bfp.loc[apt_not_on_bfp['apt_distance'] > 10]
        apt_far_from_bfp = pd.merge(self.anchorpoint_df, apt_far_from_bfp, on='feat_id')
        apt_far_from_bfp = apt_far_from_bfp.to_crs(3857)

        bfp_far_off_apt = process_bfp_apt.get_nearest_building_to_apt(anchor_point_gdf=apt_far_from_bfp,
                                                                      near_bfp=2)
        bfp_far_off_apt = process_bfp_apt.process_apt_outside_bfp(bfp_far_off_apt)
        bfp_far_off_apt = bfp_far_off_apt[
            ~bfp_far_off_apt['apt_building_id'].isin(apts_on_bfp_df['apt_building_id'])]


        bfp_far_off_apt['updated_APT_with_BFP'] = bfp_far_off_apt['apt_building_geometry'].apply(lambda x: x.centroid)
        final_df = apts_on_bfp_df.append(bfp_far_off_apt, ignore_index=True)
        logger.info("Saving processed data as pkl {}.pkl".format(out_fname))
        save_geodataframe(final_df, column_name='updated_APT_with_BFP', ext='pkl', out_dir=self.out_path,
                          filename=out_fname)

    def process_with_ppa(self):
        pass

    def process_with_stimage(self):
        pass

    def apt_at_rooftop(self, data: gpd.GeoSeries):
        building_roi = data['building_roi']
        anchor_point = data['apt_geometry']
        return self.check_point_on_polygon(bfp_polygon=building_roi, apt_point=anchor_point)

    def get_apt_to_bfp_distance(self, data):
        anchor_point = data['apt_geometry']
        bfp_centroid = data['APT_bfp_parcel']
        return get_distance(anchor_point, bfp_centroid)

    def process_apt_with_multi_bfp(self, x, area_thresh=150, bfp_count=2):
        ret = dict()
        for cols in self.req_columns:
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
                        list(x['building_roi'][:2])[get_nearest_poly(list(x['apt_geometry'])[0], building_polygons)])
            elif area_diff <= 0:
                if np.abs(area_diff) > area_thresh:
                    ret['building_roi'] = building_polygons[1]
                else:
                    ret['building_roi'] = (
                        list(x['building_roi'][:2])[get_nearest_poly(list(x['apt_geometry'])[0], building_polygons)])
        if bfp_count > 2:
            building_polygons = list(x['building_roi'][:3])
            mnr_apt = list(x['apt_geometry'])[0]
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
    parcels_data = list()
    db_schema = args.schema

    pkl_file = [os.path.join(args.path, pkl_file) for pkl_file in os.listdir(args.path) if pkl_file.startswith('APT')]
    try:
        logger.info("{} APT data found. Accessing file".format(len(pkl_file)))
        mnr_df = pd.read_pickle(pkl_file.pop())
        mnr_geo_dataframe = gpd.GeoDataFrame(mnr_df, geometry='apt_geometry', crs="EPSG:4326")
        logger.info("APT data found with {} data points".format(mnr_df.shape[0]))
    except IndexError as err:
        logger.error("{} Pkl file not available.Extracting from MNR Database".format(err))
        stime = time.time()
        mnr_database = ExtractMNRData(country_code=db_schema)
        mnr_database.connect_to_server()
        mnr_geo_dataframe = mnr_database.extract_apt_addresses_data()
        logger.info("APT dataframe created..{}".format(round(time.time()-stime),2))

    parcel_path = osp.join(args.path, 'parcel_data')
    if osp.isdir(parcel_path):
        parcels_data = [pzip for pzip in os.listdir(parcel_path) if pzip.endswith('.zip')]

    apt_preprocess = ImproveAPAScore(building_shapefile=osp.join(args.path, args.msft_bfp),
                                     mnr_apt_data=mnr_geo_dataframe,
                                     output_path=osp.join(args.path, args.out)
                                     )
    if len(parcels_data) > 0:
        logger.info("Number of Parcel data found {}:".format(len(parcels_data)))
        for parcel_file in parcels_data:
            if parcel_file.endswith('.zip'):
                logger.info("Running process for : {}".format(parcel_file))
                target_path = osp.join(parcel_path, parcel_file.split('.')[0])
                try:
                    extract_zip_files(osp.join(parcel_path, parcel_file), target_path=target_path)
                except IOError as err:
                    logging.error('{}! for parcel {}'.format(err, parcel_file))
                    continue
                parcel_shp = osp.join(target_path, parcel_file.replace('.zip', '.shp'))
                apt_preprocess.process_with_bfp_parcel(parcel_shapefile=parcel_shp, bfp_per_parcel=1)
                apt_preprocess.process_with_bfp_parcel(parcel_shapefile=parcel_shp, bfp_per_parcel=2)
                # apt_preprocess.process_apt_with_bfp_parcel(parcel_shapefile=parcel_shp, bfp_per_parcel=3)
    else:
        apt_preprocess.process_with_bfp(out_fname=args.fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     'Create Realigned Matrix of APTs using MNR APT databse, DMP Parcels '
                                     'and Building footprint')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to the data')
    parser.add_argument('-s', '--schema', type=str, required=True, help='name of the schema to run APT_process')
    parser.add_argument('-m', '--msft_bfp', type=str, required=True, help='name of the BFP file in geojson/shp')
    parser.add_argument('-o', '--out', type=str, default='mnr_apt_data', help='output path')
    parser.add_argument('-f', '--fname', type=str, default='updated_APT_matrix', help="output filename")
    args = parser.parse_args()
    main(args)
