import os
import sys

from shapely import wkt
from shapely.geometry import Point, MultiPoint
import geopandas as gpd
import psycopg2
import logging
import pandas as pd
import numpy as np
import time
from sqlalchemy import create_engine
from geometric_utils import create_logger

try:
    import pycoredb

    pycoredb_client = "http://cpp-read.maps-contentops.amiefarm.com/cdapps/coredb-main-ws/"

except ImportWarning as warning:
    logging.warning("{} Pycoredb not installed.".format(warning))

"""
Get details from http://specs.tomtomgroup.com/specification/mnr-r_daily/internal/mnr_support_spec/mnr_user_guide/getting_started/mnr_custom_tables/custom_tables_2.html
"""


class ExtractMNRData:
    def __init__(self, country_code, username="mnr_ro", password="mnr_ro", database_name='mnr',
                 databse_server='caprod-cpp-pgmnr-002.flatns.net'):
        self.__username__ = username
        self.__password__ = password
        self.__logger__ = create_logger()
        self.country_code = country_code
        self.host = databse_server
        self.database_name = database_name
        self.connection = self.connect_to_server()
        if self.connection:
            self.__logger__.info("Data base Connection Successful")

    def connect_to_server(self):
        """
        Connect to MNR Database
        :return: connection object
        """
        connection = None
        alchemyengine = create_engine(
            f'postgresql+psycopg2://{self.__username__}:{self.__password__}@{self.host}/{self.database_name}')
        try:
            connection = alchemyengine.connect()
        except (Exception, psycopg2.DatabaseError) as error:
            self.__logger__.error("{}".format(error))
        return connection

    def extract_apt_addresses_data(self, ):
        """
        Extract TomTom's Anchor point from MNR Database with complete address information using SQL Query
        :return: pandas Dataframe
        """
        sql_query = "SET search_path TO {}, public;" \
                    "SELECT mnr_apt.feat_id::TEXT, ST_AsText(mnr_apt.geom), " \
                    "mnr_address.iso_script, mnr_address.iso_lang_code, " \
                    "postal_code.postal_code as postal_code, " \
                    "house_number.hsn as house_number, " \
                    "state_province_code.name as state_province_code," \
                    "place_name.name as locality, " \
                    "street_name.name as street_name, " \
                    "country_code.name as country_code, " \
                    "street_name.nc_prefix as prefix, " \
                    "street_name.nc_suffix as suffix, " \
                    "street_name.nc_predir as predir, " \
                    "street_name.nc_postdir as postdir, " \
                    "street_name.nc_body as sn_body " \
                    "FROM mnr_apt join mnr_apt2addressset on mnr_apt.feat_id = mnr_apt2addressset.apt_id " \
                    "join mnr_address on mnr_apt2addressset.addressset_id = mnr_address.addressset_id " \
                    "left join mnr_postal_point as postal_code on mnr_address.postal_code_id = postal_code.feat_id " \
                    "left join mnr_hsn as house_number on mnr_address.house_number_id = house_number.hsn_id " \
                    "left join mnr_name as country_name on mnr_address.country_name_id = country_name.name_id " \
                    "left join mnr_name as state_province_name on mnr_address.state_province_name_id = state_province_name.name_id " \
                    "left join mnr_name as place_name on mnr_address.place_name_id = place_name.name_id " \
                    "left join mnr_name as street_name on mnr_address.street_name_id = street_name.name_id " \
                    "left join mnr_name as door on mnr_address.door_id = door.name_id " \
                    "left join mnr_name as floor_id on mnr_address.floor_id = floor_id.name_id " \
                    "left join mnr_name as building_name on mnr_address.building_name_id = building_name.name_id " \
                    "left join mnr_name as block on mnr_address.block_id = block.name_id " \
                    "left join mnr_name as landmark_nearby on mnr_address.landmark_nearby_id = landmark_nearby.name_id " \
                    "left join mnr_name as landmark_direction on mnr_address.landmark_direction_id = landmark_direction.name_id " \
                    "left join mnr_name as ddsn on mnr_address.double_dependent_street_name_id = ddsn.name_id " \
                    "left join mnr_name as dsn on mnr_address.dependent_street_name_id = dsn.name_id " \
                    "left join mnr_name as street_number on mnr_address.street_number_id = street_number.name_id " \
                    "left join mnr_name as country_code on mnr_address.country_code_id = country_code.name_id " \
                    "left join mnr_name as state_province_code on " \
                    "mnr_address.state_province_code_id = state_province_code.name_id ".format(self.country_code)

        self.__logger__.info("Running SQL query on {} schema".format(self.country_code))
        stime = time.time()
        dataframe = pd.read_sql(sql_query, self.connection)
        # convert featureID  to string
        # dataframe['feat_id'] = dataframe['feat_id'].apply(lambda x: str(x))
        self.__logger__.info("APT Data Downloaded Took {:.2f} min".format((time.time() - stime) / 60.0))
        return dataframe

    @staticmethod
    def save_dataframe_as_shpfile(pd_dataframe: pd.DataFrame, out_path='artifacts', filename='results.shp'):
        """
        Static Method to save Pandas Dataframe as geopandas shp file.
        :param pd_dataframe: Pandas Dataframe
        :param out_path: (string) output save directory
        :param filename: (string) output filename
        :return:
        """
        pd_dataframe['coordinates'] = gpd.GeoSeries.from_wkt(pd_dataframe['st_astext'])
        pd_dataframe = pd_dataframe.drop('st_astext', axis=1)
        geo_dataframe = gpd.GeoDataFrame(pd_dataframe, geometry='coordinates', crs="EPSG:4326")
        filename = os.path.join(out_path, filename)
        geo_dataframe.to_file(driver='ESRI Shapefile', filename=filename)
        return filename


class ExtractAPT:
    def __init__(self, ):
        self.client = pycoredb.Client(url="http://cpp-read.maps-contentops.amiefarm.com/cdapps/coredb-main-ws/")

    def extract_APT(self, footprint):
        """
        Function to extract TomTom's Anchor point or each building polygon
        :param footprint:
        :return:
        """
        data = self.client.get_features(*footprint, feature_types=('TTOM:4_0:TTOM-Apt:AnchorPoint',),
                                        feature_attributes=["TTOM:4_0:TTOM-Apt:AnchorPoint.ID",
                                                            "TTOM:4_0:TTOM-Apt:AnchorPoint.AnchorPointType",
                                                            "TTOM:4_0:TTOM:LinkedLocation.EntryPointFunction",
                                                            "TTOM:4_0:TTOM:LinkedLocation.Chainage",
                                                            ])
        feature_data = [(features.geometry.wkt, features.feature_id) for features in data if
                        features._feature_data['AnchorPointType'] == 'AddressParcelCentroid']
        return np.asarray(feature_data)

    def filter_aptdata(self, feature_matrix):
        """Function which filters an incoming ndarray of feature matrix and returns the APT's values.
        :param feature_matrix : Input feature array.
        :return APT values
        :rtype list
        """
        apt_values = None
        if feature_matrix.shape[0] > 0:
            apt_values = feature_matrix[:, 0]
            apt_values = [wkt.loads(point) for point in apt_values]
            if len(apt_values) > 1:
                apt_values = MultiPoint(apt_values)
            else:
                apt_values = Point(apt_values[0])
        return apt_values

    def filter_aptid(self, feature_matrix: np.ndarray) -> list:
        """Function which filters an incoming ndarray of feature matrix and returns the APT's id's.
        :param feature_matrix : Input feature array.
        :return APT values
        :rtype list
        """
        apt_ids = None
        if feature_matrix.shape[0] > 0:
            apt_ids = feature_matrix[:, 1]
        return apt_ids


def download_apt_elements(shape_file, out_path):
    ACCESS_TOKEN = "1d165393-ef1a-453a-89be-b53754641786"
    BRANCH_ID = "233b38a4-f0bf-4289-bfdc-7f2a04fc4ab3"
    try:
        from coredb_utility import executor
        final_out_dir, final_out_filename, association_out_filename = executor.get_mds_data(
            in_sf_path=shape_file,
            access_token=ACCESS_TOKEN,
            branch_id=BRANCH_ID,
            feature_name='apt', max_workers=20, out_path=out_path)
    except ImportError as error:
        logging.error(error)
    return


if __name__ == '__main__':
    mnr_database = ExtractMNRData(country_code='_2022_06_010_nam_usa_ufl')
    mnr_database.connect_to_server()
    out_path = '/mnt/c/Users/tandon/OneDrive - TomTom/Desktop/tomtom/Workspace/01_Rooftop_accuracy/BFP_Analysis_USA/data/data/Florida'
    mnr_apt_df = mnr_database.extract_apt_addresses_data()
    mnr_database.save_dataframe_as_shpfile(mnr_apt_df, out_path, filename='APT_2022_06_009_nam_usa_ufl.shp')