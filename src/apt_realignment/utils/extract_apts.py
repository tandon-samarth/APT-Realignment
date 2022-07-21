import os
import geopandas as gpd
from coredb_utility import executor
from shapely import wkt
from shapely.geometry import MultiPoint, Point
import numpy as np

import pycoredb

ACCESS_TOKEN = "1d165393-ef1a-453a-89be-b53754641786"
BRANCH_ID = "233b38a4-f0bf-4289-bfdc-7f2a04fc4ab3"


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
    final_out_dir, final_out_filename, association_out_filename = executor.get_mds_data(
        in_sf_path=shape_file,
        access_token=ACCESS_TOKEN,
        branch_id=BRANCH_ID,
        feature_name='apt', max_workers=20, out_path=out_path)
    return os.path.join(final_out_dir, final_out_filename)