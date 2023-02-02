from urllib.parse import quote

import geopandas as gpd
import psycopg2
from sqlalchemy import create_engine

from geometric_utils import create_logger


class ExtractMSFTData:
    def __init__(self, database_name='mnr', databse_server='caprod-cpp-pgmnr-002.flatns.net'):
        self.__logger__ = create_logger()
        self.host = databse_server
        self.database_name = database_name
        self.connection = self.connect_to_server()
        if self.connection:
            self.__logger__.info("connection successfull..")

    def connect_to_server(self, username='cerebroadmin', password='admin@123'):
        """
        Connect to MNR Database
        :return: connection object
        """
        connection = None
        db_connection_url = "postgresql://cerebroadmin:%s@10.128.154.4:5432/postgres" % quote(password)
        alchemyengine = create_engine(db_connection_url)
        try:
            connection = alchemyengine.connect()
        except (Exception, psycopg2.DatabaseError) as error:
            self.__logger__.error("{}".format(error))
        return connection

    def extract_bfp(self, polygon, save=True, filename='country.shp'):
        sql_query = "SELECT st_transform(geometry, 4326) " \
                    "AS geom FROM dev_ppa.msft_building_footprints " \
                    "WHERE ST_Within(geometry,st_transform(ST_GeomFromText('{}',4326),3857))".format(polygon)

        gdf = gpd.GeoDataFrame.from_postgis(sql_query, self.connection)
        gdf = gdf.to_crs("epsg:4326")
        if save:
            gdf.to_file(filename, driver='ESRI Shapefile')
        return gdf
