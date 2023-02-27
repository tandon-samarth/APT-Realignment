import os
import os.path as osp

from urllib.parse import quote
import numpy as np
import pandas as  pd
import geopandas as gpd

import shapely
import shapely.geos
from shapely import wkt
from shapely.geometry import Point, MultiPoint
from sqlalchemy import create_engine

def query_nearest_buildings(name,db_connection,min_bfp=1):
    return gpd.GeoDataFrame.from_postgis(f'''SELECT
      pd."feat_id",
      closest_building.id as {name}_building_id,
      (case when ST_IsEmpty(pd.geometry) then null else closest_building.geometry end) as {name}_building_geometry
      ,
      closest_building.intersects as {name}_intersects,
      closest_building.distance as {name}_distance
     FROM dev_apa.apt_data pd 
    LEFT JOIN LATERAL 
      (SELECT
          mblpi.id,
          ST_TRANSFORM(mblpi.geometry, 4326) as geometry,
          ST_INTERSECTS(pd.geometry, mblpi.geometry) as intersects,
          ST_Distance(pd.geometry, mblpi.geometry) as distance
          FROM dev_ppa.msft_building_footprints mblpi          
          ORDER BY pd.geometry <-> mblpi.geometry
          LIMIT {min_bfp}
       ) AS closest_building
       ON TRUE;''', con=db_connection, geom_col=f'{name}_building_geometry')


def get_nearest_building_to_apt(anchor_point_gdf,username='cerebroadmin',password='admin@123'):
    db_connection_url = "postgresql://cerebroadmin:%s@10.128.154.4:5432/postgres" % quote(password)
    connection = create_engine(db_connection_url, pool_size=20, max_overflow=0)
    anchor_point_gdf.to_postgis(name="apt_data", schema='dev_apa', con=connection, index=False, if_exists='replace')
    nearest_bfp_to_apts = query_nearest_buildings(name='apt', db_connection=connection)
    return nearest_bfp_to_apts





