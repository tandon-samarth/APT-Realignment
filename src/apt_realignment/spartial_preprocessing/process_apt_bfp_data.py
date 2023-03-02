
from urllib.parse import quote
import psycopg2
import geopandas as gpd
import pandas as pd

from sqlalchemy import create_engine
from utils.geometric_utils import create_logger
logger = create_logger()

def query_nearest_buildings(name, db_connection, min_bfp=1):
    return gpd.GeoDataFrame.from_postgis(f'''SELECT
  pd."feat_id",
  closest_building.id as {name}_building_id,
  (case when ST_IsEmpty(pd.apt_geometry) then null else closest_building.geometry end) as {name}_building_geometry
  ,
  closest_building.intersects as {name}_intersects,
  closest_building.distance as {name}_distance
 FROM dev_apa.apt_data pd 
LEFT JOIN LATERAL 
  (SELECT
      mblpi.id,
      ST_TRANSFORM(mblpi.geometry, 4326) as geometry,
      ST_INTERSECTS(pd.apt_geometry, mblpi.geometry) as intersects,
      ST_Distance(pd.apt_geometry, mblpi.geometry) as distance
      FROM dev_ppa.msft_building_footprints mblpi          
      ORDER BY pd.apt_geometry <-> mblpi.geometry
      LIMIT {min_bfp}
   ) AS closest_building
   ON TRUE;''', con=db_connection, geom_col=f'{name}_building_geometry')


def get_nearest_building_to_apt(anchor_point_gdf, username='cerebroadmin', password='admin@123', near_bfp=1):
    try:
        db_connection_url = "postgresql://cerebroadmin:%s@10.128.154.4:5432/postgres" % quote(password)
    except (Exception, psycopg2.DatabaseError) as error:
        raise("{}!Unable to connect to server please check internet connection or username/password".format(error))
    logger.info("Connection to {} successful".format(username))
    connection = create_engine(db_connection_url, pool_size=20, max_overflow=0)
    logger.info("Pushing APT data to PostGis ..")
    anchor_point_gdf.to_postgis(name="apt_data", schema='dev_apa', con=connection, index=False, if_exists='replace')
    logger.info("Acquiring Nearest Building Footprints")
    nearest_bfp_to_apts = query_nearest_buildings(name='apt', db_connection=connection, min_bfp=near_bfp)
    return nearest_bfp_to_apts


def get_bfp2bfp_dist_diff(input_df):
    diff = input_df.groupby('feat_id')['apt_distance'].apply(lambda x: x.iloc[1] - x.iloc[0]).reset_index()
    diff.columns = ['feat_id', 'distance_diff']
    diff['distance_diff'] = diff['distance_diff'].apply(lambda x: round(x, 2))
    return pd.merge(input_df, diff, on='feat_id')


def process_apt_within_bfp(input_df, thresh_distance=10):
    # Anchor Points on Building footprint
    apt_on_bfp_df = input_df.loc[input_df['apt_intersects'] == True]  # no change in APT coordinates
    apt_on_bfp_df['updated_APT_with_BFP'] = input_df['apt_geometry'].apply(lambda x: x)

    # Anchor Points close to BFP <10m
    apt_dist_in_range = input_df.loc[input_df['apt_distance'] <= thresh_distance]
    apt_dist_in_range['updated_APT_with_BFP'] = apt_dist_in_range['apt_building_geometry'].apply(
        lambda x: x.centroid)

    # Append both dataframes
    final_df = apt_on_bfp_df.append(apt_dist_in_range, ignore_index=True)
    return final_df


def process_apt_outside_bfp(input_df, distance_diff=15):
    df_with_high_diff = get_bfp2bfp_dist_diff(input_df=input_df)
    df_with_high_diff = df_with_high_diff.loc[df_with_high_diff['distance_diff'] > distance_diff]
    df_with_high_diff = df_with_high_diff[df_with_high_diff['feat_id'].duplicated(keep="last")]
    return df_with_high_diff
