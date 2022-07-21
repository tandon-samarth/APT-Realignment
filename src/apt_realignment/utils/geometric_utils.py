import os
import geopandas as gpd
import logging


def df_2_geovector(data_frame, vector, prefix='geometry', outdir='artifact', vector_format='shp'):
    gdf = gpd.GeoDataFrame(data_frame[vector], crs="epsg:4326", geometry=vector)
    gdf.to_crs("epsg:4326")
    out_dir = os.path.join(outdir, vector)
    if not os.path.isdir(out_dir):
        try:
            os.mkdir(outdir)
        except OSError as err:
            logging.error("{}".format(err))

    if vector_format == 'shp':
        gdf.to_file(os.path.join(out_dir, "{}_{}.shp".format(prefix, vector)), driver='ESRI Shapefile')
    else:
        gdf.to_file(os.path.join(out_dir, "{}_{}.geojson".format(prefix, vector)), driver="GeoJSON")
    return


def read_vector_data(vector_file):
    vector_df = gpd.read_file(vector_file)
    vector_df = gpd.GeoDataFrame(vector_df, crs="EPSG:4326", geometry='geometry')
    vector_df = vector_df.to_crs("epsg:4326")
    return vector_df
