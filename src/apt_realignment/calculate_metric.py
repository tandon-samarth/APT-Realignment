from genericpath import isfile
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
import folium
from shapely.geometry import mapping
import json
from pyspark.sql import SparkSession
import unidecode
from addressing.utils import libpostal
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords
from addressing.automatic_matching import automatic_matching
from addressing.automatic_matching.rooftop.rooftop import haversine_distance
import re
import gc
import sys, os
import sqlalchemy
from datetime import date


class CalculateMetric:
    def __init__(self, county_name, sql_database, is_benchmark=False, realigment_matrix=None):
        self.benchmark = is_benchmark
        self.county = county_name
        self.DB = sql_database
        self.upadted_data = realigment_matrix
        self.metric_distance = 50

        date.today()
        nltk.download('stopwords')
        engine = sqlalchemy.create_engine(
            f'postgresql+psycopg2://{self.DB["user"]}:{self.DB["password"]}@{self.DB["host"]}:{self.DB["port"]}/{self.DB["database"]}',
            echo=False)
        self.raw2p = ReadAndWrite2PostgresDB(engine)

        self.countries_stopwords = self.get_stopwords()
        self.countries_stopwords = {k: '|'.join(['\\b' + word + '\\b' for word in v]) for k, v in
                                    self.countries_stopwords.items()}
        source_gdf = self.get_source_data(self.county)
        sample_gdf = self.get_sample_source(name=self.county)
        if is_benchmark:
            print("calculating Benchmark")
            joined_sample = source_gdf.sjoin(sample_gdf, how='inner', predicate='intersects')
        else:
            self.rw_delta_table()
            delta_gdf = self.get_delta_table()
            source_delta = self.replace_geometries(source_gdf, delta_gdf)
            source_delta_gdf = gpd.GeoDataFrame(source_delta.drop('geometry', axis=1), geometry=source_delta.geometry,
                                                crs='EPSG:4326')
            source_delta_gdf.x = source_delta_gdf.geometry.apply(lambda p: p.x)
            source_delta_gdf.y = source_delta_gdf.geometry.apply(lambda p: p.y)
            joined_sample = source_delta_gdf.sjoin(sample_gdf, how='inner', predicate='intersects')
            del delta_gdf

        del sample_gdf

        joined_sample.rename({'hsn': 'hsnum', 'street_name': 'st_name', 'postal_code': 'zip_code'}, axis=1,
                             inplace=True)
        parsed_df = self.parse_joined_sample(joined_sample)
        similarity_df = self.apt_similarity_filter(df=parsed_df, stopwords_pattern=self.countries_stopwords.get('us'))
        del parsed_df

        results , matching_df = self.create_matching(similarity_df=similarity_df)




    def create_matching(self,similarity_df,):
        similarity_df['match'] = similarity_df['match'].fillna(0)
        match_proportion = np.mean(similarity_df['match'])
        clean_proportion = round(match_proportion * 100, 2)

        # create match matrix
        match_df = similarity_df[['feat_id', 'match', 'sample_id']]
        match_df['county'] = self.county
        match_df['datetime_run'] = pd.Timestamp.now(tz='utc')
        match_df.rename({'match': 'asf'}, axis=1, inplace=True)

        ## ASF calculation
        [lower_distance, upper_distance] = self.percentile_bootstrap(similarity_df['match'], np.mean)
        results_asf = self.save_asf(lower_distance, upper_distance, match_proportion, benchmark=self.benchmark)

        matches_df = similarity_df[similarity_df['match'] == 1]
        matches_df['mnr_query_distance'] = matches_df['mnr_query_distance'].astype(float)
        positional_accuracy_distance = round(np.quantile(matches_df['mnr_query_distance'], 0.9), 2)

        print(f'Positional Accuracy (90th percentile distance) is: {positional_accuracy_distance}m')

        [lower_percentile90, upper_percentile90] = self.percentile_bootstrap(matches_df['mnr_query_distance'],
                                                                             lambda x: np.quantile(x, 0.9))
        ## 90 Percentile Calculation
        results_90sum = self.save_90percentile(lower_percentile90, upper_percentile90, positional_accuracy_distance,
                                               benchmark=self.benchmark)
        ## 50m Match Calculation
        proportion_50m_matches = (matches_df['mnr_query_distance'] <= self.metric_distance).mean()
        nice_num_50m = round(proportion_50m_matches * 100, 1)
        print(f'The calculated percentage of matches within 50 meters is {nice_num_50m}%')

        [lower_50m_pa, upper_50m_pa] = self.percentile_bootstrap(
            matches_df['mnr_query_distance'] <= self.metric_distance, np.mean)

        results_50m = self.save_metric_match(lower_50m_pa, upper_50m_pa, proportion_50m_matches, benchmark=self.benchmark)

        results_sum = pd.concat([results_asf, results_90sum, results_50m])
        print(results_sum)
        return results_sum , match_df

    def save_metric_match(self, low_50_pa, up_50_pa, prp_match, benchmark=False):
        version = date
        if benchmark:
            version = 'benchmark'
        new_result = pd.DataFrame(
            data=[[low_50_pa, prp_match, up_50_pa, '%', 'APA', version, self.county]],
            columns=['lower_bound', 'calculated_metric', 'upper_bound', 'units', 'metric', 'version', 'county'])
        return new_result

    def save_90percentile(self, low_90, upper_90, pos_accurcay_distance, benchmark=False):
        version = date
        if benchmark:
            version = 'benchmark'
        new_result = pd.DataFrame(
            data=[[low_90, pos_accurcay_distance, upper_90, 'meters', '90p', version, self.county]],
            columns=['lower_bound', 'calculated_metric', 'upper_bound', 'units', 'metric', 'version', 'county'],
            index=None)
        return new_result

    def save_asf(self, lower_distance, upper_distance, match_prp, benchmark=False):
        version = date
        if benchmark:
            version = 'benchmark'

        results_sum = pd.DataFrame(
            data=[[lower_distance, match_prp, upper_distance, '%', 'ASF', version, self.county]],
            columns=['lower_bound', 'calculated_metric', 'upper_bound', 'units', 'metric', 'version', 'county'],
            index=None)
        return results_sum

    def bootstrap_resample(self, df, agg_fun, times=1000, seed=0):
        reboot = []

        for t in range(times):
            df_boot = df.sample(frac=1, replace=True, random_state=t + seed)
            reboot.append(agg_fun(df_boot))

        return reboot

    def percentile_bootstrap(self, df, agg_fun, conf=0.9, times=1000, seed=0):
        """Generic Percentile Bootstrap
        This function returns a percentile bootstrap confidence interval for a statistic.
        Args:
            df (pandas.DataFrame): DataFrame with the observed random vectors. Each row represents an observation an each column is a random variable.
            agg_fun (function): Aggregation function. This function should receive as input a pandas.DataFrame (resamples) and return a
            number with the computed statistic.
            conf (float, optional): Confidence level of the returned interval. Defaults to 0.9.
            times (int, optional): Bootstrap resamples. Defaults to 1000.
            seed (int, optional): Random seed. Defaults to 0.
        Returns:
            numpy.array: Percentile Boostrap CI [lower, upper]
        """
        reboot = self.bootstrap_resample(df, agg_fun, times, seed)
        return np.quantile(reboot, [(1 - conf) / 2, (1 - conf) / 2 + conf])

    def parse_joined_sample(self, spatial_joined_df: pd.DataFrame) -> pd.DataFrame:
        '''Function inversely parses the addresses to create a searched query format so that the addresses in the source
        can be compared to the addresses in the sample.

        :param spatial_joined_df: DataFrame that contains the addresses from the source that are within the polygon of
        the sample generated. It must contain the columns: ['hsn', 'unit_type', 'unit_num', 'pre_dir', 'prefix', 'suffix'
        'post_dir', 'city', 'state', 'zip_code']
        :type spatial_joined_df: pd.DataFrame
        :return: The same dataframe with a column that contains the full addresses inversely parsed.
        :rtype: pd.DataFrame
        '''

        df = spatial_joined_df.copy()

        dict_of_columns = {
            'hsnum': ' ', 'pre_dir': ' ', 'st_name': ' ', 'suffix': ', ', 'city': ' ', 'state': ' ', 'zip_code': ', ',
            'country': ''
        }
        df['pre_dir'].fillna('', inplace=True)
        df['prefix'].fillna('', inplace=True)
        df['suffix'].fillna('', inplace=True)
        df['post_dir'].fillna('', inplace=True)

        for column in dict_of_columns.keys():
            df[column + '_modified'] = df[column].astype(str) + dict_of_columns[column]

        list_of_modified_columns = [col for col in df.columns if '_modified' in col]

        df['searched_query'] = df[list_of_modified_columns].sum(axis=1)

        df['street_name'] = df['pre_dir'] + ' ' + df['prefix'] + ' ' + df['st_name'] + ' ' + df['suffix'] + ' ' + df[
            'post_dir']
        df['name'] = ''  # df['state']

        df = df.rename(columns={
            'hsnum': 'hsn', 'searched_query': 'address', 'zip_code': 'postal_code', 'city': 'place_name',
            'y': 'lat', 'x': 'lon'
        })

        return df

    def replace_geometries(self, source_gdf, delta_gdf):
        '''
        Takes a 'source' geodataframe - copy of MNR database, a 'delta' geo-dataframe and replaces, for every APT (key: feat_id) in source dataframe, the
        coordinates in the sorce with the coordinates in the new
        Param:
        - source_gdf (gpd.GeoDataFrame): geo-dataframe containing MNR coordinates for every APT (feat_id)
        - delta_gdf (gpd.GeoDataFrame): geo-dataframe containing NEW coordinates for some APTs (feat_id)
        ret:
        - source_gdf_new (gpd.GeoDataFrame): geo-dataframe containing MNR information for APT but with new coordinates
        '''
        delta_gdf_grouped = delta_gdf[
            delta_gdf.groupby('feat_id').datetime_version.transform('max') == delta_gdf.datetime_version]
        source_gdf_new = source_gdf.merge(delta_gdf_grouped[["feat_id", "geometry"]], on="feat_id", how="left")
        source_gdf_new.loc[~source_gdf_new.geometry_y.isna(), "geometry_x"] = source_gdf_new.loc[
            ~source_gdf_new.geometry_y.isna(), "geometry_y"]
        source_gdf_new = source_gdf_new.drop(["geometry_y"], axis=1).rename({"geometry_x": "geometry"}, axis=1)
        source_gdf_new = gpd.GeoDataFrame(source_gdf_new.drop('geometry', axis=1),
                                          geometry=source_gdf_new.geometry, crs='EPSG:4326')
        return source_gdf_new

    def get_delta_table(self):
        delta_query = f"""SELECT * FROM "STAN_169".delta_table where county = '{self.county}'"""
        delta_df = self.raw2p.read_from_db(query=delta_query)
        delta_geom = gpd.GeoSeries.from_wkt(delta_df.updated_geometries)
        delta_gdf = gpd.GeoDataFrame(delta_df.drop('updated_geometries', axis=1), geometry=delta_geom, crs='EPSG:4326')
        return delta_gdf

    def rw_delta_table(self):
        delta_table_read = pd.read_pickle(self.upadted_data)
        delta_table_read = delta_table_read[['feat_id', 'updated_geometries']]
        delta_table_read['datetime_version'] = pd.Timestamp.now(tz='utc')

        delta_table_write = delta_table_read[['feat_id', 'updated_geometries', 'datetime_version']]
        delta_table_write['county'] = self.county
        delta_table_write['updated_geometries'] = delta_table_write['updated_geometries'].astype(str)
        self.raw2p.write_to_db(df=delta_table_write, schema='STAN_169', table_name='delta_table')
        return 0

    def get_source_data(self, name):
        source_query = f"""SELECT * FROM "STAN_169".source_v0 where county = '{name}'"""
        source_df = self.raw2p.read_from_db(query=source_query)
        source_geom = gpd.GeoSeries.from_wkt(source_df.geometry)
        source_gdf = gpd.GeoDataFrame(source_df.drop('geometry', axis=1),
                                      geometry=source_geom,
                                      crs='EPSG:4326')
        print("Shape of the source data", source_df.shape)
        return source_gdf

    def get_sample_source(self, name):
        sample_query = f"""SELECT * FROM "STAN_169".sample where county = '{name}'"""
        sample_df = self.raw2p.read_from_db(query=sample_query)
        sample_geom = gpd.GeoSeries.from_wkt(sample_df.geometry)
        sample_gdf = gpd.GeoDataFrame(sample_df.drop('geometry', axis=1), geometry=sample_geom, crs='EPSG:4326')
        return sample_gdf

    def apt_similarity_filter(
            # country:str,
            df: pd.DataFrame,
            sample_df: pd.DataFrame,
            stopwords_pattern: str = '') -> pd.DataFrame:
        """Performs matching after making call in a given radius
        :param country: country to call in MNR
        :type country: str
        :param df: DataFrame containing the sample addresses (must have coordinates)
        :type df: pd.DataFrame
        :param sample_df: DataFrame containing libpostal components for sample (df) addresses
        :type sample_df: pd.DataFrame
        :param radius: radius of the buffer
        :type radius: float
        :param inner_radius: radius in meters of a smaller buffer. When bigger than zero, we are essentially getting the point in a disk, defaults to 0
        :type inner_radius: int or float, optional
        :param stopwords_pattern: regex pattern to remove stopwords, if needed. Optional, defaults to None
        :type stopwords_pattern: str
        :return: DataFrame with the APTs that matched
        :rtype: pd.DataFrame
        """
        apts_df = df.copy()

        # Fill NAs
        apts_df[['address', 'street_name', 'hsn', 'postal_code',
                 'place_name', 'name']] = apts_df[['address', 'street_name', 'hsn',
                                                   'postal_code', 'place_name', 'name']].fillna('')
        # Create extra columns for stopwords, optional unidecode
        cols_stopwords = ['address', 'street_name', 'place_name']
        for col in cols_stopwords:
            col_create = col + '_no_stopwords'
            apts_df[col_create] = apts_df[col].str.replace(stopwords_pattern, '', case=False, regex=True)

        for col in cols_stopwords:
            col_create = col + '_no_stopwords_unidecode'
            apts_df[col_create] = apts_df[col + '_no_stopwords'].apply(lambda x: unidecode.unidecode(x))

        # Merge to APTs
        apts_df['libpostal_road_no_stopwords'] = apts_df.libpostal_road.str.replace(stopwords_pattern, '', case=False,
                                                                                    regex=True)

        # House number similarity: filter obvious non matches
        apts_df['hsn_similarity'] = list(map(fuzz.token_set_ratio, apts_df.libpostal_house_number, apts_df.hsn))
        apts_df['re_pattern'] = '\\b' + apts_df.hsn.astype(str) + '\\b'

        dropped_df = apts_df.loc[apts_df.hsn_similarity <= 60].reset_index(drop=True)

        apts_df = apts_df.loc[apts_df.hsn_similarity > 60].reset_index(drop=True)

        # Postal code similarity
        apts_df['postcode_similarity'] = list(map(fuzz.WRatio,
                                                  apts_df.libpostal_postcode,
                                                  apts_df.postal_code.fillna('').astype(str)))
        apts_df['postcode_similarity'] = np.where(apts_df.libpostal_postcode == '', np.nan,
                                                  np.where(apts_df.postal_code == '', 50, apts_df.postcode_similarity))

        # Road similarity
        apts_df['road_similarity'] = list(map(fuzz.token_set_ratio,
                                              apts_df.libpostal_road_no_stopwords,
                                              apts_df.street_name_no_stopwords))
        apts_df['road_similarity_unidecode'] = list(map(fuzz.token_set_ratio,
                                                        apts_df.libpostal_road_no_stopwords,
                                                        apts_df.street_name_no_stopwords_unidecode))
        apts_df['road_similarity'] = apts_df[['road_similarity', 'road_similarity_unidecode']].max(axis=1)

        # Locality similarity
        apts_df['searched_query_tokens'] = (apts_df.libpostal_road.astype(str) + ' ' +
                                            apts_df.libpostal_house_number.astype(str) + ' ' +
                                            apts_df.libpostal_postcode.astype(str))

        apts_df['provider_tokens'] = (apts_df.street_name.astype(str) + ' ' +
                                      apts_df.hsn.astype(str) + ' ' + apts_df.postal_code.astype(str))
        apts_df['aux_searched_query'] = apts_df.apply(
            lambda x: automatic_matching.replace_tokens(x.searched_query_unidecode_sample, x.searched_query_tokens),
            axis=1)
        apts_df['aux_provider_address'] = apts_df.apply(
            lambda x: automatic_matching.replace_tokens(x.address, x.provider_tokens), axis=1)
        apts_df['aux_provider_address'] = apts_df.aux_provider_address.fillna('').apply(
            lambda x: unidecode.unidecode(x))
        apts_df['locality_wratio'] = apts_df.apply(
            lambda x: fuzz.WRatio(str(x.aux_searched_query).lower(), str(x.aux_provider_address).lower()), axis=1)
        apts_df['locality_city_state_ratio'] = apts_df.apply(
            lambda x: fuzz.WRatio(str(x.libpostal_city) + ' ' + str(x.libpostal_state),
                                  str(x.place_name) + ' ' + str(x.name)), axis=1)
        apts_df['locality_similarity'] = apts_df[['locality_wratio', 'locality_city_state_ratio']].mean(axis=1)

        apts_df['mnr_query_distance'] = apts_df.apply(lambda x: haversine_distance(x.lat, x.lon,
                                                                                   x.lat_sample, x.lon_sample)
        if not np.isnan(x.lat) else 1e7
                                                      , axis=1)

        # Compute mean similarity
        apts_df['mean_similarity'] = (apts_df[['locality_similarity', 'hsn_similarity',
                                               'postcode_similarity', 'road_similarity']].mean(axis=1)
                                      * np.where(apts_df.hsn_similarity >= 70, 1, 0)
                                      * np.where(apts_df.road_similarity >= 60, 1, 0)
                                      * np.where(apts_df.mnr_query_distance > 1000, 0, 1)
                                      )

        apts_df_matching = (
            apts_df.sort_values(by='mnr_query_distance')
                .loc[apts_df.groupby(['sample_id'])
                .mean_similarity.idxmax()]
                .reset_index(drop=True)
        )

        # Compute matching
        apts_df_matching['match'] = pd.NaT
        apts_df_matching['match'] = np.where(apts_df_matching.mean_similarity >= 70, 1, pd.NaT)  # 90 so far best
        address_sample_ids = apts_df_matching['sample_id']
        non_matches_ids = dropped_df[~dropped_df['sample_id'].isin(address_sample_ids)]

        addresses_to_add_ids = non_matches_ids['sample_id'].unique()

        addresses_id_df = pd.DataFrame(
            {'sample_id': addresses_to_add_ids, 'match': [pd.NaT] * len(addresses_to_add_ids)}
        )

        addresses_id_df = addresses_id_df.merge(sample_df[['sample_id', 'searched_query_unidecode_sample']],
                                                on='sample_id', how='left')
        cols_to_add = [col for col in apts_df_matching if col not in addresses_id_df.columns]

        addresses_id_df.loc[:, cols_to_add] = ''
        addresses_id_df_reordered = addresses_id_df[apts_df_matching.columns]

        apts_final = pd.concat([apts_df_matching, addresses_id_df_reordered])

        return apts_final
    @staticmethod
    def get_stopwords():
        countries_stopwords = {
            'br': stopwords.words('portuguese') + ['rua', 'avenida'],
            'ca': stopwords.words('french') + stopwords.words('english') + ['road', 'street', 'st.', 'st', 'rue',
                                                                            'chemin', 'avenue'],
            'es': stopwords.words('spanish') + ['calle', 'avenida', 'callejón', 'paseo'],
            'fr': stopwords.words('french') + ['rue', 'chemin', 'avenue'],
            'gb': stopwords.words('english') + ['street', 'road', 'avenue', 'st.', 'st', 'drive'],
            'it': stopwords.words('italian') + ['via', 'viale', 'strada'],
            'mx': stopwords.words('spanish') + ['calle', 'avenida', 'callejón', 'paseo'],
            'us': stopwords.words('english') + ['street', 'road', 'avenue', 'st.', 'st', 'drive'],
            'be': stopwords.words('french') + ['rue', 'chemin', 'avenue'],
            'za': stopwords.words('english') + ['street', 'road', 'avenue', 'st.', 'st', 'drive']
        }
        return countries_stopwords


class ReadAndWrite2PostgresDB:
    def __init__(self, engine):
        self.engine = engine

    def read_from_db(self, query, retry_num=3):
        for _ in range(retry_num):
            df = None
            try:
                df = pd.read_sql(query, self.engine)
                return df
            except Exception as e:
                print(e)
        return df

    def write_to_db(self, df, schema, table_name, retry_num=3):
        for _ in range(retry_num):
            try:
                df.to_sql(table_name, con=self.engine, if_exists='append', schema=schema, index=False)
                print("Table stored!")
                return 1
            except Exception as e:
                print(e)
        return 0


if __name__ == '__main__':
    DB = {
        'host': "10.137.173.84",
        'port': '5432',
        'database': "STAN",
        'user': "strategicadmin",
        'password': "TBmG4Yj3DdwOI+Aq"
    }
    county_name = 'broward'
    metric = CalculateMetric(county_name,DB,is_benchmark=True)

