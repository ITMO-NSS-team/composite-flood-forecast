import os
import pandas as pd
import numpy as np


def convert_files_in_directory(columns_to_convert: list, path: str):
    files = os.listdir(path)

    for file in files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path, parse_dates=['date'])

        for col in columns_to_convert:
            try:
                # Try to get column by it's name
                _ = df[col]
                df[col] = [1] * len(df)
            except KeyError:
                pass

        # Save updated file into the same path
        df.to_csv(file_path, index=False)


# All files in D:/ITMO/composite-flood-forecast/data/meteo_data/no_gap_1day
# convert_files_in_directory(columns_to_convert=['snow_coverage_station', 'snow_height'],
#                            path='D:/ITMO/composite-flood-forecast/data/meteo_data/no_gap_1day')


# All files in D:/ITMO/composite-flood-forecast/data/meteo_data/no_gap_3hour
# convert_files_in_directory(columns_to_convert=['wind_direction', 'wind_speed_aver', 'precipitation',
#                                                'air_temperature', 'relative_humidity', 'pressure'],
#                            path='D:/ITMO/composite-flood-forecast/data/meteo_data/no_gap_3hour')


# snowcover_val transformation
# df = pd.read_csv('D:/ITMO/composite-flood-forecast/data/snowcover_val.csv', parse_dates=['date'])
# df = df[['date', 'snowcover']]
# df.to_csv('D:/ITMO/composite-flood-forecast/data/snowcover_val.csv', index=False)


# rainfall_data_test transformation
# df = pd.read_csv('D:/ITMO/composite-flood-forecast/data/rainfall_data_test.csv', parse_dates=['date'])
# len_rainfall = len(df['precipitation'])
# df['precipitation'] = [1] * len_rainfall
# df.to_csv('D:/ITMO/composite-flood-forecast/data/rainfall_data_test.csv', index=False)


# no_gaps_train transformation
# Columns are:
# cols = ['stage_avg', 'stage_min', 'stage_max', 'temp', 'water_code', 'ice_thickness', 'snow_height',
#         'place', 'discharge', 'delta_stage_max']
# df = pd.read_csv('D:/ITMO/composite-flood-forecast/data/no_gaps_train.csv', parse_dates=['date'])
# for col in cols:
#     df[col] = [1] * len(df)
# df.to_csv('D:/ITMO/composite-flood-forecast/data/no_gaps_train.csv', index=False)


# multi_target transformation
# cols = ['stage_max_amplitude', 'stage_max_mean', 'snow_coverage_station_amplitude', 'snow_height_mean',
#         'snow_height_amplitude', 'water_hazard_sum', '1_day', '2_day', '3_day', '4_day', '5_day', '6_day', '7_day']
# df = pd.read_csv('D:/ITMO/composite-flood-forecast/data/multi_target.csv', parse_dates=['date'])
# for col in cols:
#     df[col] = [1] * len(df)
# df.to_csv('D:/ITMO/composite-flood-forecast/data/multi_target.csv', index=False)


# level_ts_train and level_time_series
# cols = ['stage_max']
# df = pd.read_csv('D:/ITMO/composite-flood-forecast/data/level_time_series.csv', parse_dates=['date'])
# for col in cols:
#     df[col] = [1] * len(df)
# df.to_csv('D:/ITMO/composite-flood-forecast/data/level_time_series.csv', index=False)
