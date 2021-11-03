import os

TS_PATH = '../serialised/time_series'
MULTI_PATH = '../serialised/multi_target'

TS_DATAFRAME_PATH = '../data/level_time_series.csv'
MULTI_DATAFRAME_PATH = '../data/multi_target.csv'

SERIALISED_ENSEMBLES_PATH = '../serialised/ensemble'


def get_list_with_stations_id(stations_to_check):
    if stations_to_check is not None:
        return stations_to_check
    else:
        return os.listdir(TS_PATH)
