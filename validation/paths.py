import os

TS_PATH = '../serialised/time_series'
MULTI_PATH = '../serialised/multi_target'

TS_DATAFRAME_PATH = '../data/level_time_series.csv'
MULTI_DATAFRAME_PATH = '../data/multi_target.csv'

SERIALISED_ENSEMBLES_PATH = '../serialised/ensemble'

RIVER4045_PATH = '../data/no_gaps_train.csv'
SNOWCOVER_4045_PATH = '../data/snowcover_val.csv'
PRECIP_4045_PATH = '../data/rainfall_data_test.csv'

CONVERTER_PATH = '../serialised/level_model.pkl'
SRM_PATH = '../serialised/SRM.pkl'


def get_list_with_stations_id(stations_to_check):
    if stations_to_check is not None:
        return stations_to_check
    else:
        return os.listdir(TS_PATH)
