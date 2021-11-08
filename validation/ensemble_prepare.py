import os
import pickle

import pandas as pd
from model.ensemble import init_base_ensemble, init_advanced_ensemble
from validation.paths import TS_PATH, MULTI_PATH, TS_DATAFRAME_PATH, MULTI_DATAFRAME_PATH, get_list_with_stations_id

from validation.paths import SNOWCOVER_4045_PATH, RIVER4045_PATH, PRECIP_4045_PATH, CONVERTER_PATH, SRM_PATH
from model.phys_model.train_converter import get_meteo_df
from model.phys_model.launch_srm_model import load_converter, load_SRM

def ensemble_prepare_models(stations_to_prepare: list = None, test_size: int = 805):
    """ Initialise ensemble model for each hydro gauges, fit it and serialise """
    # Full length of dataset minus test size
    train_len = 2190 - test_size
    ensemble_len = 861
    
    
    
    ts_df = pd.read_csv(TS_DATAFRAME_PATH, parse_dates=['date'])
    multi_df = pd.read_csv(MULTI_DATAFRAME_PATH, parse_dates=['date'])


    
    serialised_models = get_list_with_stations_id(stations_to_prepare)
    print(f'Ensemble model will be prepared for stations {serialised_models}')
    for serialised_model in serialised_models:
        # Create ensemble model and save it in folder
        if str(serialised_model) == str(3045):
            # TODO init model with SRM
            river_ts = pd.read_csv(RIVER4045_PATH, parse_dates=['date'])
            meteo_ts = get_meteo_df()
            snow_ts = pd.read_csv(SNOWCOVER_4045_PATH, parse_dates=['date'])
            rainfall_ts = pd.read_csv(PRECIP_4045_PATH, parse_dates=['date'])
        
            preloaded_converter = load_converter(CONVERTER_PATH)
            preloaded_SRM = load_SRM(SRM_PATH)

            init_advanced_ensemble(ts_df, multi_df, TS_PATH, MULTI_PATH, serialised_model, train_len, ensemble_len,
                                   preloaded_SRM, preloaded_converter, river_ts, (meteo_ts, snow_ts, rainfall_ts))
        else:
            init_base_ensemble(ts_df, multi_df, TS_PATH, MULTI_PATH, serialised_model, train_len, ensemble_len)


if __name__ == '__main__':
    ensemble_prepare_models(stations_to_prepare=[3045], test_size=805)
