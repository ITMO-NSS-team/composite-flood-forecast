import os

import pandas as pd
from model.ensemble import init_base_ensemble, init_advanced_ensemble
from validation.paths import TS_PATH, MULTI_PATH, TS_DATAFRAME_PATH, MULTI_DATAFRAME_PATH


def ensemble_prepare_models(stations_to_prepare: list = None, test_size: int = 805):
    """ Initialise ensemble model for each hydro gauges, fit it and serialise """
    # Full length of dataset minus test size
    train_len = 2190 - test_size
    ensemble_len = 861

    ts_df = pd.read_csv(TS_DATAFRAME_PATH, parse_dates=['date'])
    multi_df = pd.read_csv(MULTI_DATAFRAME_PATH, parse_dates=['date'])

    if stations_to_prepare is not None:
        serialised_models = stations_to_prepare
    else:
        serialised_models = os.listdir(TS_PATH)

    print(f'Ensemble model will be prepared for stations {serialised_models}')
    for serialised_model in serialised_models:
        # Create ensemble model and save it in folder
        if str(serialised_model) == str(3045):
            # TODO init model with SRM
            init_advanced_ensemble(ts_df, multi_df, TS_PATH, MULTI_PATH, serialised_model, train_len, ensemble_len)
        else:
            init_base_ensemble(ts_df, multi_df, TS_PATH, MULTI_PATH, serialised_model, train_len, ensemble_len)


if __name__ == '__main__':
    ensemble_prepare_models(stations_to_prepare=[3045], test_size=805)
