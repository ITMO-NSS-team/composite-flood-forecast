import os

import pandas as pd
from model.ensemble import init_ensemble


def ensemble_prepare_models(stations_to_prepare: list = None, test_size: int = 805):
    """ Initialise ensemble model for each hydro gauges, fit it and serialise """
    # Full length of dataset minus test size
    train_len = 2190 - test_size
    ensemble_len = 861

    # Calculate train part
    ts_path = '../serialised/time_series'
    multi_path = '../serialised/multi_target'

    ts_dataframe_path = '../data/level_time_series.csv'
    multi_dataframe_path = '../data/multi_target.csv'

    ts_df = pd.read_csv(ts_dataframe_path, parse_dates=['date'])
    multi_df = pd.read_csv(multi_dataframe_path, parse_dates=['date'])

    if stations_to_prepare is not None:
        serialised_models = stations_to_prepare
    else:
        serialised_models = os.listdir(ts_path)

    print(f'Ensemble model will be prepared for stations {serialised_models}')
    for serialised_model in serialised_models:
        # Create ensemble model and save it in folder
        if str(serialised_model) == str(3045):
            # TODO init model with SRM
            pass
        else:
            init_ensemble(ts_df, multi_df, ts_path, multi_path, serialised_model, train_len, ensemble_len)


if __name__ == '__main__':
    ensemble_prepare_models(stations_to_prepare=[3045], test_size=805)
