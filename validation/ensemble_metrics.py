import os

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error

from model.metrics import smape, nash_sutcliffe
from model.ensemble import init_ensemble, prepare_ensemle_data


def ensemble_metric_calculation(metrics: list, stations_to_check: list = None, test_size: int = 805):
    metric_by_name = {'smape': smape,
                      'mae': mean_absolute_error,
                      'nse': nash_sutcliffe}

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

    if stations_to_check is not None:
        serialised_models = stations_to_check
    else:
        serialised_models = os.listdir(ts_path)

    for metric in metrics:
        metric_function = metric_by_name[metric]
        metric_values = []

        for serialised_model in serialised_models:
            # Create ensemble model
            model = init_ensemble(ts_df, multi_df, ts_path, multi_path, serialised_model, train_len, ensemble_len)

            # Prepare data for test
            test_df = prepare_ensemle_data(ts_df, multi_df, ts_path, multi_path, serialised_model, test_size)

            predicted = model.predict(np.array(test_df[['month', 'day', 'ts', 'multi']]))

            metric_value = metric_function(np.array(test_df['actual']),
                                           predicted)
            metric_values.append(metric_value)

        metric_values = np.array(metric_values)
        print(f'Metric {metric} value for all stations is {np.mean(metric_values):.2f}')


if __name__ == '__main__':
    ensemble_metric_calculation(metrics=['nse', 'mae', 'smape'],
                                test_size=805)
