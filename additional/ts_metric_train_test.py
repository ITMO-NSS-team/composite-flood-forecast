import pandas as pd
import numpy as np

from additional.ensemble_metric_train_test import clip_last_n_values
from model.ensemble import get_ts_forecast
from model.metrics import metric_by_name
from validation.paths import TS_PATH, TS_DATAFRAME_PATH, get_list_with_stations_id


def time_series_metric_calculation(metrics: list, stations_to_check: list = None,
                                   test_size: int = 805):
    """ Calculate metrics for time series forecasting algorithm """
    validation_len = 805
    df = pd.read_csv(TS_DATAFRAME_PATH)

    serialised_models = get_list_with_stations_id(stations_to_check)

    for metric in metrics:
        metric_function = metric_by_name[metric]
        metric_values = []

        for serialised_model in serialised_models:
            station_df = df[df['station_id'] == int(serialised_model)]
            station_df = clip_last_n_values(station_df, last_values=validation_len)
            val_predict, actual, dates = get_ts_forecast(station_df, TS_PATH, serialised_model, test_size)

            metric_value = metric_function(actual, val_predict)
            metric_values.append(metric_value)

        metric_values = np.array(metric_values)
        print(f'Metric {metric} value for all stations is {np.mean(metric_values):.2f}')


if __name__ == '__main__':
    time_series_metric_calculation(metrics=['nse', 'mae', 'smape'],
                                   stations_to_check=[3045])
