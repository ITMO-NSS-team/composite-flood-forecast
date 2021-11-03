import os
import pandas as pd
import numpy as np

from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.pipelines.pipeline import Pipeline

from model.ensemble import get_ts_forecast
from model.metrics import metric_by_name
from model.wrap import prepare_ts_input_data
from validation.paths import TS_PATH, TS_DATAFRAME_PATH


def time_series_metric_calculation(metrics: list, stations_to_check: list = None,
                                   test_size: int = 805):
    """ Calculate metrics for time series forecasting algorithm """
    df = pd.read_csv(TS_DATAFRAME_PATH)

    if stations_to_check is not None:
        serialised_models = stations_to_check
    else:
        serialised_models = os.listdir(TS_PATH)

    for metric in metrics:
        metric_function = metric_by_name[metric]
        metric_values = []

        for serialised_model in serialised_models:
            station_df = df[df['station_id'] == int(serialised_model)]
            val_predict, actual, dates = get_ts_forecast(station_df, TS_PATH, serialised_model, test_size)

            metric_value = metric_function(actual, val_predict)
            metric_values.append(metric_value)

        metric_values = np.array(metric_values)
        print(f'Metric {metric} value for all stations is {np.mean(metric_values):.2f}')


if __name__ == '__main__':
    time_series_metric_calculation(metrics=['nse', 'mae', 'smape'], stations_to_check=None)
