import os
import pandas as pd
import numpy as np

from fedot.core.pipelines.pipeline import Pipeline

from model.ensemble import get_multi_forecast
from model.metrics import metric_by_name
from model.wrap import prepare_table_input_data
from validation.paths import MULTI_PATH, MULTI_DATAFRAME_PATH


def multi_target_metric_calculation(metrics: list, test_size: int = 805,
                                    stations_to_check: list = None):

    df = pd.read_csv(MULTI_DATAFRAME_PATH, parse_dates=['date'])

    if stations_to_check is not None:
        serialised_models = stations_to_check
    else:
        serialised_models = os.listdir(MULTI_PATH)
    for metric in metrics:
        metric_function = metric_by_name[metric]
        metric_values = []

        for serialised_model in serialised_models:
            station_df = df[df['station_id'] == int(serialised_model)]
            test_data = np.array(station_df['1_day'])[-test_size - 1: -1]

            forecasts = get_multi_forecast(station_df, MULTI_PATH, serialised_model, test_size)

            metric_value = metric_function(test_data, np.array(forecasts))
            metric_values.append(metric_value)

        metric_values = np.array(metric_values)
        print(f'Metric {metric} value for all stations is {np.mean(metric_values):.2f}')


if __name__ == '__main__':
    multi_target_metric_calculation(metrics=['nse', 'mae', 'smape'],
                                    stations_to_check=[3045])
