import os
import pandas as pd
import numpy as np

from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.pipelines.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from model.metrics import smape, nash_sutcliffe
from model.wrap import prepare_ts_input_data


def time_series_metric_calculation(metrics: list, stations_to_check: list = None,
                                   test_size: int = 805):
    metric_by_name = {'smape': smape,
                      'mae': mean_absolute_error,
                      'nse': nash_sutcliffe}

    path = '../serialised/time_series'
    dataframe_path = '../data/level_time_series.csv'
    df = pd.read_csv(dataframe_path)

    if stations_to_check is not None:
        serialised_models = stations_to_check
    else:
        serialised_models = os.listdir(path)

    for metric in metrics:
        metric_function = metric_by_name[metric]
        metric_values = []

        for serialised_model in serialised_models:
            # Read serialised model
            model_path = os.path.join(path, str(serialised_model), 'model.json')
            pipeline = Pipeline()
            pipeline.load(model_path)

            station_df = df[df['station_id'] == int(serialised_model)]

            # Source time series and test part
            time_series = np.array(station_df['stage_max'])
            test_part = time_series[-test_size:]

            input_data = prepare_ts_input_data(time_series)

            val_predict = in_sample_ts_forecast(pipeline=pipeline,
                                                input_data=input_data,
                                                horizon=test_size)

            metric_value = metric_function(test_part, val_predict)
            metric_values.append(metric_value)

        metric_values = np.array(metric_values)
        print(f'Metric {metric} value for all stations is {np.mean(metric_values):.2f}')


if __name__ == '__main__':
    time_series_metric_calculation(metrics=['nse', 'mae', 'smape'], stations_to_check=[3050])
