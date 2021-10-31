import os
import pandas as pd
import numpy as np

from fedot.core.pipelines.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from model.metrics import smape, nash_sutcliffe
from model.wrap import prepare_table_input_data


def multi_target_metric_calculation(metrics: list):
    metric_by_name = {'smape': smape,
                      'mae': mean_absolute_error,
                      'nse': nash_sutcliffe}

    test_size = 1505
    path = '../serialised/multi_target'
    dataframe_path = '../data/multi_target.csv'
    df = pd.read_csv(dataframe_path, parse_dates=['date'])

    serialised_models = os.listdir(path)
    for metric in metrics:
        metric_function = metric_by_name[metric]
        metric_values = []

        for serialised_model in serialised_models:
            # Read serialised model
            model_path = os.path.join(path, serialised_model, 'model.json')
            pipeline = Pipeline()
            pipeline.load(model_path)

            station_df = df[df['station_id'] == int(serialised_model)]
            test_data = np.array(station_df['1_day'])[-test_size - 1: -1]

            # Save only test part
            station_df = station_df.tail(test_size)
            features = np.array(station_df[['stage_max_amplitude', 'stage_max_mean',
                                            'snow_coverage_station_amplitude',
                                            'snow_height_mean',
                                            'snow_height_amplitude',
                                            'water_hazard_sum']])
            target = np.array(station_df[['1_day', '2_day', '3_day',
                                          '4_day', '5_day', '6_day',
                                          '7_day']])

            input_data = prepare_table_input_data(features, target)
            output_data = pipeline.predict(input_data)
            predicted = np.array(output_data.predict)

            forecasts = []
            for i in range(0, test_size, 7):
                forecasts.extend(predicted[i, :])

            metric_value = metric_function(test_data,
                                           np.array(forecasts))
            metric_values.append(metric_value)

        metric_values = np.array(metric_values)
        print(f'Metric {metric} value for all stations is {np.mean(metric_values):.2f}')


if __name__ == '__main__':
    multi_target_metric_calculation(metrics=['nse'])
