import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.pipelines.pipeline import Pipeline

from model.wrap import prepare_ts_input_data


def time_series_forecasting_plot(stations_to_check: list = None):
    test_size = 1505
    path = '../serialised/time_series'
    dataframe_path = '../data/level_time_series.csv'
    df = pd.read_csv(dataframe_path)

    if stations_to_check is not None:
        serialised_models = stations_to_check
    else:
        serialised_models = os.listdir(path)

    for serialised_model in serialised_models:
        # Read serialised model
        model_path = os.path.join(path, serialised_model, 'model.json')
        pipeline = Pipeline()
        pipeline.load(model_path)

        station_df = df[df['station_id'] == int(serialised_model)]

        # Source time series and test part
        time_series = np.array(station_df['stage_max'])
        train_part = time_series[:-test_size]

        input_data = prepare_ts_input_data(time_series)

        val_predict = in_sample_ts_forecast(pipeline=pipeline,
                                            input_data=input_data,
                                            horizon=test_size)

        plt.plot(input_data.idx, input_data.target,
                 label='Actual time series')
        plt.plot(np.arange(len(train_part), len(train_part) + len(val_predict)),
                 val_predict, label='Forecast for 7 elements ahead')
        plt.legend()
        plt.title(f'Station {str(serialised_model)}')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Maximum level value, cm', fontsize=12)
        plt.grid()
        plt.show()


if __name__ == '__main__':
    time_series_forecasting_plot()
