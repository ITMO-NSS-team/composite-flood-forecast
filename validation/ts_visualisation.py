import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 7, 6

from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.pipelines.pipeline import Pipeline

from model.wrap import prepare_ts_input_data


def time_series_forecasting_plot(stations_to_check: list = None, test_size: int = 805):
    path = '../serialised/time_series'
    dataframe_path = '../data/level_time_series.csv'
    df = pd.read_csv(dataframe_path, parse_dates=['date'])

    if stations_to_check is not None:
        serialised_models = stations_to_check
    else:
        serialised_models = os.listdir(path)

    for serialised_model in serialised_models:
        # Read serialised model
        model_path = os.path.join(path, str(serialised_model), 'model.json')
        pipeline = Pipeline()
        pipeline.load(model_path)

        station_df = df[df['station_id'] == int(serialised_model)]

        # Source time series and test part
        time_series = np.array(station_df['stage_max'])
        input_data = prepare_ts_input_data(time_series)
        val_predict = in_sample_ts_forecast(pipeline=pipeline,
                                            input_data=input_data,
                                            horizon=test_size)

        dates = station_df['date']
        plt.plot(dates, input_data.target, label='Actual time series')

        train_part = time_series[:-test_size]
        test_ids = np.arange(len(train_part), len(train_part) + len(val_predict))
        plt.plot(dates.iloc[test_ids],
                 val_predict, label='Forecast for 7 elements ahead')
        plt.legend()
        plt.title(f'Station {str(serialised_model)}')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Maximum level value, cm', fontsize=12)

        deviation = np.std(val_predict)
        index = len(dates) - test_size
        plt.plot([dates.iloc[index], dates.iloc[index]],
                 [min(input_data.target) - deviation, max(input_data.target) + deviation],
                 c='black', linewidth=1)
        plt.xlim(dates.iloc[index - 400], dates.iloc[len(dates)-1])
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()


if __name__ == '__main__':
    time_series_forecasting_plot(stations_to_check=[3028, 3029, 3030, 3041])
