import os
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from pylab import rcParams

from model.ensemble import get_ts_forecast, get_multi_forecast
from model.metrics import nash_sutcliffe

rcParams['figure.figsize'] = 7, 6


def create_biplots(stations_to_check: list = None, test_size: int = 805):
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

    for serialised_model in serialised_models:
        # Get time series forecast
        station_ts_df = ts_df[ts_df['station_id'] == int(serialised_model)]
        ts_predict, actual, _ = get_ts_forecast(station_ts_df, ts_path, serialised_model, test_size)

        # Get output from multi-target regression
        station_multi_df = multi_df[multi_df['station_id'] == int(serialised_model)]
        multi_predict = get_multi_forecast(station_multi_df, multi_path, serialised_model, test_size)

        vis_df = pd.DataFrame({'Actual values': actual,
                               'Forecasts of time series model': ts_predict,
                               'Multi-target regression forecast': multi_predict})

        ts_metric_value = nash_sutcliffe(actual, ts_predict)
        multi_metric_value = nash_sutcliffe(actual, multi_predict)

        print(f'NSE ts metric: {ts_metric_value:.2f}')
        print(f'NSE multi metric: {multi_metric_value:.2f}')

        for pal, color, forecast_name in zip(['Blues', 'Oranges'],
                                             ['#000D95', '#E89400'],
                                             ['Forecasts of time series model', 'Multi-target regression forecast']):

            sns.lineplot(data=vis_df, x='Actual values', y='Actual values', color='black', linewidth=0.5)
            sns.kdeplot(data=vis_df, x='Actual values', y=forecast_name,
                        fill=True, thresh=0, levels=150, cmap=pal)
            sns.scatterplot(data=vis_df, x='Actual values', y=forecast_name,
                            s=10, alpha=0.2, color=color, edgecolor=color)
            plt.show()


if __name__ == '__main__':
    create_biplots(stations_to_check=[3028, 3029, 3030, 3041], test_size=805)
