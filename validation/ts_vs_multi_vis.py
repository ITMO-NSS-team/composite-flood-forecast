import os
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from pylab import rcParams

from model.ensemble import prepare_base_ensemle_data
from model.metrics import nash_sutcliffe
from validation.paths import TS_PATH, MULTI_PATH, TS_DATAFRAME_PATH, MULTI_DATAFRAME_PATH, get_list_with_stations_id

rcParams['figure.figsize'] = 7, 6


def create_biplots(stations_to_check: list = None, test_size: int = 805):
    """ Plot graphs "Prediction vs Actual values" for time series and multi-target models

    :param stations_to_check: list with stations ids to plot graphs
    :param test_size: number of objects to validate
    """
    ts_df = pd.read_csv(TS_DATAFRAME_PATH, parse_dates=['date'])
    multi_df = pd.read_csv(MULTI_DATAFRAME_PATH, parse_dates=['date'])

    serialised_models = get_list_with_stations_id(stations_to_check)
    for serialised_model in serialised_models:
        test_df = prepare_base_ensemle_data(ts_df, multi_df, TS_PATH, MULTI_PATH, serialised_model, test_size)
        target = np.array(test_df['actual'])

        test_df = test_df.rename(columns={'actual': 'Actual values', 'ts': 'Forecasts of time series model',
                                          'multi': 'Multi-target regression forecast'})

        ts_metric_value = nash_sutcliffe(target, np.array(test_df['Forecasts of time series model']))
        multi_metric_value = nash_sutcliffe(target, np.array(test_df['Multi-target regression forecast']))

        print(f'NSE ts metric: {ts_metric_value:.2f}')
        print(f'NSE multi metric: {multi_metric_value:.2f}')

        for pal, color, forecast_name in zip(['Blues', 'Oranges'],
                                             ['#000D95', '#E89400'],
                                             ['Forecasts of time series model', 'Multi-target regression forecast']):

            sns.lineplot(data=test_df, x='Actual values', y='Actual values', color='black', linewidth=0.5)
            sns.kdeplot(data=test_df, x='Actual values', y=forecast_name,
                        fill=True, thresh=0, levels=150, cmap=pal)
            sns.scatterplot(data=test_df, x='Actual values', y=forecast_name,
                            s=10, alpha=0.2, color=color, edgecolor=color)
            plt.show()


if __name__ == '__main__':
    create_biplots(stations_to_check=[3028, 3029, 3030, 3041], test_size=805)
