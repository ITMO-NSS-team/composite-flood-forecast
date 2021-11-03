import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 7, 6

from model.ensemble import get_ts_forecast
from validation.paths import TS_PATH, TS_DATAFRAME_PATH, get_list_with_stations_id


def time_series_forecasting_plot(stations_to_check: list = None, test_size: int = 805):
    """ Plot simple linear plots with predictions and actual values

    :param stations_to_check: list with stations ids to plot graphs
    :param test_size: number of objects to validate
    """
    df = pd.read_csv(TS_DATAFRAME_PATH, parse_dates=['date'])

    serialised_models = get_list_with_stations_id(stations_to_check)
    for serialised_model in serialised_models:
        station_df = df[df['station_id'] == int(serialised_model)]

        val_predict, time_series, dates = get_ts_forecast(station_df, TS_PATH, serialised_model, test_size)

        plt.plot(dates, time_series, label='Actual time series')
        plt.plot(dates, val_predict, label='Forecast for 7 elements ahead')
        plt.title(f'Station {str(serialised_model)}')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Maximum level value, cm', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    time_series_forecasting_plot(stations_to_check=[3045])
