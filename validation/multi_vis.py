import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from model.ensemble import get_multi_forecast
from validation.paths import MULTI_PATH, MULTI_DATAFRAME_PATH


def multi_target_forecasting_plot(test_size: int = 805, stations_to_check: list = None):
    df = pd.read_csv(MULTI_DATAFRAME_PATH, parse_dates=['date'])

    if stations_to_check is not None:
        serialised_models = stations_to_check
    else:
        serialised_models = os.listdir(MULTI_PATH)
    for serialised_model in serialised_models:
        station_df = df[df['station_id'] == int(serialised_model)]
        forecasts = get_multi_forecast(station_df, MULTI_PATH, serialised_model, test_size)

        forecasts_df = pd.DataFrame({'date': station_df['date'][-test_size:],
                                     'predict': forecasts})

        # Convert station_train into time-series dataframe
        station_df['stage_max'] = station_df['1_day'].shift(1)
        station_df = station_df.tail(test_size)

        # Remove first row
        station_df = station_df.tail(len(station_df) - 1)
        plt.plot(station_df['date'], station_df['stage_max'], label='Actual time series')
        plt.plot(forecasts_df['date'], forecasts_df['predict'], label='Forecast for 7 elements ahead')
        plt.title(f'Station {str(serialised_model)}')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Maximum level value, cm', fontsize=12)
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    multi_target_forecasting_plot(stations_to_check=None)
