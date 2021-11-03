import pandas as pd
import numpy as np

from model.ensemble import get_multi_forecast
from model.metrics import metric_by_name
from validation.ensemble_part_metric import clip_df_to_may_and_jul
from validation.paths import MULTI_PATH, MULTI_DATAFRAME_PATH, get_list_with_stations_id


def multi_target_part_metric_calculation(metrics: list, test_size: int = 805,
                                         stations_to_check: list = None):
    df = pd.read_csv(MULTI_DATAFRAME_PATH, parse_dates=['date'])

    serialised_models = get_list_with_stations_id(stations_to_check)
    for metric in metrics:
        metric_function = metric_by_name[metric]
        metric_values = []

        for serialised_model in serialised_models:
            station_df = df[df['station_id'] == int(serialised_model)]
            test_data = np.array(station_df['1_day'])[-test_size - 1: -1]

            forecasts = get_multi_forecast(station_df, MULTI_PATH, serialised_model, test_size)
            forecast_df = pd.DataFrame({'date': station_df['date'][-test_size:],
                                        'actual': test_data,
                                        'forecast': forecasts})
            forecast_df['month'] = pd.DatetimeIndex(forecast_df['date']).month

            forecast_df = clip_df_to_may_and_jul(forecast_df)

            metric_value = metric_function(np.array(forecast_df['actual']),
                                           np.array(forecast_df['forecast']))
            metric_values.append(metric_value)

        metric_values = np.array(metric_values)
        print(f'Metric {metric} value for all stations is {np.mean(metric_values):.2f}')


if __name__ == '__main__':
    multi_target_part_metric_calculation(metrics=['nse', 'mae', 'smape'],
                                         stations_to_check=None)
