import pandas as pd
import numpy as np

from model.ensemble import get_ts_forecast
from model.metrics import metric_by_name
from validation.ensemble_part_metric import clip_df_to_april_and_jul
from validation.paths import TS_PATH, TS_DATAFRAME_PATH, get_list_with_stations_id


def time_series_part_metric_calculation(metrics: list, stations_to_check: list = None,
                                        test_size: int = 805):
    """ Calculate metrics for time series forecasting algorithm """
    df = pd.read_csv(TS_DATAFRAME_PATH)

    serialised_models = get_list_with_stations_id(stations_to_check)

    for metric in metrics:
        metric_function = metric_by_name[metric]
        metric_values = []

        for serialised_model in serialised_models:
            station_df = df[df['station_id'] == int(serialised_model)]
            val_predict, actual, dates = get_ts_forecast(station_df, TS_PATH, serialised_model, test_size)

            forecast_df = pd.DataFrame({'date': dates, 'actual': actual, 'forecast': val_predict})
            forecast_df['month'] = pd.DatetimeIndex(forecast_df['date']).month
            forecast_df = clip_df_to_april_and_jul(forecast_df)

            metric_value = metric_function(np.array(forecast_df['actual']),
                                           np.array(forecast_df['forecast']))
            metric_values.append(metric_value)

        metric_values = np.array(metric_values)
        print(f'Metric {metric} value for all stations is {np.mean(metric_values):.2f}')


if __name__ == '__main__':
    time_series_part_metric_calculation(metrics=['nse', 'mae', 'smape'], stations_to_check=None)
