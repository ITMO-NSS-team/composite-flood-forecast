import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 5, 3

from model.metrics import metric_by_name
from model.ensemble import prepare_base_ensemle_data, load_ensemble, prepare_advanced_ensemle_data
from model.wrap import prepare_table_input_data
from validation.paths import TS_PATH, MULTI_PATH, TS_DATAFRAME_PATH, MULTI_DATAFRAME_PATH, SERIALISED_ENSEMBLES_PATH, \
    get_list_with_stations_id

from validation.paths import SNOWCOVER_4045_PATH, RIVER4045_PATH, PRECIP_4045_PATH, CONVERTER_PATH, SRM_PATH

from model.phys_model.train_converter import get_meteo_df
from model.phys_model.launch_srm_model import load_converter, load_SRM


def ensemble_metric_calculation(metrics: list, stations_to_check: list = None, test_size: int = 805):
    """ Plot metrics for different forecast horizons """
    ts_df = pd.read_csv(TS_DATAFRAME_PATH, parse_dates=['date'])
    multi_df = pd.read_csv(MULTI_DATAFRAME_PATH, parse_dates=['date'])

    serialised_models = get_list_with_stations_id(stations_to_check)
    # print(serialised_models)
    for metric in metrics:
        metric_function = metric_by_name[metric]

        metrics_results = []
        horizons = []
        for horizon in [1, 2, 3, 4, 5, 6, 7]:
            metric_values = []
            for serialised_model in serialised_models:
                # Load ensemble model
                model = load_ensemble(SERIALISED_ENSEMBLES_PATH, serialised_model)

                if str(serialised_model) == str(3045):
                    river_ts = pd.read_csv(RIVER4045_PATH, parse_dates=['date'])
                    meteo_ts = get_meteo_df()
                    meteo_ts = meteo_ts.drop(labels=['precipitation'], axis=1)
                    snow_ts = pd.read_csv(SNOWCOVER_4045_PATH, parse_dates=['date'])
                    rainfall_ts = pd.read_csv(PRECIP_4045_PATH, parse_dates=['date'])

                    # Load SRM model and RF model to convert discharge into water levels
                    preloaded_converter = load_converter(CONVERTER_PATH)
                    preloaded_srm = load_SRM(SRM_PATH)

                    test_df = prepare_advanced_ensemle_data(ts_df, multi_df, TS_PATH, MULTI_PATH, serialised_model, test_size,
                                                            preloaded_srm, preloaded_converter, river_ts, (meteo_ts, snow_ts, rainfall_ts))
                    test_features = np.array(test_df[['month', 'day', 'ts', 'multi', 'srm']])
                    test_target = np.array(test_df['actual'])
                else:
                    # Prepare data for test
                    test_df = prepare_base_ensemle_data(ts_df, multi_df, TS_PATH, MULTI_PATH, serialised_model, test_size)
                    test_features = np.array(test_df[['month', 'day', 'ts', 'multi']])
                    test_target = np.array(test_df['actual'])

                # Wrap into InputData
                input_data = prepare_table_input_data(features=test_features,
                                                      target=test_target)
                predicted = model.predict(input_data)

                forecast_matrix = prepare_horizon_validation_set(predicted.predict,
                                                                 horizon)
                val_matrix = prepare_horizon_validation_set(test_target, horizon)

                metric_value = metric_function(val_matrix, forecast_matrix)
                metric_values.append(metric_value)

            # Update info about metrics for current horizon
            metrics_results.extend(metric_values)
            horizons.extend([horizon] * len(metric_values))

        metric_title = {'nse': 'NSE', 'mae': 'MAE', 'smape': 'SMAPE'}
        current_metric = metric_title[metric]
        result_df = pd.DataFrame({current_metric: metrics_results,
                                  'Forecast horizon': horizons})

        result_df = result_df.astype({current_metric: float,
                                      'Forecast horizon': str})
        print(result_df)

        # Plot boxplot
        if current_metric == 'SMAPE':
            color = 'white'
        else:
            color = 'grey'

        with sns.axes_style("darkgrid"):
            rcParams['figure.figsize'] = 5, 3
            sns.boxplot(x='Forecast horizon', y=current_metric, data=result_df,
                        color=color, width=0.5)
            plt.show()


def prepare_horizon_validation_set(forecast, horizon):
    """ Take only several time series elements by horizon """
    sparsed_forecast = []
    for i in range(0, len(forecast), 7):
        sparsed_forecast.extend(forecast[i:i + horizon])

    return np.array(sparsed_forecast)


if __name__ == '__main__':
    ensemble_metric_calculation(metrics=['nse', 'smape'],
                                stations_to_check=None,
                                test_size=805)
