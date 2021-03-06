import pandas as pd
import numpy as np

from model.metrics import metric_by_name
from model.ensemble import prepare_base_ensemle_data, load_ensemble, prepare_advanced_ensemle_data
from model.wrap import prepare_table_input_data
from validation.paths import TS_PATH, MULTI_PATH, TS_DATAFRAME_PATH, MULTI_DATAFRAME_PATH, SERIALISED_ENSEMBLES_PATH, \
    get_list_with_stations_id

from validation.paths import SNOWCOVER_4045_PATH, RIVER4045_PATH, PRECIP_4045_PATH, CONVERTER_PATH, SRM_PATH

from model.phys_model.train_converter import get_meteo_df
from model.phys_model.launch_srm_model import load_converter, load_SRM


def ensemble_metric_calculation(metrics: list, stations_to_check: list = None, test_size: int = 805):
    """ Calculate metric for ensemble of models for all test size """
    ts_df = pd.read_csv(TS_DATAFRAME_PATH, parse_dates=['date'])
    multi_df = pd.read_csv(MULTI_DATAFRAME_PATH, parse_dates=['date'])

    # Clip validation part
    validation_len = 805

    serialised_models = get_list_with_stations_id(stations_to_check)
    # print(serialised_models)
    for metric in metrics:
        metric_function = metric_by_name[metric]
        metric_values = []

        for serialised_model in serialised_models:
            # Load ensemble model
            model = load_ensemble(SERIALISED_ENSEMBLES_PATH, serialised_model)
            ts_local = ts_df[ts_df['station_id'] == int(serialised_model)]
            multi_local = multi_df[multi_df['station_id'] == int(serialised_model)]

            ts_local = clip_last_n_values(ts_local, last_values=validation_len)
            multi_local = clip_last_n_values(multi_local, last_values=validation_len)

            if str(serialised_model) == str(3045):
                river_ts = pd.read_csv(RIVER4045_PATH, parse_dates=['date'])
                meteo_ts = get_meteo_df()
                meteo_ts = meteo_ts.drop(labels=['precipitation'], axis=1)
                snow_ts = pd.read_csv(SNOWCOVER_4045_PATH, parse_dates=['date'])
                rainfall_ts = pd.read_csv(PRECIP_4045_PATH, parse_dates=['date'])

                # Load SRM model and RF model to convert discharge into water levels
                preloaded_converter = load_converter(CONVERTER_PATH)
                preloaded_srm = load_SRM(SRM_PATH)

                test_df = prepare_advanced_ensemle_data(ts_local, multi_local, TS_PATH, MULTI_PATH,
                                                        serialised_model, test_size,
                                                        preloaded_srm, preloaded_converter, river_ts,
                                                        (meteo_ts, snow_ts, rainfall_ts))
                test_features = np.array(test_df[['month', 'day', 'ts', 'multi', 'srm']])
                test_target = np.array(test_df['actual'])
            else:
                # Prepare data for test
                test_df = prepare_base_ensemle_data(ts_local, multi_local, TS_PATH,
                                                    MULTI_PATH, serialised_model, test_size)
                test_features = np.array(test_df[['month', 'day', 'ts', 'multi']])
                test_target = np.array(test_df['actual'])

            # Wrap into InputData
            input_data = prepare_table_input_data(features=test_features,
                                                  target=test_target)
            predicted = model.predict(input_data)
            # predicted = np.array(test_df['srm'])

            metric_value = metric_function(test_target, predicted.predict)
            metric_values.append(metric_value)

        metric_values = np.array(metric_values)
        print(f'Metric {metric} value for all stations is {np.mean(metric_values):.2f}')


def clip_last_n_values(dataframe: pd.DataFrame, last_values: int):
    """ Stay only first values in the dataframe """
    all_df_len = len(dataframe)
    clipped_len = round(all_df_len - last_values)
    return dataframe.head(clipped_len)


if __name__ == '__main__':
    ensemble_metric_calculation(metrics=['nse', 'mae', 'smape'],
                                stations_to_check=[3045],
                                test_size=805)
