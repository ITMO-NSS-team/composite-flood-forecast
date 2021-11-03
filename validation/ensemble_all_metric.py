import pandas as pd
import numpy as np

from model.metrics import metric_by_name
from model.ensemble import prepare_base_ensemle_data, load_ensemble, prepare_advanced_ensemle_data
from model.wrap import prepare_table_input_data
from validation.paths import TS_PATH, MULTI_PATH, TS_DATAFRAME_PATH, MULTI_DATAFRAME_PATH, SERIALISED_ENSEMBLES_PATH, \
    get_list_with_stations_id


def ensemble_metric_calculation(metrics: list, stations_to_check: list = None, test_size: int = 805):
    """ Calculate metric for ensemble of models for all test size """
    ts_df = pd.read_csv(TS_DATAFRAME_PATH, parse_dates=['date'])
    multi_df = pd.read_csv(MULTI_DATAFRAME_PATH, parse_dates=['date'])

    serialised_models = get_list_with_stations_id(stations_to_check)
    for metric in metrics:
        metric_function = metric_by_name[metric]
        metric_values = []

        for serialised_model in serialised_models:
            # Load ensemble model
            model = load_ensemble(SERIALISED_ENSEMBLES_PATH, serialised_model)

            if str(serialised_model) == str(3045):
                test_df = prepare_advanced_ensemle_data(ts_df, multi_df, TS_PATH, MULTI_PATH, serialised_model, test_size)
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

            metric_value = metric_function(test_target,  predicted.predict)
            metric_values.append(metric_value)

        metric_values = np.array(metric_values)
        print(f'Metric {metric} value for all stations is {np.mean(metric_values):.2f}')


if __name__ == '__main__':
    ensemble_metric_calculation(metrics=['nse', 'mae', 'smape'],
                                stations_to_check=[3019, 3027, 3028, 3029, 3030, 3035, 3041, 3045, 3050, 3230],
                                test_size=805)
