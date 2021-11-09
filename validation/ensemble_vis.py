import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

from model.phys_model.launch_srm_model import load_converter, load_SRM
from model.phys_model.train_converter import get_meteo_df
from model.wrap import prepare_table_input_data

rcParams['figure.figsize'] = 9, 6

from model.ensemble import prepare_base_ensemle_data, load_ensemble, prepare_advanced_ensemle_data
from validation.paths import TS_PATH, MULTI_PATH, TS_DATAFRAME_PATH, \
    MULTI_DATAFRAME_PATH, SERIALISED_ENSEMBLES_PATH, \
    get_list_with_stations_id, RIVER4045_PATH, SNOWCOVER_4045_PATH, \
    PRECIP_4045_PATH, CONVERTER_PATH, SRM_PATH


def ensemble_forecasting_plot(stations_to_check: list = None, test_size: int = 805):
    ts_df = pd.read_csv(TS_DATAFRAME_PATH, parse_dates=['date'])
    multi_df = pd.read_csv(MULTI_DATAFRAME_PATH, parse_dates=['date'])

    serialised_models = get_list_with_stations_id(stations_to_check)
    for serialised_model in serialised_models:
        # Load ensemble model
        model = load_ensemble(SERIALISED_ENSEMBLES_PATH, serialised_model)

        if str(serialised_model) == str(3045):
            river_ts = pd.read_csv(RIVER4045_PATH, parse_dates=['date'])
            meteo_ts = get_meteo_df()
            meteo_ts = meteo_ts.drop(labels=['precipitation'], axis=1)
            snow_ts = pd.read_csv(SNOWCOVER_4045_PATH, parse_dates=['date'])
            rainfall_ts = pd.read_csv(PRECIP_4045_PATH,
                                      parse_dates=['date'])

            preloaded_converter = load_converter(CONVERTER_PATH)
            preloaded_SRM = load_SRM(SRM_PATH)

            test_df = prepare_advanced_ensemle_data(ts_df, multi_df, TS_PATH,
                                                    MULTI_PATH,
                                                    serialised_model, test_size,
                                                    preloaded_SRM,
                                                    preloaded_converter,
                                                    river_ts, (
                                                    meteo_ts, snow_ts,
                                                    rainfall_ts))
            test_features = np.array(test_df[['month', 'day', 'ts', 'multi', 'srm']])
            test_target = np.array(test_df['actual'])
        else:
            # Prepare data for test
            test_df = prepare_base_ensemle_data(ts_df, multi_df, TS_PATH, MULTI_PATH, serialised_model, test_size)
            test_features = np.array(test_df[['month', 'day', 'ts', 'multi']])
            test_target = np.array(test_df['actual']).reshape((-1, 1))

        input_data = prepare_table_input_data(features=test_features,
                                              target=test_target)
        predicted = model.predict(input_data)

        plt.plot(test_df['date'], test_df['actual'], label='Actual time series')
        plt.plot(test_df['date'], test_df['ts'], label='Time series model', alpha=0.4)
        plt.plot(test_df['date'], test_df['multi'], label='Multi-target regression', alpha=0.4)
        plt.plot(test_df['date'], predicted.predict, label='Ensemble')
        if str(serialised_model) == str(3045):
            plt.plot(test_df['date'], test_df['srm'], label='SRM forecast', alpha=0.4)
        plt.grid()
        plt.title(f'Station {str(serialised_model)}')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Maximum level value, cm', fontsize=12)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    ensemble_forecasting_plot(stations_to_check=[3045],
                              test_size=805)
