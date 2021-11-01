import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 9, 6

from model.ensemble import init_ensemble, prepare_ensemle_data


def ensemble_forecasting_plot(stations_to_check: list = None, test_size: int = 805):
    # Full length of dataset minus test size
    train_len = 2190 - test_size
    ensemble_len = 861

    # Calculate train part
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
        # Create ensemble model
        model = init_ensemble(ts_df, multi_df, ts_path, multi_path, serialised_model, train_len, ensemble_len)

        # Prepare data for test
        test_df = prepare_ensemle_data(ts_df, multi_df, ts_path, multi_path, serialised_model, test_size)

        predicted = model.predict(np.array(test_df[['month', 'day', 'ts', 'multi']]))

        plt.plot(test_df['date'], test_df['actual'], label='Actual time series')
        plt.plot(test_df['date'], test_df['ts'], label='Time series model', alpha=0.4)
        plt.plot(test_df['date'], test_df['multi'], label='Multi-target regression', alpha=0.4)
        plt.plot(test_df['date'], predicted, label='Ensemble')
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    ensemble_forecasting_plot(stations_to_check=[3028, 3029, 3030, 3041], test_size=805)
