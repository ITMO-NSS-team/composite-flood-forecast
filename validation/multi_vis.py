import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from fedot.core.pipelines.pipeline import Pipeline
from model.wrap import prepare_table_input_data


def multi_target_forecasting_plot(test_size: int = 805):
    path = '../serialised/multi_target'
    dataframe_path = '../data/multi_target.csv'
    df = pd.read_csv(dataframe_path, parse_dates=['date'])

    serialised_models = os.listdir(path)
    for serialised_model in serialised_models:
        # Read serialised model
        model_path = os.path.join(path, serialised_model, 'model.json')
        pipeline = Pipeline()
        pipeline.load(model_path)

        station_df = df[df['station_id'] == int(serialised_model)]
        station_df = station_df.tail(test_size)
        features = np.array(station_df[['stage_max_amplitude', 'stage_max_mean',
                                        'snow_coverage_station_amplitude',
                                        'snow_height_mean',
                                        'snow_height_amplitude',
                                        'water_hazard_sum']])
        target = np.array(station_df[['1_day', '2_day', '3_day',
                                      '4_day', '5_day', '6_day',
                                      '7_day']])

        input_data = prepare_table_input_data(features, target)
        output_data = pipeline.predict(input_data)
        predicted = np.array(output_data.predict)

        forecasts = []
        for i in range(0, test_size, 7):
            forecasts.extend(predicted[i, :])

        forecasts_df = pd.DataFrame({'date': station_df['date'],
                                     'predict': forecasts})

        # Convert station_train into time-series dataframe
        station_df['stage_max'] = station_df['1_day'].shift(1)

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
    multi_target_forecasting_plot()
