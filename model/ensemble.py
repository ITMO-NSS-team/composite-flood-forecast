import os
import numpy as np

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from sklearn.linear_model import LinearRegression

from model.wrap import prepare_table_input_data, prepare_ts_input_data


def get_ts_forecast(ts_df, ts_path, serialised_model, test_size):
    station_ts_df = ts_df[ts_df['station_id'] == int(serialised_model)]
    # Read serialised model for time series forecasting
    ts_model_path = os.path.join(ts_path, str(serialised_model), 'model.json')
    ts_pipeline = Pipeline()
    ts_pipeline.load(ts_model_path)

    # Time series forecast
    time_series = np.array(station_ts_df['stage_max'])
    input_data = prepare_ts_input_data(time_series)
    ts_predict = in_sample_ts_forecast(pipeline=ts_pipeline, input_data=input_data, horizon=test_size)

    return ts_predict, time_series[-test_size:]


def get_multi_forecast(multi_df, multi_path, serialised_model, test_size):
    # Read serialised model for multi-target regression
    station_multi_df = multi_df[multi_df['station_id'] == int(serialised_model)]
    multi_model_path = os.path.join(multi_path, str(serialised_model), 'model.json')
    multi_pipeline = Pipeline()
    multi_pipeline.load(multi_model_path)

    station_multi_df = station_multi_df.tail(test_size)
    features = np.array(station_multi_df[['stage_max_amplitude', 'stage_max_mean',
                                          'snow_coverage_station_amplitude',
                                          'snow_height_mean',
                                          'snow_height_amplitude',
                                          'water_hazard_sum']])
    target = np.array(station_multi_df[['1_day', '2_day', '3_day',
                                        '4_day', '5_day', '6_day',
                                        '7_day']])

    input_data = prepare_table_input_data(features, target)
    output_data = multi_pipeline.predict(input_data)
    predicted = np.array(output_data.predict)

    multi_predict = []
    for i in range(0, test_size, 7):
        multi_predict.extend(predicted[i, :])

    multi_predict = np.array(multi_predict)
    return multi_predict


def prepare_ensemle_data():
    pass


def init_ensemble():
    """ Create ensembling algorithm for water level forecasting based on linear regression """
    pass
