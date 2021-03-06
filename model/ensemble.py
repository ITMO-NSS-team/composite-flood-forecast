import os
import pandas as pd
import numpy as np
from fedot.core.pipelines.node import PrimaryNode

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from sklearn.metrics import mean_absolute_error

from model.wrap import prepare_table_input_data, prepare_ts_input_data
from model.phys_model.srm_model import  apply_3045_phys_model
from model.phys_model.launch_srm_model import convert_discharge_into_stage_max


def get_srm_forecast(preloaded_srm, preloaded_convert_model, river_df, *meteo_dfs):
    """ Forecast from SRM model

    :param preloaded_srm: callable object, fitted model
    :param preloaded_convert_model: model to convert SRM output (discharge) into water levels
    :param river_df: dataframe with information from hydro gauges
    :param meteo_dfs: dataframes with meteo info
    """
    print(type(river_df), [type(df) for df in meteo_dfs])
    intervals = 7
    forecast_discharge = apply_3045_phys_model(preloaded_srm, river_df, intervals, *meteo_dfs)
    output = convert_discharge_into_stage_max(preloaded_convert_model, forecast_discharge, 
                                              [date.month for date in meteo_dfs[0]['date']])
    print('SRM calculations conducted')
    return output, meteo_dfs[0]['date']


def get_ts_forecast(station_ts_df, ts_path, serialised_model, test_size):
    """ Load serialised model for time series forecasting. And then make a forecast.

    :param station_ts_df: DataFrame with features for prediction
    :param ts_path: path to serialised time series models
    :param serialised_model: identifier for model (station id)
    :param test_size: number of objects to validate

    :return ts_predict: numpy array with time series forecast
    :return : actual values
    :return : pandas series with dates for forecasting time indices
    """
    # Read serialised model for time series forecasting

    ts_model_path = os.path.join(ts_path, str(serialised_model), 'model.json')
    ts_pipeline = Pipeline()
    ts_pipeline.load(ts_model_path)

    # Time series forecast
    time_series = np.array(station_ts_df['stage_max'])
    input_data = prepare_ts_input_data(time_series)
    ts_predict = in_sample_ts_forecast(pipeline=ts_pipeline, input_data=input_data, horizon=test_size)

    return ts_predict, time_series[-test_size:], station_ts_df['date'][-test_size:]


def get_multi_forecast(station_multi_df, multi_path, serialised_model, test_size):
    """ Load serialised model for time series forecasting. And then make a forecast.

    :param station_multi_df: DataFrame with features for multi-target prediction
    :param multi_path: path to serialised multi-target models
    :param serialised_model: identifier for model (station id)
    :param test_size: number of objects to validate
    """

    # Read serialised model for multi-target regression
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


def prepare_base_ensemle_data(ts_df, multi_df, ts_path: str,
                              multi_path: str, serialised_model, test_size):
    # Get time series forecast
    station_ts_df = ts_df[ts_df['station_id'] == int(serialised_model)]
    ts_predict, actual, dates = get_ts_forecast(station_ts_df, ts_path, serialised_model, test_size)

    # Get output from multi-target regression
    station_multi_df = multi_df[multi_df['station_id'] == int(serialised_model)]
    multi_predict = get_multi_forecast(station_multi_df, multi_path, serialised_model, test_size)

    df = pd.DataFrame({'date': dates, 'ts': ts_predict, 'multi': multi_predict, 'actual': actual})
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day'] = pd.DatetimeIndex(df['date']).day

    return df


def prepare_advanced_ensemle_data(ts_df, multi_df, ts_path: str,
                                  multi_path: str, serialised_model, test_size,
                                  preloaded_srm, preloaded_convert_model, river_df, meteo_dfs):
    # Get output from SRM model
    srm_predict, srm_dates = get_srm_forecast(preloaded_srm, preloaded_convert_model, river_df, *meteo_dfs)
    srm_df = pd.DataFrame({'srm_preds': srm_predict, 'date': srm_dates})

    # Get time series forecast
    station_ts_df = ts_df[ts_df['station_id'] == int(serialised_model)]
    ts_predict, actual, dates = get_ts_forecast(station_ts_df, ts_path, serialised_model, test_size)

    srm_df = srm_df[srm_df['date'] >= dates.iloc[0]]
    srm_df = srm_df[srm_df['date'] <= dates.iloc[-1]]

    # Get output from multi-target regression
    station_multi_df = multi_df[multi_df['station_id'] == int(serialised_model)]
    multi_predict = get_multi_forecast(station_multi_df, multi_path, serialised_model, test_size)

    df = pd.DataFrame({'date': dates, 'ts': ts_predict, 'multi': multi_predict,
                       'actual': actual, 'srm': np.array(srm_df['srm_preds'])})
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day'] = pd.DatetimeIndex(df['date']).day

    return df


def init_base_ensemble(ts_df: pd.DataFrame, multi_df: pd.DataFrame, ts_path: str,
                       multi_path: str, serialised_model, train_len: int, ensemble_len: int):
    """
    Create ensembling algorithm for water level forecasting based on linear regression.
    Ensemble will combine forecasts from time series and multi-target model
    """
    # Get time series forecast
    station_ts_df = ts_df[ts_df['station_id'] == int(serialised_model)]
    cutted_df = station_ts_df.head(train_len)
    ts_predict, actual, dates = get_ts_forecast(cutted_df, ts_path, serialised_model, ensemble_len)

    # Get output from multi-target regression
    station_multi_df = multi_df[multi_df['station_id'] == int(serialised_model)]
    cutted_df = station_multi_df.head(train_len)
    multi_predict = get_multi_forecast(cutted_df, multi_path, serialised_model, ensemble_len)

    train_df = pd.DataFrame({'date': dates, 'ts': ts_predict, 'multi': multi_predict,
                             'actual': actual})
    train_df['month'] = pd.DatetimeIndex(train_df['date']).month
    train_df['day'] = pd.DatetimeIndex(train_df['date']).day

    pipeline = Pipeline(PrimaryNode('rfr'))

    train_features = np.array(train_df[['month', 'day', 'ts', 'multi']])
    train_target = np.array(train_df['actual'])
    input_data = prepare_table_input_data(features=train_features,
                                          target=train_target)
    pipeline = pipeline.fine_tune_all_nodes(loss_function=mean_absolute_error,
                                            loss_params=None,
                                            input_data=input_data,
                                            iterations=100,
                                            cv_folds=5)
    pipeline.fit(input_data)
    pipeline.save('ensemble')


def init_advanced_ensemble(ts_df: pd.DataFrame, multi_df: pd.DataFrame, ts_path: str,
                           multi_path: str, serialised_model, train_len: int, ensemble_len: int,
                           preloaded_srm, preloaded_convert_model, river_df, meteo_dfs):
    """
    Create ensembling algorithm for water level forecasting based on linear regression.
    Ensemble will combine forecasts from time series, multi-target model and SRM model
    """
    # Get output from SRM model
    srm_predict, srm_dates = get_srm_forecast(preloaded_srm, preloaded_convert_model, river_df, *meteo_dfs)
    srm_df = pd.DataFrame({'srm_preds': srm_predict, 'date': srm_dates})

    # Get time series forecast
    station_ts_df = ts_df[ts_df['station_id'] == int(serialised_model)]
    cutted_df = station_ts_df.head(train_len)
    ts_predict, actual, dates = get_ts_forecast(cutted_df, ts_path, serialised_model, ensemble_len)

    srm_df = srm_df[srm_df['date'] >= dates.iloc[0]]
    srm_df = srm_df[srm_df['date'] <= dates.iloc[-1]]

    # Get output from multi-target regression
    station_multi_df = multi_df[multi_df['station_id'] == int(serialised_model)]
    cutted_df = station_multi_df.head(train_len)
    multi_predict = get_multi_forecast(cutted_df, multi_path, serialised_model, ensemble_len)

    train_df = pd.DataFrame({'date': dates, 'ts': ts_predict, 'multi': multi_predict,
                             'actual': actual, 'srm': np.array(srm_df['srm_preds'])})
    train_df['month'] = pd.DatetimeIndex(train_df['date']).month
    train_df['day'] = pd.DatetimeIndex(train_df['date']).day

    pipeline = Pipeline(PrimaryNode('rfr'))

    train_features = np.array(train_df[['month', 'day', 'ts', 'multi', 'srm']])
    train_target = np.array(train_df['actual'])
    input_data = prepare_table_input_data(features=train_features,
                                          target=train_target)
    pipeline = pipeline.fine_tune_all_nodes(loss_function=mean_absolute_error,
                                            loss_params=None,
                                            input_data=input_data,
                                            iterations=100,
                                            cv_folds=5)
    pipeline.fit(input_data)
    pipeline.save('ensemble')


def load_ensemble(serialised_ensembles_path: str, serialised_model):
    """ Load ensemble model (FEDOT pipeline) and return it to make predictions """
    model_path = os.path.join(serialised_ensembles_path, str(serialised_model), 'ensemble.json')
    pipeline = Pipeline()
    pipeline.load(model_path)

    return pipeline
