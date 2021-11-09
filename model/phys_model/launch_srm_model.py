import os
import pickle
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

from model.phys_model.calculate_levels import convert_max_into_delta
from model.phys_model.srm_model import fit_3045_phys_model, get_const_for_3045, apply_3045_phys_model
from model.phys_model.train_converter import get_meteo_df
# from sat.upload_sat import 

""" Физическая модель прогнозирования расхода вода на гидрологическом посту 3045.
На гидрологическом посту 3045 нет данных о расходах, но есть значения уровней. 

Для моделирования требуется значения с соседних станций:
    - 3036 (расход), станция, расположенная выше по течению реки
    - 3042 (расход), станция, расположенная близко к 3045. Можем принять, что 
значения расходов на станции 3042 хорошо взаимосвязаны с расходом и уровнем воды 
на моделируемой станции 3045.

Физическая модель предсказывает расходы, не уровни! Перерасчет расходов в уровни 
осуществляется при помощи ML модели (случайного леса). 
"""

def load_SRM(filename='SRM.pkl'):
    print(filename)
    with open(filename, 'rb') as f:
        clf = pickle.load(f)

    return clf
    

def load_converter(filename='level_model.pkl'):
    print(filename)
    with open(filename, 'rb') as f:
        clf = pickle.load(f)

    return clf


def get_all_data_for_3045_forecasting():
    df_meteo = get_meteo_df()

    df_levels = pd.read_csv('data/4rd_checkpoint/no_gaps_train.csv', parse_dates=['date'])
    df_3042 = df_levels[df_levels['station_id'] == 3042]
    df_3036 = df_levels[df_levels['station_id'] == 3036]
    df_3045 = df_levels[df_levels['station_id'] == 3045]

    df_hydro = pd.merge(df_3042, df_3036, on='date', suffixes=['_3042', '_3036'])
    df_merge = pd.merge(df_meteo, df_hydro)
    df_merge = pd.merge(df_merge, df_3045, on='date')
    return df_merge


def convert_discharge_into_stage_max(model, forecast, months):
    """ Перерасчет расхода в уровне на основе обученной модели """
    forecast = np.array(forecast).reshape((-1, 1))
    months = np.array(months).reshape((-1, 1))

    x_test = np.hstack((forecast, months))
    # Предсказание уровня на основе расходов
    stage_max = model.predict(x_test)
    stage_max = np.ravel(stage_max)
    return stage_max


def hackaton_approach():
    cwd = os.getcwd()
    meteo_df = get_meteo_df()

    
    # Обучаем модель для периода на 2004 год:
    start_date = '2003-12-31'
    end_date = '2004-12-30'
    mask = (meteo_df['date'] >= start_date) & (meteo_df['date'] <= end_date)
    meteo_period = meteo_df.loc[mask]
    meteo_period = meteo_period.dropna()

    scover = pd.read_csv('/home/maslyaev/hton/edn/snowcover_2004.csv', parse_dates=['date'])
    scover_filtered = snowcover_preprocessing(scover)

    # Обучение физической модели
    dm = fit_3045_phys_model(meteo_period, scover_filtered)

    # Загружаем модель машинного обучения, которая будет конвертировать расходы в уровни
    convert_model = load_converter()

    # Загружаем все необходимые данные
    df_merge = get_all_data_for_3045_forecasting()

    df_submit = pd.read_csv('submissions/sample_submissions/sample_sub_4.csv', parse_dates=['date'])
    df_submit = df_submit[df_submit['station_id'] == 3045]

    # Неизменяемые параметры для станции номер 3045
    params = get_const_for_3045()
    const_area = params['area']
    section_to = params['section_to']
    lapse = params['lapse']
    h_mean = params['h_mean']
    h_st = params['h_st']

    # Пресдсказание алгоритма
    forecasts = []
    for i in range(0, len(df_submit), 7):
        row = df_submit.iloc[i]
        current_date_for_pred = row['date']

        # Получаем данные только до нужной даты, на которую даём прогноз
        local_df = df_merge[df_merge['date'] < current_date_for_pred]
        # Берём текущее значения уровня воды на станции 3045
        current_level = np.array(local_df['stage_max'])[-1]

        # Получаем данные о текущем месяце
        local_sb_df = df_submit.iloc[i:i+7]

        # Задаем предикторы в модель
        last_row = local_df.iloc[-1]
        # Рассчитываем параметр температуры
        tmp = last_row['air_temperature'] + lapse * (h_mean - h_st) * 0.01
        snow_cover = last_row['snow_coverage_station']
        rainfall = last_row['precipitation']
        disch_3042 = last_row['discharge_3042']
        disch_3036 = last_row['discharge_3036']
        start_variables = np.array([tmp, snow_cover, const_area, rainfall, disch_3042, disch_3036])
        start_variables = np.nan_to_num(start_variables)
        start_variables = tuple(start_variables)

        # Метеопарамеры
        lr_col = last_row[['snow_height_y', 'snow_coverage_station', 'air_temperature',
                           'relative_humidity', 'pressure', 'wind_direction', 'wind_speed_aver',
                           'precipitation']]
        lr_col = lr_col.fillna(value=0.0)
        start_meteodata = np.array(lr_col)
        start_meteodata = np.nan_to_num(start_meteodata)

        # Прогноз при помощи физической модели
        forecast = dm.predict_period(start_variables, start_meteodata, period=7)

        # Трансформация расходов в уровни
        stage = convert_discharge_into_stage_max(model=convert_model,
                                                 forecast=forecast,
                                                 months=local_sb_df['month'])

        # Функция перерасчета предсказанных значений stage_max в delta_stage_max
        deltas = convert_max_into_delta(current_level, stage)
        forecasts.extend(deltas)

    df_submit['delta_stage_max'] = forecasts
    path_for_save = 'submissions/submission_data/phys_model_3045'
    file_name = '/home/maslyaev/hton/edn/models/phys_model/discharge_3042_into_stage_3045.pkl'

    df_submit.to_csv(os.path.join(path_for_save, file_name), index=False)

def load_validation_snowcover(filenames : list):
    scover_partials = []
    for filename in filenames:
        scover_partials.append(pd.read_csv(filename, parse_dates=['date']))
    scover = pd.concat(scover_partials)
    scover = scover.drop_duplicates(subset = ['date']).drop('Unnamed: 0', axis = 1).reset_index()
    print(scover.columns)
    return scover

def smooth_snowcover(data, sigma):
    data['snowcover'] = gaussian_filter1d(data['snowcover'], sigma)
    return data

def filter_period(data, start, end):
    assert 'date' in data.columns
    mask = (data['date'] >= start) & (data['date'] <= end)
    return data.loc[mask].interpolate()
    
def snowcover_preprocessing(snowcover, sigma = 3):
    snowcover = snowcover.interpolate(method = 'linear')
    snowcover['snowcover'] = gaussian_filter1d(snowcover['snowcover'], sigma)
    return snowcover

def get_srm_forecast():
    cwd = os.getcwd()
    meteo_df = get_meteo_df()
    meteo_df = meteo_df.drop(labels = ['precipitation'], axis = 1)
    
    # Обучаем модель для периода на 2004 год:
    train_start_date = '2003-12-31'
    train_end_date = '2004-12-30'
    train_meteo_period = filter_period(meteo_df, train_start_date, train_end_date)

    train_rainfall = pd.read_csv('/home/maslyaev/hton/edn/data/rainfall_data_train.csv', parse_dates=['date'])
    train_rainfall = filter_period(train_rainfall, train_start_date, train_end_date)
    
    train_scover = pd.read_csv('/home/maslyaev/hton/edn/data/snowcover_2004.csv', parse_dates=['date'])
    train_scover = filter_period(train_scover, train_start_date, train_end_date)
    train_scover_filtered = snowcover_preprocessing(train_scover)

    # Обучение физической модели
    dm = fit_3045_phys_model(train_meteo_period, train_scover_filtered, train_rainfall)

    # Загружаем модель машинного обучения, которая будет конвертировать расходы в уровни
    convert_model = load_converter(filename = '/home/maslyaev/hton/kek/serialised/level_model.pkl')
    
    val_start_date = '2009-10-17'
    val_end_date = '2011-12-31'
    val_meteo_period = filter_period(meteo_df, val_start_date, val_end_date)

    val_rainfall = pd.read_csv('/home/maslyaev/hton/edn/data/rainfall_data_test.csv', parse_dates=['date'])
    val_rainfall = filter_period(val_rainfall, val_start_date, val_end_date)

    filenames = ['/home/maslyaev/hton/edn/data/snowcover_val_1.csv', 
                 '/home/maslyaev/hton/edn/data/snowcover_val_2.csv', 
                 '/home/maslyaev/hton/edn/data/snowcover_val_3.csv']    
    val_scover = load_validation_snowcover(filenames)
    val_scover = filter_period(val_scover, val_start_date, val_end_date)
    val_scover_filtered = snowcover_preprocessing(val_scover)

    forecast = apply_3045_phys_model(dm, 7, val_meteo_period, 
                                     val_scover_filtered, val_rainfall)
    forecast = np.array(forecast)
        # Трансформация расходов в уровни
    stage_max = convert_discharge_into_stage_max(model=convert_model,
                                             forecast=forecast,
                                             months=[date.month for date in val_meteo_period['date']])
    return dm, stage_max