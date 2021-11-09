import os
import pickle
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

from model.phys_model.calculate_levels import convert_max_into_delta
from model.phys_model.srm_model import fit_3045_phys_model, get_const_for_3045, apply_3045_phys_model
from model.phys_model.train_converter import get_meteo_df


""" A physical model for predicting water discharge at hydro gauge 3045.
There are no discharge data at hydro gauge 3045, but there are water level values. 

The model requires values from neighboring stations:
    - 3036 (discharge), station upstream of the river
    - 3042 (discharge), a station close to 3045. We can assume that 
The flow values at station 3042 are well correlated with the flow and water level 
at modeled station 3045.

The physical model predicts flow rates, not levels! The recalculation of flow rates into levels 
is done using the ML model (random forest). 
"""


def load_SRM(filename='SRM.pkl'):
    """ Load fitted and serialised SRM model from pkl file """
    with open(filename, 'rb') as f:
        clf = pickle.load(f)
    return clf
    

def load_converter(filename='level_model.pkl'):
    """ Load fitted and serialised "discharge into levels" model from pkl file """
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
    """ Recalculation of the flow rate in the level based on the trained model """
    forecast = np.array(forecast).reshape((-1, 1))
    months = np.array(months).reshape((-1, 1))

    x_test = np.hstack((forecast, months))
    # Predicting the level based on discharge
    stage_max = model.predict(x_test)
    stage_max = np.ravel(stage_max)
    return stage_max


def srm_approach():
    meteo_df = get_meteo_df()

    # Train the model for the period for 2004:
    start_date = '2003-12-31'
    end_date = '2004-12-30'
    mask = (meteo_df['date'] >= start_date) & (meteo_df['date'] <= end_date)
    meteo_period = meteo_df.loc[mask]
    meteo_period = meteo_period.dropna()

    scover = pd.read_csv('/home/maslyaev/hton/edn/snowcover_2004.csv', parse_dates=['date'])
    scover_filtered = snowcover_preprocessing(scover)

    # Train physical model
    dm = fit_3045_phys_model(meteo_period, scover_filtered)

    # Load a machine learning model that will convert discharge into levels
    convert_model = load_converter()

    # Load all the necessary data
    df_merge = get_all_data_for_3045_forecasting()

    df_submit = pd.read_csv('submissions/sample_submissions/sample_sub_4.csv', parse_dates=['date'])
    df_submit = df_submit[df_submit['station_id'] == 3045]

    # Unchangeable parameters for station 3045
    params = get_const_for_3045()
    const_area = params['area']
    section_to = params['section_to']
    lapse = params['lapse']
    h_mean = params['h_mean']
    h_st = params['h_st']

    # Forecasts of the algorithm
    forecasts = []
    for i in range(0, len(df_submit), 7):
        row = df_submit.iloc[i]
        current_date_for_pred = row['date']

        # Get data only up to the desired date, for which the forecast are given
        local_df = df_merge[df_merge['date'] < current_date_for_pred]
        # Take the current value of the water level at station 3045
        current_level = np.array(local_df['stage_max'])[-1]

        # Get data for current month
        local_sb_df = df_submit.iloc[i:i+7]

        # Give predictors to model
        last_row = local_df.iloc[-1]
        # Calculate parameters for temperature
        tmp = last_row['air_temperature'] + lapse * (h_mean - h_st) * 0.01
        snow_cover = last_row['snow_coverage_station']
        rainfall = last_row['precipitation']
        disch_3042 = last_row['discharge_3042']
        disch_3036 = last_row['discharge_3036']
        start_variables = np.array([tmp, snow_cover, const_area, rainfall, disch_3042, disch_3036])
        start_variables = np.nan_to_num(start_variables)
        start_variables = tuple(start_variables)

        # Weather conditions
        lr_col = last_row[['snow_height_y', 'snow_coverage_station', 'air_temperature',
                           'relative_humidity', 'pressure', 'wind_direction', 'wind_speed_aver',
                           'precipitation']]
        lr_col = lr_col.fillna(value=0.0)
        start_meteodata = np.array(lr_col)
        start_meteodata = np.nan_to_num(start_meteodata)

        # Forecast from physical model
        forecast = dm.predict_period(start_variables, start_meteodata, period=7)

        # Transform discharge into levels
        stage = convert_discharge_into_stage_max(model=convert_model,
                                                 forecast=forecast,
                                                 months=local_sb_df['month'])

        # Recalculating predicted stage_max values to delta_stage_max (if it is required)
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
    scover = scover.drop_duplicates(subset=['date']).drop('Unnamed: 0', axis=1).reset_index()
    return scover


def smooth_snowcover(data, sigma):
    data['snowcover'] = gaussian_filter1d(data['snowcover'], sigma)
    return data


def filter_period(data, start, end):
    assert 'date' in data.columns
    mask = (data['date'] >= start) & (data['date'] <= end)
    return data.loc[mask].interpolate()


def snowcover_preprocessing(snowcover, sigma=3):
    snowcover = snowcover.interpolate(method='linear')
    snowcover['snowcover'] = gaussian_filter1d(snowcover['snowcover'], sigma)
    return snowcover
