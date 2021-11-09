import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


""" A physical model for predicting water flow at hydro gauge 3045.
There are no discharge data at hydro gauge 3045, but there are level values. 

The model requires values from neighboring stations:
    - 3036 (discharge), station upstream of the river
    - 3042 (discharge), a station close to 3045. We can assume that 
The flow values at station 3042 are well correlated with the flow and water level 
at modeled station 3045.

The physical model predicts flow rates, not levels! The recalculation of flow rates into levels 
is done using the ML model (random forest). 
"""


def get_meteo_df():
    """ The function returns the interpolated weather parameters at node 3045 """
    meteo_snow = pd.read_csv('../data/meteo_data/no_gap_1day/no_gap_meteo_1day_int_3045.csv',
                             parse_dates=['date'])
    # Meteorological parameters: pressure and wind direction at station 3045
    meteo_press = pd.read_csv('../data/meteo_data/no_gap_3hour/no_gap_meteo_3hour_int_3045_press.csv',
                              parse_dates=['date'])
    meteo_wind = pd.read_csv('../data/meteo_data/no_gap_3hour/no_gap_meteo_3hour_int_3045_wind.csv',
                             parse_dates=['date'])
    # Combining dataframes by station id
    meteo_integrated = pd.merge(meteo_snow, meteo_press, on=['station_id', 'date'])
    meteo_integrated = pd.merge(meteo_integrated, meteo_wind, on=['station_id', 'date'])

    return meteo_integrated


if __name__ == '__main__':
    # Get a dataframe with weather parameters
    df_meteo = get_meteo_df()

    df_levels = pd.read_csv('/home/maslyaev/hton/edn/data/meteo_data/4rd_checkpoint/no_gaps_train.csv', parse_dates=['date'])
    df_3042 = df_levels[df_levels['station_id'] == 3042]
    df_3045 = df_levels[df_levels['station_id'] == 3045]

    # Training the model to translate dishcharge at station 3042 into levels at gauge 3045
    df_levels = pd.merge(df_3045, df_3042, on='date', suffixes=['_3045', '_3042'])
    df_levels = df_levels[['date', 'stage_max_3045', 'discharge_3042', 'month_3045']]
    df_levels = df_levels.dropna()

    non_linear_m = RandomForestRegressor()
    x_train = np.array(df_levels[['discharge_3042', 'month_3045']])
    y_train = np.array(df_levels['stage_max_3045'])

    # Fit model
    non_linear_m.fit(x_train, y_train)

    print(x_train)
    print(y_train)

    # Save model
    filename = 'discharge_3042_into_stage_3045.pkl'
    with open(filename, 'wb') as fid:
        pickle.dump(non_linear_m, fid)
