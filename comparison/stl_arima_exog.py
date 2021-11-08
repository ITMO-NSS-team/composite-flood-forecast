from statsmodels.tsa.api import STLForecast
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from model.metrics import smape, nash_sutcliffe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../data/level_ts_train.csv', parse_dates=['date'])
df2 = pd.read_csv('../data/level_time_series.csv', parse_dates=['date'])

forecast_len = 805
stations = [ 3027, 3028, 3029, 3030, 3035, 3041, 3045, 3050, 3230]

NSE = []
MAE = []
SMAPE = []

for i in range(len(stations)):
    sub_df = df[df['station_id'] == stations[i]]
    if i!=0:
        exog_sub_df = df[df['station_id'] == stations[i - 1]]
    else:
        exog_sub_df = df[df['station_id'] == stations[1]]

    stlf = STLForecast(np.array(sub_df['stage_max']), ARIMA, period=345, model_kwargs={"exog": np.array(exog_sub_df['stage_max']), "order": (4, 1, 4)})
    res = stlf.fit()
    real = df2[df2['station_id'] == stations[i]]['stage_max'][:forecast_len]
    if i!=0:
        exog_real = df2[df2['station_id'] == stations[i-1]]['stage_max'][:forecast_len]
    else:
        exog_real = df2[df2['station_id'] == stations[1]]['stage_max'][:forecast_len]

    forecasts = res.forecast(forecast_len, exog=exog_real)
    real_dates = df2[df2['station_id'] == stations[i]]['date'][:forecast_len]

    print(f'Station {str(stations[i])}')
    print(f'NSE = {nash_sutcliffe(np.array(real), np.array(forecasts))}')
    NSE.append(nash_sutcliffe(np.array(real), np.array(forecasts)))
    print(f'MAE = {mean_absolute_error(np.array(real), np.array(forecasts))}')
    MAE.append(mean_absolute_error(np.array(real), np.array(forecasts)))
    print(f'SMAPE = {smape(np.array(real), np.array(forecasts))}')
    SMAPE.append(smape(np.array(real), np.array(forecasts)))