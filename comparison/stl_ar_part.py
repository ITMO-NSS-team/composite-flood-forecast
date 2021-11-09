from statsmodels.tsa.api import STLForecast
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error
from model.metrics import smape, nash_sutcliffe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('../data/level_ts_train.csv', parse_dates=['date'])
df2 = pd.read_csv('../data/level_time_series.csv', parse_dates=['date'])
df2 = pd.concat([df, df2], ignore_index=True)

forecast_len = 805
stations = [3045]

NSE = []
MAE = []
SMAPE = []
only_spring = False
for station in stations:
    sub_df = df2[df2['station_id'] == station][:-forecast_len]
    stlf = STLForecast(sub_df['stage_max'], AutoReg, period=365, model_kwargs={"lags": 3})
    res = stlf.fit()
    forecasts = np.array(res.forecast(forecast_len))
    real = df2[df2['station_id'] == station]['stage_max'][-forecast_len:].to_numpy()
    real_dates = df2[df2['station_id'] == station]['date'][-forecast_len:].to_numpy()

    if only_spring == True:
        result_df = pd.DataFrame()
        result_df['dates'] = real_dates
        result_df['real'] = real
        result_df['forecast'] = forecasts
        result_df['dates'] = pd.to_datetime(result_df['dates'])

        result_df = result_df[(5 <= result_df['dates'].dt.month) & (result_df['dates'].dt.month <= 7)]

        real = result_df['real']
        forecasts = result_df['forecast']
        real_dates = result_df['dates']

    print(f'Station {str(station)}')
    print(f'NSE = {nash_sutcliffe(np.array(real), np.array(forecasts))}')
    NSE.append(nash_sutcliffe(np.array(real), np.array(forecasts)))
    print(f'MAE = {mean_absolute_error(np.array(real), np.array(forecasts))}')
    MAE.append(mean_absolute_error(np.array(real), np.array(forecasts)))
    print(f'SMAPE = {smape(np.array(real), np.array(forecasts))}')
    SMAPE.append(smape(np.array(real), np.array(forecasts)))

    plt.title(f'Station {str(station)}')
    plt.plot(sub_df['date'][-365:], sub_df['stage_max'][-365:], c='g', linewidth=1)
    plt.axvline(x=pd.to_datetime('2009-10-18'), c='black', linewidth=0.8)
    plt.plot(real_dates, real, c='g', linewidth=1, label='Actual time series')
    plt.plot(real_dates, forecasts, c='orange', linewidth=1, label='STL AR forecast')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Maximum level value, cm', fontsize=12)
    #plt.legend()
    plt.xticks(rotation=30)
    plt.grid()
    plt.show()

print(f'\nMEAN NSE: {np.mean(NSE)}')
print(f'\nMEAN MAE: {np.mean(MAE)}')
print(f'\nMEAN SMAPE: {np.mean(SMAPE)}')