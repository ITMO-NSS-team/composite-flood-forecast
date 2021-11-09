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
stations = [3045]

NSE = []
MAE = []
SMAPE = []

for station in stations:
    sub_df = df[df['station_id'] == station]
    stlf = STLForecast(sub_df['stage_max'], ARIMA, period=345, model_kwargs={"order": (4, 1, 4)})
    res = stlf.fit()
    forecasts = res.forecast(forecast_len)
    real = df2[df2['station_id'] == station]['stage_max'][:forecast_len]
    real_dates = df2[df2['station_id'] == station]['date'][:forecast_len]

    print(f'Station {str(station)}')
    print(f'NSE = {nash_sutcliffe(np.array(real), np.array(forecasts))}')
    NSE.append(nash_sutcliffe(np.array(real), np.array(forecasts)))
    print(f'MAE = {mean_absolute_error(np.array(real), np.array(forecasts))}')
    MAE.append(mean_absolute_error(np.array(real), np.array(forecasts)))
    print(f'SMAPE = {smape(np.array(real), np.array(forecasts))}')
    SMAPE.append(smape(np.array(real), np.array(forecasts)))

    plt.title(f'Station {str(station)}')
    plt.plot(sub_df['date'][-365:], sub_df['stage_max'][-365:], c='g', linewidth=1)
    plt.axvline(x=pd.to_datetime('2006-01-01'), c='black', linewidth=0.8)
    plt.plot(real_dates, real, c='g', linewidth=1, label='Actual time series')
    plt.plot(real_dates, forecasts, c='r', linewidth=1, label='STL ARIMA forecast')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Maximum level value, cm', fontsize=12)
    plt.legend()
    plt.xticks(rotation=30)
    plt.grid()
    plt.show()

print(f'\nMEAN NSE: {np.mean(NSE)}')
print(f'\nMEAN MAE: {np.mean(MAE)}')
print(f'\nMEAN SMAPE: {np.mean(SMAPE)}')