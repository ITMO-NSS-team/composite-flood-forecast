import numpy as np
from sklearn.metrics import mean_absolute_error


def smape(y_true, y_pred):
    return np.mean(100 *(2*np.abs((y_true - y_pred)) / (np.abs(y_true) + np.abs(y_pred))))


def nash_sutcliffe(y_true, y_pred):
    return 1-(np.sum((y_pred - y_true)**2)/np.sum((y_true - np.mean(y_true))**2))


metric_by_name = {'smape': smape,
                  'mae': mean_absolute_error,
                  'nse': nash_sutcliffe}
