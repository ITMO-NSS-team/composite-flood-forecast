import numpy as np


def smape(y_true, y_pred):
    return np.mean(100 *(2*np.abs((y_true - y_pred)) / (np.abs(y_true) + np.abs(y_pred))))


def nash_sutcliffe(y_true, y_pred):
    return 1-(np.sum((y_pred - y_true)**2)/np.sum((y_true - np.mean(y_true))**2))
