import timeit
import numpy as np
from sklearn.metrics import mean_squared_error
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def prepare_ts_input_data(time_series: np.array):
    """ Warp time series into InputData for FEDOT framework """
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=7))
    input_data = InputData(idx=np.arange(0, len(time_series)),
                           features=time_series,
                           target=time_series, task=task,
                           data_type=DataTypesEnum.ts)
    return input_data


def prepare_table_input_data(features: np.array, target: np.array):
    """ Warp tabular data into InputData for FEDOT framework """
    task = Task(TaskTypesEnum.regression)
    input_data = InputData(idx=np.arange(0, len(features)),
                           features=features,
                           target=target, task=task,
                           data_type=DataTypesEnum.table)
    return input_data
