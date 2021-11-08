import os
import pickle
import pandas as pd
import numpy as np
from model.phys_model.calculate_levels import convert_max_into_delta
from model.phys_model.srm_model import fit_3045_phys_model, get_const_for_3045
from model.phys_model.train_converter import get_meteo_df
from model.phys_model.launch_srm_model import load_converter, get_all_data_for_3045_forecasting


if __name__ == '__main__':
    # TODO добавить визуализацию того, как хорошо прогнозирует физическая модель
    pass
