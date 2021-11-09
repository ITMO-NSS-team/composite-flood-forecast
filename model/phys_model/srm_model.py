import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib


import scipy.optimize as optimize
import scipy as sp

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans


pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.max_columns', 15)

font = {'family': 'normal',
        'weight': 'normal',
        'size': 10}

matplotlib.rc('font', **font)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 14),
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'xx-large',
          # 'axes.weight': 'bold',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large'}
pylab.rcParams.update(params)


def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.cosine):
    """
    Function taken from
    https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan
    """

    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int) * -1

    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(x_new, x_core) < dbscan_model.eps:
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break
    return y_new


def opt_sourceless_equation(params, *variables):
    """
    params : np.array
        parameters of the runoff model, params[0] - discharge recession coeff,
        params[1] - downstream flow coeff;

    variables : tuple
        physical variables of the runoff model
        variables[0], np.ndarray - degree days; variables[1], np.ndarray - fractional snow cover; 
        variables[2], np.ndarray - watershed area; variables[3], np.ndarray - rainfall;
        variables[4], np.ndarray - discharge; variables[5], np.ndarray - upstream point data;
    """

    f_py = lambda t_idx: (params[0] * variables[4][t_idx - 1] +
                          params[1] * (variables[4][t_idx - 1] - variables[5][t_idx - 1]) - 
                          variables[4][t_idx])
    f_vect = np.vectorize(f_py)
    errs = f_vect(np.arange(1, variables[4].size))
    res = np.sum(np.abs(errs))
    return res


def model_equation(params, variables):
    """
    params : np.array
        parameters of the runoff model, params[0] - discharge recession coeff, 
        params[1] - downstream flow coeff, params[2] - snow runoff coeff,
        params[3] - rain_runoff_coeff, params[4] - degree day factor;
        ;

    variables : tuple
        physical variables of the runoff model
        variables[0] - degree days, variables[1] - fractional snow cover, variables[2] - watershed area,
        variables[3] - rainfall, variables[4] - discharge, variables[5] - upstream point data
    """
    k = 10000. / 86400.
    return ((params[2] * params[4] * variables[0] * variables[1] +
             params[3] * variables[3]) * variables[2] * k * (1 - params[0]) +
            params[0] * variables[4] +
            params[1] * (variables[4] - variables[5]))


def opt_equation(params, *variables):
    """
    params : np.array
        parameters of the runoff model, params[0] - snow runoff coeff, params[1] - rain_runoff_coeff,
        params[2] - degree day factor.

    variables : tuple
        physical variables of the runoff model
        variables[0], np.ndarray - degree days; variables[1], np.ndarray - fractional snow cover; 
        variables[2], int/float - watershed area; variables[3], np.ndarray - rainfall;
        variables[4], np.ndarray - discharge; variables[5], np.ndarray - upstream point data;
        variables[6], float - discharge recession coeff, obtained from previous optimization;
        variables[7], float - downstream flow coeff, obtained from previous optimization.
    """
    k = 10000. / 86400.

    f_py = lambda t_idx: ((params[0] * params[2] * variables[0][t_idx - 1] * variables[1][t_idx - 1] +
                           params[1] * variables[3][t_idx - 1]) * variables[2] * k * (1 - variables[6]) +
                          variables[6] * variables[4][t_idx - 1] +
                          variables[7] * (variables[4][t_idx - 1] - variables[5][t_idx - 1]) - variables[4][t_idx])
    f_vect = np.vectorize(f_py)
    res = np.sum(np.abs(f_vect(np.arange(1, variables[4].size))))
    return res


def mask_by_recession(data : pd.DataFrame):
    RAINFALL_EPS = 0.04
    SNOWCOVER_EPS = 0.2
    ICEMELT_TEMP = 0    
    
    data_rec_mask = (((data['precipitation'] < RAINFALL_EPS) & (data['snowcover'] < SNOWCOVER_EPS)) | 
                     (data['air_temperature'] < ICEMELT_TEMP))
    return data_rec_mask


def select_data(data, mask, gauge_params):
    param_columns = ['snow_height', 'snowcover',
                     'air_temperature', 'relative_humidity',
                     'pressure', 'wind_direction', 'wind_speed_aver',
                     'precipitation']
    area = gauge_params['area']
    lapse = gauge_params['lapse']
    h_mean = gauge_params['h_mean']
    h_st = gauge_params['h_st']
    
    data_section = data.loc[mask]
    data_section = data_section.interpolate()
    
    temps = data_section.air_temperature.to_numpy()
    degree_days = temps + lapse * (h_mean - h_st) * 0.01
    snow_cover = data_section.snowcover.to_numpy()
    total_precip = data_section.precipitation.to_numpy()

    # Let's modify the values of precipitation
    rainfall = rainfall_convert(total_precip, temps)

    upstream_discharge = data_section.discharge_3036.to_numpy()
    station_discharge = data_section.discharge_3042.to_numpy()
    variables = (degree_days, snow_cover, area, rainfall,
                  station_discharge, upstream_discharge)
    return variables, data_section[param_columns].to_numpy()
    

class DischargeModel(object):
    def __init__(self):
        self.params_list = ['degree_day_factor', 'downstream flow', 'snow_runoff', 
                            'rain_runoff', 'discharge_recession',]
        self.var_list = ['degree_days', 'frac_snow_cover', 'area', 'rainfall', 
                         'point_discharge', 'upsteam_discharge']

    def get_clusters(self, data, eps=0.5, base_clusters=3):
        self.data_shape = data.shape
        self.clustering_method = 'DBSCAN'
        self.pca = PCA(n_components=2)
        self.pca.fit(data)
        # print(self.pca.explained_variance_ratio_)
        data_transformed = self.pca.transform(data)

        self.scaler = StandardScaler()
        data_transformed_scaled = self.scaler.fit_transform(data_transformed)

        self.clustering = DBSCAN(eps=0.3, min_samples=10).fit(data_transformed_scaled)
        if len(set(self.clustering.labels_)) < 5:
            self.clustering_method = 'KMeans'
            self.clustering = KMeans(n_clusters=base_clusters).fit(data_transformed_scaled)

    def get_params(self, variables, data):
        """
        variables : tuple
            physical variables of the runoff model
            variables[0] - degree days, variables[1] - fractional snow cover, variables[2] - watershed area,
            variables[3] - rainfall, variables[4] - discharge, variables[5] - upstream point data
        """
        assert isinstance(variables, tuple)

        self.get_clusters(data)

        bounds = ((0., 1.), (0., 1.),
                  (0., 1.), (0., 0.8),
                  (-10, 10))

        self.cluster_params = {}
        for cluster_label in set(self.clustering.labels_):
            if cluster_label != -1:
                indexes = list(np.where(self.clustering.labels_ == cluster_label))
                var_temp = []
                for var in variables:
                    if isinstance(var, (int, float)):
                        var_temp.append(var)
                    else:
                        var_temp.append(var[indexes])
                self.cluster_params[cluster_label] = optimize.differential_evolution(opt_equation, bounds,
                                                                                     args=var_temp)

    def get_params_simplified(self, data, gauge_params):
        """
        variables : tuple
            physical variables of the runoff model
            variables[0] - degree days, variables[1] - fractional snow cover, variables[2] - watershed area,
            variables[3] - rainfall, variables[4] - discharge, variables[5] - upstream point data
        """

        assert isinstance(data, pd.DataFrame)
    
        bounds_1 = ((0., 1.), (-10, 10))
        bounds_2 = ((0., 1.), (0., 1.),
                  (0., 1.))

        recession_mask = mask_by_recession(data)
        
        # training of recession and transfer coefficients 
        variables_section, data_section = select_data(data, recession_mask, gauge_params)
        
        rec_params = optimize.differential_evolution(opt_sourceless_equation, bounds_1,
                                                     args=variables_section).x
        
        print('rec_params:', rec_params)        
        # Training of water input parameters
        variables_section, data_section = select_data(data, ~recession_mask, gauge_params)
        
        self.get_clusters(data_section)
        
        self.cluster_params = {}
        self.cluster_params['recession_only'] = np.concatenate((rec_params, np.zeros(3)))
        
        print(self.cluster_params['recession_only'])
        time.sleep(5)

        for cluster_label in set(self.clustering.labels_):
            if cluster_label != -1:
                indexes = list(np.where(self.clustering.labels_ == cluster_label))
                var_temp = []
                for var in variables_section:
                    if isinstance(var, (int, float)):
                        var_temp.append(var)
                    else:
                        var_temp.append(var[indexes])
                var_temp.extend(list(rec_params))
                inp_params = optimize.differential_evolution(opt_equation, bounds_2, args=var_temp).x
                self.cluster_params[cluster_label] = np.concatenate((rec_params, inp_params))
                print(self.cluster_params[cluster_label])
                time.sleep(5)

    def predict_1day(self, variables, meteodata):
        RAINFALL_EPS = 0.04
        SNOWCOVER_EPS = 0.2
        ICEMELT_TEMP = 0        
        
        # print(meteodata)
        meteodata = np.array(meteodata)
        if meteodata[2] > ICEMELT_TEMP and (meteodata[1] > RAINFALL_EPS 
                                            or meteodata[-1] > SNOWCOVER_EPS):
            mdata_transformed = np.dot(meteodata.reshape((1, -1)), self.pca.components_.T)
            mdata_transformed = self.scaler.transform(mdata_transformed)
            if self.clustering_method == 'KMeans':
                variable_cluster = self.clustering.predict(mdata_transformed)
            elif self.clustering_method == 'DBSCAN':
                variable_cluster = dbscan_predict(self.clustering, mdata_transformed)
            return model_equation(self.cluster_params[variable_cluster[0]], variables)
        else:
            return model_equation(self.cluster_params['recession_only'], variables)

    def predict_period(self, variables, meteodata, period=None):
        """
        period = meteodata.shape[0]
        """
        raise NotImplementedError('Not refactored yet')

        if period is None:
            preds = np.empty(meteodata.shape[0])
        else:
            preds = np.empty(period)
            meteodata = np.stack([meteodata for i in np.arange(period)])

        for idx in np.arange(preds.size):
            preds[idx] = self.predict_1day(variables, meteodata[idx, :])
            variables = list(variables)
            variables[4] = preds[idx]
            variables = tuple(variables)

        return preds


def rainfall_convert(total_precip, temps):
    rainfall = np.empty_like(total_precip)
    for idx in np.arange(total_precip.size):
        rainfall[idx] = total_precip[idx] if temps[idx] > 0 else 0

    return rainfall


def get_const_for_3045():
    params = {'area': 897000 - 770000,
              # 'section_to': 5700,
              'lapse': 0.65,
              'h_mean': 360.0,
              'h_st': 98.0}
    return params


def fit_3045_phys_model(*meteo_dataframes):
    """
    The function trains the physical model for hydro gauge number 3045
    """
    # Obtain the constants for this item
    gauge_params = get_const_for_3045()

    data_partial = meteo_dataframes[0]
    for idx, frame in enumerate(meteo_dataframes[1:]):
        data_partial = pd.merge(left=data_partial, right=frame, how='left', on='date')
    
    print('data shape', data_partial.shape)
    dm = DischargeModel()

    # Dataframe with discharge values by station
    river = pd.read_csv('/home/maslyaev/hton/edn/data/4rd_checkpoint/no_gaps_train.csv', parse_dates=['date'])
    columns = ['date', 'station_id', 'discharge']
    river_3042 = river[river.station_id == 3042][columns]
    river_3036 = river[river.station_id == 3036][columns]
    # Leave only the discharge values
    rivers = pd.merge(left=river_3042, right=river_3036, how='inner',
                      on='date', suffixes=('_3042', '_3036'))[['date', 'discharge_3042', 'discharge_3036']]
    # Combine dataframes with discharge and meteorological parameters
    data_full = pd.merge(left=data_partial, right=rivers, how='left', on='date')
    dm.get_params_simplified(data_full, gauge_params)

    return dm


def apply_3045_phys_model(dm, river_dataframe, forecast_intervals = 7, *meteo_dataframes):
    data_partial = meteo_dataframes[0]
    for idx, frame in enumerate(meteo_dataframes[1:]):
        data_partial = pd.merge(left=data_partial, right=frame, how='left', on='date',
                                suffixes=False)

    data_partial = data_partial.interpolate(limit_direction='both')

    columns = ['date', 'station_id', 'discharge']
    river_3042 = river_dataframe[river_dataframe.station_id == 3042][columns]
    river_3036 = river_dataframe[river_dataframe.station_id == 3036][columns]
    # Leave only the discharge values
    rivers = pd.merge(left=river_3042, right=river_3036, how='inner',
                      on='date', suffixes=('_3042', '_3036'))[['date', 'discharge_3042', 'discharge_3036']]
    # Combine dataframes with discharge and meteorological parameters
    data_full = pd.merge(left=data_partial, right=rivers, how='left', on='date')
    param_columns = ['snow_height', 'snowcover', 'air_temperature',
                     'relative_humidity', 'pressure', 'wind_direction',
                     'wind_speed_aver', 'precipitation']
    gauge_params = get_const_for_3045()    
    area = gauge_params['area']
    lapse = gauge_params['lapse']
    h_mean = gauge_params['h_mean']
    h_st = gauge_params['h_st'] 
    
    forecasts = []
    for slice_idx in np.arange(start = 0, stop = data_full.shape[0], step = forecast_intervals):
        for time_fcast_idx in np.arange(forecast_intervals):
            global_idx = slice_idx + time_fcast_idx
            if global_idx >= data_full.shape[0]:
                break
            meteoparams = [data_full.iloc[global_idx][key] for key in param_columns]
            
            temperature = data_full.iloc[global_idx]['air_temperature']
            degree_days = temperature + lapse * (h_mean - h_st) * 0.01
            snow_cover = data_full.iloc[global_idx]['snowcover']
            total_precip = data_full.iloc[global_idx]['precipitation']
        
            # Modify rainfall values
            ICEMELT_TEMP = 0            
            rainfall = total_precip if temperature > ICEMELT_TEMP else 0
        
            upstream_discharge = data_full.iloc[slice_idx]['discharge_3036']
            if time_fcast_idx == 0:
                station_discharge = data_full.iloc[global_idx-1]['discharge_3042']           
            else: 
                station_discharge = forecasts[-1]
            variables = (degree_days, snow_cover, area, rainfall, station_discharge, upstream_discharge)
            forecasts.append(dm.predict_1day(variables, meteoparams))   
    return np.array(forecasts)
