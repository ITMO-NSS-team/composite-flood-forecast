import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from matplotlib import pyplot as plt
import warnings

from typing import Callable

warnings.filterwarnings('ignore')


def interpolation_for_dataset(meteo_df: pd.DataFrame, stations_df: pd.DataFrame,
                              features_to_move: list, stations_ids: list,
                              n_neighbors: int = 2,
                              knn_model: Callable = KNeighborsRegressor,
                              save_path: str = 'file.csv',
                              vis_station_stage: bool = False):
    """ Applies interpolation functions to the desired dataframes.
    Warning: the algorithm is applied on the raw data!

    :param meteo_df: dataframe with meteo information
    :param stations_df: dataframe with level data
    :param features_to_move: signs to be moved from weather stations to hydro gauge
    :param stations_ids: list of station identifiers
    :param n_neighbors: the number of neighbors over which the values are interpolated
    :param knn_model: algorithm for calculating values (you can use a classifier
    or regressor depending on the nature of the features)
    :param save_path: where you want to save the interpolated data
    :param vis_station_stage: whether it is necessary to draw level charts
    """

    # File with the coordinates of gauging stations
    hydro_coord = pd.read_csv('hydro_coord.csv')
    # File with the coordinates of weather stations
    meteo_coord = pd.read_csv('meteo_coord.csv')
    meteo_coord['station_id_meteo'] = meteo_coord['station_id']

    dates = []
    for j, station_id in enumerate(stations_ids):
        print(f'Process hydrological station {station_id}')
        # We leave the data for only one gauging station
        station_df_local = stations_df[stations_df['station_id'] == station_id]

        # Combine dateframes by date
        # To one hydroprost at one moment of time several weather stations are compared
        merged = pd.merge(station_df_local, meteo_df, on='date', suffixes=['_hydro', '_meteo'])

        # We get the coordinates of the gauging station
        df_local_coords = hydro_coord[hydro_coord['station_id'] == station_id]
        x_test = np.array(df_local_coords[['lat', 'lon']])

        # For each attribute in the dateframe with meteorological parameters
        for index, feature in enumerate(features_to_move):
            print(f'Process feature {feature}')

            # For each moment of time for the gauging station
            interpolated_values = []
            start_date = min(station_df_local['date'])
            end_date = max(station_df_local['date'])
            all_dates = pd.date_range(start_date, end_date)
            for current_date in all_dates:
                # We get the combined data for the selected term - one day
                merged_current = merged[merged['date'] == current_date]
                # Adding coordinates to the data for weather stations
                new_merged = pd.merge(merged_current, meteo_coord,
                                      on='station_id_meteo')
                new_merged = new_merged.reset_index()

                try:
                    # According to the coordinates and altitude we predict the
                    # value in the hydropost
                    dataset = new_merged[['lat', 'lon', feature]]
                    # Remove the extremely high values - these are gaps
                    if feature == 'snow_coverage_station':
                        dataset[dataset[feature] > 50] = np.nan
                    else:
                        dataset[dataset[feature] > 9000] = np.nan
                    dataset = dataset.dropna()

                    knn = knn_model(n_neighbors)
                    target = np.array(dataset[feature])
                    knn.fit(np.array(dataset[['lat', 'lon']]), target)
                    interpolated_v = knn.predict(x_test)[0]
                except Exception:
                    # So the values contain omissions
                    interpolated_v = None

                interpolated_values.append(interpolated_v)

            if vis_station_stage:
                # Plotting the graph of the course of levels and
                # the parameter that was interpolated
                fig, ax1 = plt.subplots()
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Maximum level value, cm')
                ax1.plot(station_df_local['date'], station_df_local['stage_max_hydro'], c='blue')
                ax1.tick_params(axis='y')
                plt.grid(c='#DCDCDC')

                ax2 = ax1.twinx()
                ax2.plot(station_df_local['date'], interpolated_values, c='orange')
                ax2.tick_params(axis='y')
                ax2.set_ylabel(feature)
                plt.title(f'Hydro station id - {station_id}')
                plt.show()

            if index == 0:
                # Add dates
                dates.extend(all_dates)
                new_f_values = np.array(interpolated_values).reshape((-1, 1))
            else:
                int_column = np.array(interpolated_values).reshape((-1, 1))
                # Join the additional feature on the right
                new_f_values = np.hstack((new_f_values, int_column))

        if j == 0:
            # Dataframe with interpolated values
            new_station_info = pd.DataFrame(new_f_values, columns=features_to_move)
            new_station_info['station_id'] = [station_id] * len(new_station_info)
        else:
            new_dataframe = pd.DataFrame(new_f_values, columns=features_to_move)
            new_dataframe['station_id'] = [station_id] * len(new_dataframe)

            # Add to the dataframe that has already been generated
            frames = [new_station_info, new_dataframe]
            new_station_info = pd.concat(frames)

    new_station_info['date'] = dates
    new_station_info.to_csv(save_path, index=False)
