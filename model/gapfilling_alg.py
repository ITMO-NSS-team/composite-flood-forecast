import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.utilities.ts_gapfilling import ModelGapFiller

folder_path = 'sub_datasets_no_gaps/with_nan'
output_folder_path = 'sub_datasets_no_gaps/no_gaps'
# Fill in the gaps with FEDOT (for weather parameters and time
# series with short gaps (less than six months))
###########################
# Bi-directional forecast #
###########################
for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)

    new_df_open = pd.read_csv(os.path.join(folder_path, file))
    df = pd.read_csv(file_path)
    new_df = df
    df = df.fillna(9999)
    gap_names = ['stage_avg', 'stage_min', 'stage_max']
    for gap_name in gap_names:

        gap_data = df[gap_name]
        print(len(gap_data))

        # Filling in gaps using pipeline from FEDOT
        node_lagged = PrimaryNode('lagged')
        node_lagged.custom_params = {'window_size': 50}
        node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
        ridge_pipeline = Pipeline(node_ridge)
        ridge_gapfiller = ModelGapFiller(gap_value=9999, pipeline=ridge_pipeline)

        without_gap_arr_ridge = ridge_gapfiller.forward_filling(gap_data)
        new_df[gap_name] = without_gap_arr_ridge

    new_df.to_csv(os.path.join(output_folder_path, file), index=False)


stations = df['station_id'].unique()
for station in stations:
    sub_df = df[df['station_id'] == station]
    print(station)

    sub_df.to_csv('sub_datasets/train_' + str(station) + '.csv', index=False)

folder_path = 'sub_datasets'
for file in os.listdir(folder_path):
    sub_df = pd.read_csv(os.path.join(folder_path, file))
    sub_df['date'] = pd.to_datetime(sub_df['date'])
    sub_df.set_index('date', inplace=True)
    idx = pd.date_range(start='1/1/1985', end='31/10/2019', freq='D')

    new_df = pd.DataFrame()
    cols = ['stage_avg', 'stage_min', 'stage_max', 'temp', 'water_code',
            'ice_thickness', 'snow_height', 'place', 'discharge', 'year',
            'month', 'day', 'delta_stage_max']
    for column in cols:
        stage_avg_df = sub_df[column]
        stage_avg_df.index = pd.DatetimeIndex(stage_avg_df.index)
        stage_avg_df = stage_avg_df.reindex(idx, fill_value=np.nan)
        new_df[column] = stage_avg_df
    print(new_df)
    new_df['station_id'] = sub_df['station_id'][0]
    new_df.to_csv('sub_datasets_no_gaps/with_nan/no_gap_' + file, index=True)

########################
# Gaps for temperature #
########################
from statsmodels.tsa.seasonal import seasonal_decompose

folder_path = 'sub_datasets_no_gaps/no_gaps'
for file in os.listdir(folder_path):
    df = pd.read_csv(os.path.join(folder_path, file))
    df = df.rename(columns={"Unnamed: 0": "date"})
    df['date'] = pd.to_datetime(df['date'])

    df[['temp2']] = df[['temp']].fillna(value=0)

    plt.rcParams["figure.figsize"] = (20, 3)
    plt.plot(df['date'], df['temp'])
    plt.show()

    plt.rcParams["figure.figsize"] = (20, 10)
    result = seasonal_decompose(df['temp2'], model='additive', period=365)
    result.plot()
    plt.show()

    plt.rcParams["figure.figsize"] = (20, 3)
    seas = result.seasonal + result.trend.median()
    plt.plot(seas)
    plt.show()

    df['temp'][df['temp'].isna()] = seas
    plt.plot(df['temp'])
    plt.show()

    df = df.drop(['temp2'], axis=1)
    df.to_csv(os.path.join(folder_path, file), index=False)

#####################
# Gaps for disharge #
#####################
folder_path = 'sub_datasets_no_gaps/no_gaps'
st_with_discharge = [3042]
for file in os.listdir(folder_path):
    if file in st_with_discharge:
        print(file)
        df = pd.read_csv(os.path.join(folder_path, file))
        df['date'] = pd.to_datetime(df['date'])

        df[['discharge2']] = df[['discharge']].fillna(value=0)

        plt.rcParams["figure.figsize"] = (20,3)
        plt.plot(df['date'], df['discharge'])
        plt.show()

        plt.rcParams["figure.figsize"] = (20,10)
        result = seasonal_decompose(df['discharge2'], model='additive', period=365)

        plt.rcParams["figure.figsize"] = (20,3)
        seas = result.seasonal+result.trend.median()
        plt.plot(seas)
        plt.show()

        df['discharge'][df['discharge'].isna()] = seas
        plt.plot(df['discharge'])
        plt.show()

        df = df.drop(['discharge2'], axis=1)
        df.to_csv(os.path.join(folder_path, file), index=False)
