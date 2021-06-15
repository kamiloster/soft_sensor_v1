import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")
warnings.simplefilter('ignore', ConvergenceWarning)
from plotly.subplots import make_subplots
import base64
import io
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.figure_factory as ff
import dash_html_components as html
import dash_core_components as dcc
import dash
import numpy as np
import pandas as pd
import plotly.express as px
from pandas.tseries.offsets import DateOffset

from tkinter import *
import xlsxwriter
import dash_table
import scipy.optimize as optimization
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
import os
from tkinter import filedialog
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import datetime
import glob
import h5py
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense, Bidirectional, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization, \
    TimeDistributed, RepeatVector
from keras.models import Sequential
import tensorflow as tf

import keras
from keras.optimizers import Adam
from scipy.stats import median_absolute_deviation as mad
from scipy.stats import chi2, f, linregress
import time
import math
import dash_bootstrap_components as dbc
from datetime import datetime


def colnum_string(n):
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string


# Seed value
# Apparently you may use different seed values at each stage
seed_value = 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

# tf.random.set_seed(seed_value)
# for later versions:
tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K

# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
# for later versions:
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


def load_data_Xy():
    return pd.read_csv(folder_path + '/X.csv'), pd.read_csv(folder_path + '/y.csv')


def load_directory():
    root = Tk()
    root.directory = filedialog.askdirectory()

    root.destroy()

    return (root.directory)


def load_sensors_data_for_softsensors():
    data_sensors = pd.read_csv(folder_path + '/sensors_soft_sensors.csv')
    data_sensors['date'] = pd.to_datetime(data_sensors['date'], format='%d/%m/%Y %H:%M:%S')
    data_sensors = data_sensors.fillna(value=0.0)

    return data_sensors


def load_LIMS_data_for_softsensors():
    data_LIMS = pd.read_csv(folder_path + '/LIMS_soft_sensors.csv')

    if 'date' and 'time' in data_LIMS.columns:
        data_LIMS['date'] = pd.to_datetime(data_LIMS['date'], format='%d/%m/%Y')
        data_LIMS['time'] = pd.to_datetime(data_LIMS['time'], format='%H:%M:%S')
        data_LIMS = data_LIMS.fillna(value=0.0)

        return data_LIMS

    if 'date' and not 'time' in data_LIMS.columns:
        data_LIMS['date'] = pd.to_datetime(data_LIMS['date'], format='%d/%m/%Y %H:%M:%S')
        data_LIMS = data_LIMS.fillna(value=0.0)
        data_LIMS['time'] = data_LIMS['date'].dt.time
        data_LIMS['date'] = data_LIMS['date'].dt.date
        data_LIMS['date'] = pd.to_datetime(data_LIMS['date']).dt.strftime('%d/%m/%Y')
        data_LIMS['time'] = pd.to_datetime(data_LIMS['time'], format='%H:%M:%S')

        return data_LIMS


def load_LIMS_data():
    data_LIMS = pd.read_csv(folder_path + '/LIMS.csv')
    data_LIMS['date'] = pd.to_datetime(data_LIMS['date'], format='%d/%m/%Y %H:%M:%S')
    data_LIMS = data_LIMS.fillna(value=0.0)
    labels_LIMS = np.array(data_LIMS.drop('date', axis=1).columns)
    labels = []

    for i in np.unique(labels_LIMS):
        labels.append(i.split('_')[0])

    data_LIMS_separated = dict()

    for i in list(np.unique(np.array(labels))):
        labels_3 = []

        labels_2 = labels_LIMS[pd.Series(labels).str.startswith(i)]

        for j in labels_2:
            labels_3 = np.append(labels_3,
                                 j.split('_')[1].strip('%'))

        data_LIMS['time'] = data_LIMS['date']

        data_LIMS_separated[i] = data_LIMS.loc[:,
                                 data_LIMS.columns.str.startswith(i)]
        data_LIMS_separated[i]['date'] = data_LIMS['date'].dt.date
        data_LIMS_separated[i]['time'] = data_LIMS['date'].dt.time

        data_LIMS_separated[i]['date'] = pd.to_datetime(
            data_LIMS_separated[i]['date'])
        data_LIMS_separated[i]['date'] = data_LIMS_separated[i]['date'].dt.strftime('%d/%m/%Y')

    return data_LIMS['date'], data_LIMS, data_LIMS_separated


def list_h5_files():
    os.chdir(folder_path)
    return glob.glob('*.{}'.format('h5'))


def load_h5(h5_file):
    with h5py.File(folder_path + '/' + h5_file, "r") as h5f:
        data_loaded = h5f.get('/data')[()]
        times_loaded = h5f.get('/time')[()].astype('datetime64[ns]')
        df_data = pd.DataFrame(data=data_loaded, index=times_loaded)

    return df_data


def load_sensors_data(data_date, period_sensors):
    """data_sensors = pd.read_csv(folder_path + '/sensors.csv')
    data_sensors['date'] = pd.to_datetime(data_sensors['date'])
    data_sensors_date = data_sensors['date']

    labels = pd.read_csv(folder_path + '/' + 'labels.csv')

    data_sensors = data_sensors.loc[:, labels['Name'].loc[(labels['To include'] == 'Yes') & (labels['Rejection'] != 'Yes')]]
    data_sensors['date'] = data_sensors_date

    return data_sensors, labels"""

    # start_date = np.datetime64('2016-07-01T00:00:00.000000')
    # end_date = np.datetime64('2020-05-12T21:06:00.000000')

    df_data = pd.DataFrame([])

    h5_list = list_h5_files()

    print('A list of available h5 files: {}'.format(h5_list))

    if data_date.all() == None:

        data_loaded = load_h5(h5_list[0])

        start_date = data_loaded.index.min()
        end_date = data_loaded.index.min() + pd.Timedelta(int(period_sensors * 30), unit="d")

        for i in h5_list:
            print('Currently processing this h5 file: ' + i)

            data_loaded = load_h5(i)

            if data_loaded.index.max() < start_date:
                pass

            if data_loaded.index.max() > start_date:

                if data_loaded.index.max() < end_date:
                    data_loaded = data_loaded.loc[start_date - np.timedelta64(1, 'D'):end_date + np.timedelta64(1, 'D'),
                                  :]

                    df_data = pd.concat([df_data, data_loaded], axis=0)

                if data_loaded.index.max() > end_date:
                    data_loaded = data_loaded.loc[start_date - np.timedelta64(1, 'D'):end_date + np.timedelta64(1, 'D'),
                                  :]

                    df_data = pd.concat([df_data, data_loaded], axis=0)
	
        #df_data = df_data.drop(df_data.columns[-2:], axis=1)

        labels = pd.read_csv(folder_path + '/' + 'labels.csv')
	
        df_data = df_data.iloc[:, :labels.shape[0]]

        description = dict()

        for idx, i in enumerate(labels['Name']):
            description[i] = str(labels['Description'].iloc[idx])

        df_data.columns = labels['Name']

        df_data = df_data.loc[:, labels['Name'].loc[labels['To include'] == 'Yes']]

        for i in df_data.columns[df_data.columns.str.endswith('SUM')]:
            df_data[i] = df_data[i].diff(60)

        df_data = df_data.dropna(axis=0)

        df_data['date'] = df_data.index
        df_data = df_data.reset_index(drop=True)
        df_data['date'] = pd.to_datetime(df_data['date'])
        df_data['date'] = df_data['date'].dt.strftime('%d/%m/%Y %H:%M:%S')

    if data_date.all() != None:

        start_date = np.datetime64(data_date.min())
        end_date = np.datetime64(data_date.max())

        for i in h5_list:
            print('Currently processing this h5 file: ' + i)

            data_loaded = load_h5(i)

            if data_loaded.index.max() < start_date:
                pass

            if data_loaded.index.max() > start_date:

                if data_loaded.index.max() < end_date:
                    data_loaded = data_loaded.loc[start_date - np.timedelta64(
                        1, 'D'):end_date + np.timedelta64(1, 'D'), :]

                    df_data = pd.concat([df_data, data_loaded], axis=0)

                if data_loaded.index.max() > end_date:
                    data_loaded = data_loaded.loc[start_date - np.timedelta64(
                        1, 'D'):end_date + np.timedelta64(1, 'D'), :]

                    df_data = pd.concat([df_data, data_loaded], axis=0)

        #df_data = df_data.drop(df_data.columns[-2:], axis=1)

        labels = pd.read_csv(folder_path + '/' + 'labels.csv')
	
        df_data = df_data.iloc[:, :labels.shape[0]]
	
        description = dict()

        for idx, i in enumerate(labels['Name']):
            description[i] = str(labels['Description'].iloc[idx])

        df_data.columns = labels['Name']

        df_data = df_data.loc[:, labels['Name'].loc[labels['To include'] == 'Yes']]

        for i in df_data.columns[df_data.columns.str.endswith('SUM')]:
            df_data[i] = df_data[i].diff(60)

        df_data = df_data.dropna(axis=0)

        df_data['date'] = df_data.index
        df_data = df_data.reset_index(drop=True)
        df_data['date'] = pd.to_datetime(df_data['date'])
        df_data['date'] = df_data['date'].dt.strftime('%d/%m/%Y %H:%M:%S')

    return df_data, labels, description


def remove_zeros(data):
    data_non_zero = dict()

    for i in data.keys():
        zeros = np.where(
            (data[i].drop(['date', 'time'], axis=1).T == 0).any() == True)
        nines = np.where(
            (data[i].drop(['date', 'time'], axis=1).T == -999.99).any() == True)
        to_remove = np.sort(np.unique(np.append(zeros, nines)))

        data_non_zero[i] = data[i].drop(
            to_remove, axis=0).reset_index(drop=True)

    return data_non_zero


def remove_duplicates(data):
    data_no_duplicates = dict()

    for i in data.keys():

        unique_dates = pd.DataFrame(data[i]['date']).drop_duplicates()
        duplicates_filtered = np.array([])

        for jdx, j in enumerate(unique_dates['date']):

            if len(np.where(data[i]['date'] == j)[0]) == 1:
                duplicates_filtered = np.append(
                    duplicates_filtered, int(np.where(data[i]['date'] == j)[0]))

            if len(np.where(data[i]['date'] == j)[0]) == 2:

                if np.sum(np.array(data[i].drop(['date', 'time'], axis=1).iloc[np.where(data[i]['date'] == j)[0],
                                   :].diff().dropna())) == 0:
                    duplicates_filtered = np.append(
                        duplicates_filtered, int(np.where(data[i]['date'] == j)[0][0]))

                if np.sum(np.array(data[i].drop(['date', 'time'], axis=1).iloc[np.where(data[i]['date'] == j)[0],
                                   :].diff().dropna())) != 0:
                    duplicates_filtered = np.append(
                        duplicates_filtered, int(np.where(data[i]['date'] == j)[0][0]))
                    duplicates_filtered = np.append(
                        duplicates_filtered, int(np.where(data[i]['date'] == j)[0][1]))

        data_no_duplicates[i] = data[i].iloc[duplicates_filtered, :]

    return data_no_duplicates


def func(x, A, B, T0):
    return T0 + T0 * (((A / B) * np.log(1 / (1 - x))) ** (1 / B))


def fit_riazi(data):
    riazi_results_dict = dict()
    data_riazi = dict()
    deviations_riazi = dict()

    for position_i, i in enumerate(data.keys()):
        print('Riazi modelling - currently processing: ', i)

        volumes = []

        for j in data[i].drop(['date', 'time'], axis=1).columns:
            volumes = np.append(volumes,
                                str(j).split('_')[1].strip('%'))

        try:

            popt1_riazi = []
            popt2_riazi = []
            popt3_riazi = []
            R2_riazi = []
            T99_riazi = []
            riazi_results_df = pd.DataFrame([])
            riazi_predictions_df = pd.DataFrame(
                np.append(np.arange(0, 1, 0.02), np.array(0.99)), columns=['volume'])
            riazi_deviations_df = pd.DataFrame([])

            for position_j in range(0, len(data[i])):

                y_data = np.array(data[i].drop(
                    ['date', 'time'], axis=1).iloc[position_j, :])
                x_data = np.array(volumes, dtype=float)

                if 100 in x_data:
                    x_data[np.where(x_data == 100)[0]] = 99.9

                x_data = x_data / 100

                popt, pcov = optimization.curve_fit(func,
                                                    x_data,
                                                    y_data,
                                                    p0=[1, 1, 200],
                                                    bounds=([0, 0, 100],
                                                            [100, 200, 1000]),
                                                    maxfev=10000)

                popt1_riazi = np.append(popt1_riazi, popt[0])
                popt2_riazi = np.append(popt2_riazi, popt[1])
                popt3_riazi = np.append(popt3_riazi, popt[2])
                T99_riazi = np.append(T99_riazi, func(
                    0.999, popt[0], popt[1], popt[2]))
                R2_riazi = np.append(R2_riazi, r2_score(
                    y_data, func(x_data, popt[0], popt[1], popt[2])))

                riazi_predictions_df[data[i]['date'].iloc[position_j]] = round(func(riazi_predictions_df['volume'],
                                                                                    popt[0],
                                                                                    popt[1],
                                                                                    popt[2]), 2)

                riazi_deviations_df[data[i]['date'].iloc[position_j]] = np.array(func(x_data,
                                                                                      popt[0],
                                                                                      popt[1],
                                                                                      popt[2])).flatten() - y_data

            riazi_results_df['date'] = data[i]['date']
            riazi_results_df['popt1_riazi'] = popt1_riazi
            riazi_results_df['popt2_riazi'] = popt2_riazi
            riazi_results_df['popt3_riazi'] = popt3_riazi
            riazi_results_df['T99_riazi'] = T99_riazi
            riazi_results_df['R2_riazi'] = R2_riazi

            riazi_results_dict[i] = riazi_results_df
            data_riazi[i] = riazi_predictions_df
            deviations_riazi[i] = riazi_deviations_df

            print('Riazi modelling - currently processing: ', i, ' - completed')

        except:
            pass

    return data_riazi, riazi_results_dict, deviations_riazi


def fit_arima(data):
    ARIMA_results = dict()
    ARIMA_results['errors_LIMS'] = dict()

    data_ARIMA = dict()

    sub_tags = []

    for position_i, i in enumerate(data.keys()):
        print('ARIMA modelling - currently processing: ', i)

        data_ARIMA_df = data[i]['date'].reset_index(drop=True)

        for position_j, j in enumerate(data[i].drop(['date', 'time'], axis=1).columns):
            X = data[i][j].reset_index(drop=True)

            model = SARIMAX(X, order=(3, 1, 3),
                            seasonal_order=(3, 1, 3, 5),
                            trend='ct', initialization='approximate_diffuse')

            results = model.fit(disp=False,
                                maxiter=50)

            ARIMA_results['errors_LIMS'][j] = round(
                np.mean(np.abs(X[6:] - results.predict()[6:])), 2)

            prediction = np.array(results.predict()).flatten()

            data_ARIMA_df = pd.concat([data_ARIMA_df,
                                       pd.DataFrame(prediction, columns=[j])], axis=1)

        data_ARIMA[i] = data_ARIMA_df

        print('ARIMA modelling - currently processing: ', i, ' - completed')

        sub_tags = np.sort(np.unique(np.append(sub_tags, np.array(
            data[i].drop(['date', 'time'], axis=1).columns))))

    results_df = pd.DataFrame(np.zeros((1, len(sub_tags))))

    results_df.columns = [i for i in np.array(sub_tags).astype(str)]

    for i in results_df.index:
        for j in results_df.columns:
            results_df.loc[0, j] = ARIMA_results['errors_LIMS'][j]

    results_df['product'] = results_df.index
    results_df = results_df[results_df.columns[::-1]].reset_index(drop=True)

    return data_ARIMA, results_df


def remove_outliers(data, ARIMA_data):
    data_processed = dict()
    outliers_list = dict()
    limits = dict()
    limits['upper'] = dict()
    limits['lower'] = dict()
    limits['Q1'] = dict()
    limits['Q3'] = dict()
    limits['IQR'] = dict()
    ARIMA_replace_outliers = dict()

    for i in data.keys():

        outliers = []
        data_key = data[i]

        for j in range(0, data_key.drop(['date', 'time'], axis=1).shape[1]):
            Q1 = data_key.drop(
                ['date', 'time'], axis=1).iloc[:, j].quantile(0.25)
            Q3 = data_key.drop(
                ['date', 'time'], axis=1).iloc[:, j].quantile(0.75)
            IQR = Q3 - Q1
            upper = Q3 + 1.5 * IQR
            lower = Q1 - 1.5 * IQR

            outliers_position_lower = np.where(data_key.drop(
                ['date', 'time'], axis=1).iloc[:, j] < lower)[0]
            outliers_position_upper = np.where(data_key.drop(
                ['date', 'time'], axis=1).iloc[:, j] > upper)[0]

            outliers_position = np.append(
                outliers_position_lower, outliers_position_upper).astype(int)
            good_data_position = [np.where((data_key.drop(['date', 'time'], axis=1).iloc[:, j] >= lower) &
                                           (data_key.drop(['date', 'time'], axis=1).iloc[:, j] <= upper))]

            outliers = np.sort(
                np.unique(np.append(outliers, outliers_position)))

            limits['upper'][data_key.drop(
                ['date', 'time'], axis=1).columns[j]] = upper
            limits['lower'][data_key.drop(
                ['date', 'time'], axis=1).columns[j]] = lower
            limits['Q1'][data_key.drop(
                ['date', 'time'], axis=1).columns[j]] = Q1
            limits['Q3'][data_key.drop(
                ['date', 'time'], axis=1).columns[j]] = Q3
            limits['IQR'][data_key.drop(
                ['date', 'time'], axis=1).columns[j]] = IQR

        data_no_outliers = data_key.reset_index(
            drop=True).drop(outliers.astype('int'), axis=0)

        ARIMA_all = pd.concat(
            [ARIMA_data[i].iloc[outliers.astype('int'), :].drop('date', axis=1).reset_index(drop=True),
             data_key[['date', 'time']].iloc[outliers.astype('int'), :].reset_index(drop=True)], axis=1)

        good_data_positions = []

        for idj, j in enumerate(ARIMA_all.drop(['date', 'time'], axis=1).columns):
            good_data_positions = np.append(good_data_positions, np.where(
                (ARIMA_all[j] > limits['lower'][j]) & (ARIMA_all[j] < limits['upper'][j])))

        good_data_positions = np.sort(np.unique(good_data_positions))

        ARIMA_replace_outliers_list = outliers[good_data_positions.astype(int)]
        outliers = np.delete(outliers, good_data_positions.astype(int))

        ARIMA_all = ARIMA_all.iloc[good_data_positions, :].reset_index(
            drop=True)

        data_no_outliers = pd.concat(
            [data_no_outliers, ARIMA_all.round(2)], axis=0)

        data_no_outliers['date'] = pd.to_datetime(data_no_outliers['date'], format='%d/%m/%Y')

        data_no_outliers = data_no_outliers.sort_values(
            by=['date']).reset_index(drop=True)
        data_no_outliers['date'] = pd.to_datetime(data_no_outliers['date'].dt.date).dt.strftime('%d/%m/%Y')

        data_processed[i] = data_no_outliers
        outliers_list[i] = outliers
        ARIMA_replace_outliers[i] = ARIMA_replace_outliers_list

    return data_processed, outliers_list, ARIMA_replace_outliers, limits


def prioritisation(data):
    print('Current priority key: Amount of data available')
    keys = data.keys()

    keys_list = []
    data_length = []

    for i in keys:
        keys_list.append(i)
        data_length.append(data[i].shape[0])

    print('Key chosen via prioritisation:', keys_list[np.argmax(data_length)],
          'Number of data points for this property: ',
          np.max(data_length))

    return keys_list[np.argmax(data_length)]


def correlations_LIMS(data):
    keys = data.keys()

    keys_list = []
    data_length = []

    for i in keys:
        keys_list.append(i)
        data_length.append(data[i].shape[0])

    keys_sorted = np.array(keys_list)[np.argsort(data_length)[::-1]]

    data_combined = data[keys_sorted[0]].drop('time', axis=1)

    for i in keys_sorted[1:]:
        data_combined = pd.merge(
            data_combined, data[i].drop('time', axis=1), on='date')

    corr = data_combined.drop('date', axis=1).corr(method='pearson')
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    if len(keys) < 2:
        knn_results_df = pd.DataFrame(
            np.array(['Only one product provided: Required at least 2']))
        models_knn = None

    if len(keys) >= 2:

        results_array = []
        models_knn = dict()

        for i in keys_sorted:

            for j in np.delete(keys_sorted, np.where(keys_sorted == i)):

                X = data[i]
                y = data[j]

                data_combined = pd.merge(
                    X.drop('time', axis=1), y.drop('time', axis=1), on='date')

                X = data_combined.loc[:,
                    data_combined.columns.str.startswith(i)]
                y = data_combined.loc[:,
                    data_combined.columns.str.startswith(j)]

                knn = KNeighborsRegressor()

                if X.shape[0] < 25 or y.shape[0] < 25:
                    max_neighbours = 2

                if X.shape[0] > 25 or y.shape[0] > 25:
                    max_neighbours = 20

                params = {'n_neighbors': np.arange(0, max_neighbours)}

                models_knn['%s-%s' %
                           (str(i), str(j))] = GridSearchCV(knn, params, cv=3)

                models_knn['%s-%s' % (str(i), str(j))].fit(X, y)

                results_array = np.append(results_array, ['%s - %s' % (str(i), str(j)),
                                                          int(models_knn['%s-%s' % (str(i), str(j))].best_params_[
                                                                  'n_neighbors']), round(
                        mean_absolute_error(y, models_knn['%s-%s' % (str(i), str(j))].predict(X)), 3)])

        knn_results_df = pd.DataFrame(
            results_array.reshape(int(len(results_array) / 3), 3))
        knn_results_df.columns = ['Base label - predicted label',
                                  'Number of neighbours', 'Mean absolute error of prediction']

    return corr, knn_results_df, models_knn


def remove_short_term_outliers(data, tags, dw=60, resample=1):
    tol = 10 / np.sqrt(dw)
    data_date = data['date']
    data = np.array(data.drop('date', axis=1))

    data_processed = pd.DataFrame([])
    data_smooth = pd.DataFrame([])
    data_smooth_corrected = pd.DataFrame([])
    data_no_outliers = pd.DataFrame([])
    outliers_list = pd.DataFrame([])
    results = pd.DataFrame([])

    print('Removing short-term outliers. Number of sensors to process: ' + str(len(tags)) + ' - Start')

    for position_i, i in enumerate(tags):
        print('Removing short-term outliers. Currently processing: ' + i)

        t = data[:, position_i]

        w = round(dw / resample)  # window width
        n = int(np.floor(len(t) / w))
        N = int(w * n)
        t = t[0:N]

        T = np.reshape(t, (n, w))

        mea = np.median(T, axis=1)

        t2 = np.kron(mea, np.ones(w))

        mea = np.array([mea] * T.shape[1])
        mea = mea.transpose()

        stdest = np.mean(abs(T - mea))

        ind_s = 0
        ind_e = 1
        mea2 = []
        nrp = 1

        while ind_e <= len(mea):
            piece = mea[ind_s:ind_e]
            appr = (np.max(piece) + np.min(piece)) / 2
            if np.linalg.norm(piece - appr, np.inf) < tol * stdest and ind_e < len(mea):
                ind_e = ind_e + 1
            else:
                if ind_e - ind_s >= 1:
                    nrp = nrp + 1
                mea2 = np.append(mea2, np.ones(
                    ind_e - ind_s) * np.median(piece))
                ind_s = ind_e
                ind_e = ind_e + 1

        t3 = np.kron(mea2, np.ones(w))

        # filter outliers
        t4 = t.copy()
        std = mad(t - t3)
        upper = t3 + 3 * std
        lower = t3 - 3 * std
        t4[(t < lower) ^ (t > upper)] = t3[(t < lower) ^ (t > upper)]
        outliers = data_date.iloc[:len((t < lower) ^ (
                t > upper))].loc[(t < lower) ^ (t > upper)]
        noise = np.mean(np.abs(t - t3))
        STN = np.mean(t3) / np.std(t3)

        data_processed['%s' % (i)] = t
        data_smooth['%s' % (i)] = t2
        data_smooth_corrected['%s' % (i)] = t3
        data_no_outliers['%s' % (i)] = t4
        outliers_list = pd.concat([outliers_list, pd.DataFrame(
            np.array(outliers), columns=['%s' % (i)])], axis=1)
        results['%s' % (i)] = [noise,
                               len(outliers) / len(t3) * 100,
                               np.mean(t3),
                               np.std(t3),
                               STN]

    results.index = ['Noise level',
                     'Number of outliers (% of total amount)',
                     'Mean',
                     'Standard deviation',
                     'Signal-to-noise ratio']

    data_processed['date'] = data_date[:data_processed.shape[0]]
    data_no_outliers['date'] = data_date[:data_no_outliers.shape[0]]
    data_smooth_corrected['date'] = data_date[:data_smooth_corrected.shape[0]]

    print('Removing short-term outliers. Number of sensors to process: ' + str(len(tags)) + ' - Completed')

    return data_processed, data_no_outliers, data_smooth_corrected, outliers_list, results


def fit_PCA(data_scaled, factor):
    pca = PCA(factor)
    X_pca = pca.fit_transform(data_scaled)

    return pca, X_pca


def remove_long_term_outliers(data):
    PCA_results = dict()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(data.drop('date', axis=1).values)

    pca, X_pca = fit_PCA(X_std, 0.9)

    proj = pca.inverse_transform(X_pca)

    components = 2

    scaler = StandardScaler()
    X_std = scaler.fit_transform(data.drop('date', axis=1))

    # pca = PCA(n_components=components)

    # X_pca = pca.fit_transform(X_std)

    p = 0.95
    # df = components

    # chi_square = chi2.ppf(p, df)
    # print('\nChi-square: ', chi_square)
    # X_pca = np.transpose(X_pca)
    # cov_matrix = np.cov(X_pca)
    # print('\nCovariance matrix: \n', cov_matrix)
    # mean = np.mean(X_pca, axis=1)
    # print('\nMean: \n', mean)
    # inv_cov_matrix = np.linalg.inv(cov_matrix)
    # print('\nInverse covariance matrix: \n', inv_cov_matrix)
    # eigenvalues = np.linalg.eig(cov_matrix)[0]
    # print('\nEigenvalues: \n', eigenvalues)
    # a = np.sqrt(eigenvalues[0]) * np.sqrt(chi_square)
    # print('\na: \n', a)
    # b = np.sqrt(eigenvalues[1]) * np.sqrt(chi_square)
    # print('\nb: \n', b)
    # rad = np.arctan2(cov_matrix[0, 1], eigenvalues[0] - cov_matrix[0, 0])
    # angle = rad * 180 / 3.14
    # print('\nangle: \n', rad, '/', angle)
    # rotation_matrix = np.array([[np.sin(rad), -np.cos(rad)],[np.cos(rad), np.sin(rad)]])
    # print('\nRotation matrix: \n', rotation_matrix)
    # Q = []

    # for i in range(0, 21):
    # Q.append(np.pi * i / 10)

    # X_elipse = (rotation_matrix[0, 0] * a * np.cos(Q) + rotation_matrix[0, 1] * b * np.sin(Q)) + mean[0]
    # y_elipse = (rotation_matrix[1, 0] * a * np.cos(Q) + rotation_matrix[1, 1] * b * np.sin(Q)) + mean[1]

    # X_bar = X_pca[0, :] - mean[0]
    # y_bar = X_pca[1, :] - mean[1]

    # decision = np.array(((inv_cov_matrix[0, 0] * X_bar + inv_cov_matrix[0, 1] * y_bar) * X_bar) + (
    # (inv_cov_matrix[1, 0] * X_bar + inv_cov_matrix[1, 1] * y_bar) * y_bar))

    pca, X_pca = fit_PCA(X_std, 2)

    T2 = np.zeros(len(X_pca))

    for i in range(0, X_pca.shape[1]):
        T2 = T2 + (X_pca[:, i] / pca.explained_variance_[i]) ** 2

    # deviation = np.std(T2) * 2
    deviation = f.ppf(p, X_pca.shape[1], X_pca.shape[0]) * (X_pca.shape[0] ** 2 - 1) * X_pca.shape[1] / (
                X_pca.shape[0] * (X_pca.shape[0] - X_pca.shape[1]))

    long_term_outliers_position = np.arange(0, len(T2))[T2 > deviation]
    long_term_outliers_dates = data['date'][long_term_outliers_position]

    data_time = data['date']
    tags = data.columns
    data = np.array(data.drop('date', axis=1))
    medians = np.median(data, axis=0)

    for i in long_term_outliers_position:
        data[i, :] = medians

    data_treated = pd.concat([pd.DataFrame(data), pd.DataFrame(data_time)], axis=1)
    data_treated.columns = tags

    pca, X_pca = fit_PCA(X_std, X_std.shape[1])

    PCA_summary = pd.DataFrame(
        np.arange(1, int(X_std.shape[1] + 1)), columns=['Principal component'])
    PCA_summary['Variance'] = pd.DataFrame(pca.explained_variance_ratio_ * 100)
    PCA_summary['Cumulative variance'] = pd.DataFrame(
        np.cumsum(pca.explained_variance_ratio_) * 100)

    PCA_results['summary'] = PCA_summary
    PCA_results['X_pca'] = pd.DataFrame(X_pca)
    PCA_results['T2'] = T2

    return data_treated, long_term_outliers_dates, PCA_results


def feature_pre_selection(data, labels):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(data.drop('date', axis=1).values)
    tags = data.drop('date', axis=1).columns

    pca, X_pca = fit_PCA(X_std, X_std.shape[1])

    n_pcs = pca.n_components_

    print('Feature pre-selection. Number of sensors: ' + str(X_std.shape[1]) + ' - Start')

    tags_selected = np.sort(
        np.unique([np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]))

    print('Feature pre-selection. Number of sensors after reduction: ' + str(len(tags_selected)))

    results_df = pd.DataFrame(np.abs(pca.components_))

    index = []

    for i in range(1, len(results_df.index) + 1):
        index.append('PC%i' % i)

    results_df.index = index
    results_df.columns = tags

    info = []

    for i in range(0, len(results_df.index)):
        info.append('The maximum variance for principal component %i is represented by %s in the feature space' % (
            int(i + 1), str(data.drop('date', axis=1).columns[np.argmax(np.abs(pca.components_[i]))])))

    info = pd.DataFrame(info, columns=['Info'])
    info.index = index

    results_df = pd.concat([results_df, info], axis=1)

    tags_pre_selected = np.append('date',
                                  tags[tags_selected])

    data_pre_selected = data[tags_pre_selected]

    tags_accepted = data_pre_selected.drop('date', axis=1).columns
    tags_rejected = []

    for i in data.drop('date', axis=1).columns:

        if i not in tags_accepted:
            tags_rejected = np.append(tags_rejected, i)

    for i in labels['Name'].loc[labels['Force'] == 'Yes']:
        data_pre_selected[i] = data[i]

    print('Feature pre-selection. Final number of sensors: ' + str(len(tags_accepted)) + ' - Completed')

    return data_pre_selected, tags_accepted, tags_rejected, results_df


def correlations_sensors(data_sensors):
    corr_sensors_df = data_sensors.corr()

    return corr_sensors_df


def data_split(data_sensors_means, data_LIMS, train_test_size, test_valid_size, random_state):
    X = data_sensors_means
    y = data_LIMS

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    print('Size of input: samples - {}, sensors - {}'.format(X_std.shape[0], X_std.shape[1]))
    print('Size of output: samples - {}, products - {}'.format(y.shape[0], y.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(np.array(X_std),
                                                        np.array(y),
                                                        test_size=train_test_size,
                                                        random_state=random_state)

    X_test, X_valid, y_test, y_valid = train_test_split(X_test,
                                                        y_test,
                                                        test_size=test_valid_size,
                                                        random_state=random_state)

    print('Size of training samples: {}'.format(X_train.shape[0]))
    print('Size of testing samples: {}'.format(X_test.shape[0]))
    print('Size of validation samples: {}'.format(X_valid.shape[0]))

    return X, y, X_train, X_test, X_valid, y_train, y_test, y_valid, X_std, data_sensors_means.columns, data_LIMS.columns


def model_builder_adam(dim, number_of_products, hidden_layers, nodes, activation, optimiser, loss, kernel, bias):
    model = Sequential()
    for i in range(1, hidden_layers + 1):
        model.add(Dense(nodes, activation=activation, input_dim=(
            dim), kernel_initializer=kernel, bias_initializer=bias))
    model.add(Dense(number_of_products, activation="linear"))
    model.compile(optimizer=optimiser, loss=loss)
    model.summary()
    return model


def feature_importance(X_train, X_test, X_valid, model, labels):
    print('Calculating feature importance')

    X = np.arange(0, 1.01, 0.01)
    y = np.concatenate((X_train, X_test, X_valid), axis=0)

    y_pred = np.zeros((len(X), y.shape[1]))

    for i in range(0, y.shape[1]):
        y_est = y[0, :].reshape(1, y.shape[1])
        y_est[0, :] = 0

        for j, value in enumerate(X):
            y_est[0, i] = value
            y_pred[j, i] = np.sum(np.abs(model.predict(y_est)[0]))

    # use y_pred to plot everything as a map

    y_pred_abs = np.abs(y_pred / y_pred[0, 0] * 100 - 100)

    slopes = []

    for i in range(0, y_pred_abs.shape[1]):
        slopes.append(linregress(X, y_pred_abs[:, i])[0])

    # use slopes for absolute

    columns_sorted = np.array(labels)[np.array(
        np.argsort(slopes)[::-1], dtype=int)]
    y_pred_norm_sorted = np.sort(slopes)[::-1]

    # use the two above for sorted

    return slopes, np.array(labels), y_pred_norm_sorted, columns_sorted


def time_alignment(priority_key, data_LIMS, data_sensors, description):
    print('Aligning LIMS with sensors in time: Start')

    if priority_key != None:
        data_LIMS = data_LIMS[priority_key]
        data_LIMS['date'] = pd.to_datetime(data_LIMS['date'], format='%d/%m/%Y')
        data_LIMS['time'] = pd.to_datetime(data_LIMS['time'], format='%H:%M:%S')

    if np.array(data_LIMS).all() == None:
        data_LIMS = load_LIMS_data_for_softsensors()

    if np.array(data_sensors).all() == None:
        data_sensors = load_sensors_data_for_softsensors()

    if np.array(description).all() == None:

        try:
            description = pd.read_csv(folder_path + '/labels.csv')[['Name', 'Description']]
            description = description.iloc[
                          np.where(description['Name'] == data_sensors.drop('date', axis=1).columns)[0], :]

        except:
            description = [None]

    if priority_key != None:
        description = description

    data_sensors['date'] = pd.to_datetime(data_sensors['date'], format='%d/%m/%Y %H:%M:%S')

    date_combined = []
    data_mean = pd.DataFrame([])

    for i in range(data_LIMS.shape[0]):
        date_combined.append(str(pd.to_datetime(data_LIMS['date'].iloc[i]).strftime('%d/%m/%Y')) + ' ' + str(
            pd.to_datetime(data_LIMS['time'].iloc[i]).strftime('%H:%M:%S')))

    data_LIMS['date_combined'] = pd.to_datetime(date_combined, format='%d/%m/%Y %H:%M:%S')

    for i in data_LIMS['date_combined']:
        start = i - DateOffset(hours=1)
        end = i

        mean_array = np.mean(np.array(data_sensors.drop('date', axis=1).iloc[
                                      int(np.where(data_sensors['date'] == start)[0]): int(
                                          np.where(data_sensors['date'] == end)[0]), :]), axis=0)

        data_mean = pd.concat([data_mean, pd.DataFrame(mean_array.reshape(1, len(mean_array)))], axis=0)

    data_mean.columns = data_sensors.drop('date', axis=1).columns

    data_mean = data_mean.reset_index(drop=True)

    print('Aligning LIMS with sensors in time: Completed')

    return data_mean, data_LIMS.drop(['date_combined', 'time', 'date'], axis=1), data_LIMS['date_combined'], description


def correlations_LIMS_sensors(data_LIMS, data_sensors):
    data = pd.concat([data_LIMS, data_sensors], axis=1)

    return data.corr()


def learning_rate(X_train, X_test, X_valid, y_train, y_test, y_valid, n_hidden_layers, n_nodes, optimiser, loss,
                  activation_function, kernel_initializer, bias_initializer):
    # n_hidden_layers = 2
    # n_nodes = int(X_train.shape[1])
    # optimiser = Adam(lr = 0.01)
    # loss = 'huber_loss'
    # activation_function = 'relu'
    # kernel_initializer = 'random_uniform'
    # bias_initializer = 'zeros'

    model = model_builder_adam(X_train.shape[1], n_hidden_layers, n_nodes,
                               activation_function, optimiser, loss, kernel_initializer, bias_initializer)

    history = keras.callbacks.History()

    start = 0.00001
    denominator = 40

    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: start * 10 ** (epoch / denominator), verbose=1)

    history = model.fit(X_train, y_train, epochs=200, verbose=2, batch_size=50, validation_data=(X_test, y_test),
                        shuffle=True, callbacks=[history, lr_schedule])

    lr = []

    for i in range(0, 200):
        lr.append(start * 10 ** (i / denominator))

    print(np.min(history.history['loss']), np.argmin(
        history.history['loss']), lr[np.argmin(np.diff(history.history['loss']))])
    print(np.min(history.history['val_loss']), np.argmin(
        history.history['val_loss']), lr[np.argmin(np.diff(history.history['loss']))])


def quick_search(X_train, X_test, X_valid, y_train, y_test, y_valid, labels, folder_models_soft_sensors):
    print('Quick search: Started')
    n_hidden_layers = 2

    if X_train.shape[1] < 10:
        n_nodes = 5

    if X_train.shape[1] >= 10:
        n_nodes = int(X_train.shape[1]) // 10 * 10

    print('Number of hidden layers: {}, number of nodes in each hidden layer: {}'.format(n_hidden_layers, n_nodes))

    optimiser = Adam(lr=0.01)
    loss = 'huber_loss'
    activation_function = 'relu'
    kernel_initializer = 'random_uniform'
    bias_initializer = 'zeros'

    metrics = dict()
    feature_importance_dict = dict()

    start = time.perf_counter()

    metrics['configuration_layers'] = n_hidden_layers
    metrics['configuration_nodes'] = n_nodes
    metrics['configuration_optimiser'] = optimiser
    metrics['configuration_loss'] = loss
    metrics['configuration_activation'] = activation_function

    epochs = 100

    metrics['history'] = optimise_model(X_train, X_test, X_valid, y_train, y_test, y_valid, n_hidden_layers, n_nodes,
                                        optimiser, loss, kernel_initializer, bias_initializer, activation_function,
                                        'quick_search_model', folder_models_soft_sensors, epochs, 0)

    end = time.perf_counter()
    metrics['time_elapsed'] = int(end - start)

    model, metrics['training_error'], metrics['testing_error'], metrics['validation_error'], metrics['std'], metrics[
        'max'], metrics['deviations'] = validate_model(
        X_train, X_test, X_valid, y_train, y_test, y_valid, folder_path, optimiser, loss, 'quick_search_model',
        folder_models_soft_sensors, 'Quick search model')

    feature_importance_dict['feature_imp'], feature_importance_dict['labels'], feature_importance_dict[
        'feature_imp_sorted'], feature_importance_dict['feature_imp_sorted_labels'] = feature_importance(X_train,
                                                                                                         X_test,
                                                                                                         X_valid, model,
                                                                                                         labels)

    if metrics['time_elapsed'] < 60:
        print(f"It took {metrics['time_elapsed']} seconds to train the network")

    if metrics['time_elapsed'] > 60 and metrics['time_elapsed'] < 3600:
        minutes = math.floor(metrics['time_elapsed'] / 60)
        seconds = metrics['time_elapsed'] - minutes * 60

        print(f"It took {minutes} minutes and {seconds} seconds to train the network")

    if metrics['time_elapsed'] > 3600:
        hours = math.floor(metrics['time_elapsed'] / 3600)
        minutes = math.floor((metrics['time_elapsed'] - hours * 3600) / 60)
        seconds = metrics['time_elapsed'] - minutes * 60 - hours * 3600

        print(f"It took {hours} hours {minutes} minutes {seconds} seconds to train the network")

    print('Quick search: Completed')

    return metrics, feature_importance_dict


def full_search(X_train, X_test, X_valid, y_train, y_test, y_valid, labels, folder_models_soft_sensors):
    print('Full search: Started')
    n_hidden_layers = [1, 2, 3]
    n_nodes = np.arange(5, int(X_train.shape[1]) // 10 * 10 + 2, 5)

    if X_train.shape[1] < 10:
        n_nodes = np.arange(5, 16, 5)

    print('Number of hidden layers: {}, number of nodes in each hidden layer: {}'.format(n_hidden_layers, n_nodes))

    optimiser = Adam(lr=0.001)
    loss = 'huber_loss'
    activation_function = 'relu'
    kernel_initializer = 'random_uniform'
    bias_initializer = 'zeros'

    configuration = dict()
    configuration['Hidden layers'] = n_hidden_layers
    configuration['Number of nodes'] = n_nodes
    configuration['Optimizer'] = optimiser
    configuration['Loss function'] = loss
    configuration['Activation function'] = activation_function
    configuration['Kernel initializer'] = kernel_initializer
    configuration['Bias initializer'] = bias_initializer

    performance_dict = dict()
    importance_dict = dict()

    epochs = 10000

    for i in n_hidden_layers:
        for j in n_nodes:

            print('Currently training: {} hidden layers x {} nodes'.format(i, j))

            performance_dict['{}_{}'.format(int(i), int(j))] = dict()
            importance_dict['{}_{}'.format(int(i), int(j))] = dict()

            # learning_rate(X_train, X_test, X_valid, y_train, y_test, y_valid, i, j, optimiser, loss, activation_function, kernel_initializer, bias_initializer)

            start = time.perf_counter()

            performance_dict['{}_{}'.format(int(i), int(j))]['history'] = optimise_model(X_train, X_test, X_valid,
                                                                                         y_train, y_test, y_valid, i, j,
                                                                                         optimiser,
                                                                                         loss, kernel_initializer,
                                                                                         bias_initializer,
                                                                                         activation_function,
                                                                                         'layers_{}_nodes_{}'.format(
                                                                                             int(i), int(j)),
                                                                                         folder_models_soft_sensors,
                                                                                         epochs, 0)

            end = time.perf_counter()
            performance_dict['{}_{}'.format(int(i), int(j))]['time_elapsed'] = int(end - start)

            model, performance_dict['{}_{}'.format(int(i), int(j))]['training_error'], \
            performance_dict['{}_{}'.format(int(i), int(j))]['testing_error'], \
            performance_dict['{}_{}'.format(int(i), int(j))]['validation_error'], \
            performance_dict['{}_{}'.format(int(i), int(j))]['std'], performance_dict['{}_{}'.format(int(i), int(
                j))]['max'], performance_dict['{}_{}'.format(int(i), int(j))]['deviations'] = validate_model(X_train,
                                                                                                             X_test,
                                                                                                             X_valid,
                                                                                                             y_train,
                                                                                                             y_test,
                                                                                                             y_valid,
                                                                                                             folder_path,
                                                                                                             optimiser,
                                                                                                             loss,
                                                                                                             'layers_{}_nodes_{}'.format(
                                                                                                                 int(i),
                                                                                                                 int(
                                                                                                                     j)),
                                                                                                             folder_models_soft_sensors,
                                                                                                             'Model architecture: hidden layers - {}, nodes - {}'.format(
                                                                                                                 int(i),
                                                                                                                 int(
                                                                                                                     j)))
            importance_dict['{}_{}'.format(int(i), int(j))]['feature_imp'], \
            importance_dict['{}_{}'.format(int(i), int(j))]['labels'], importance_dict['{}_{}'.format(int(i), int(
                j))]['feature_imp_sorted'], importance_dict['{}_{}'.format(int(i), int(j))][
                'feature_imp_sorted_labels'] = feature_importance(X_train, X_test, X_valid, model, labels)

            if performance_dict['{}_{}'.format(int(i), int(j))]['time_elapsed'] < 60:
                print(
                    f"It took {performance_dict['{}_{}'.format(int(i), int(j))]['time_elapsed']} seconds to train the network")

            if performance_dict['{}_{}'.format(int(i), int(j))]['time_elapsed'] > 60 and \
                    performance_dict['{}_{}'.format(int(i), int(j))]['time_elapsed'] < 3600:
                minutes = math.floor(performance_dict['{}_{}'.format(int(i), int(j))]['time_elapsed'] / 60)
                seconds = performance_dict['{}_{}'.format(int(i), int(j))]['time_elapsed'] - minutes * 60

                print(f"It took {minutes} minutes and {seconds} seconds to train the network")

            if performance_dict['{}_{}'.format(int(i), int(j))]['time_elapsed'] > 3600:
                hours = math.floor(performance_dict['{}_{}'.format(int(i), int(j))]['time_elapsed'] / 3600)
                minutes = math.floor(
                    (performance_dict['{}_{}'.format(int(i), int(j))]['time_elapsed'] - hours * 3600) / 60)
                seconds = performance_dict['{}_{}'.format(int(i), int(j))]['time_elapsed'] - minutes * 60 - hours * 3600

                print(f"It took {hours} hours {minutes} minutes {seconds} seconds to train the network")

    print('Full search: Completed')

    return configuration, performance_dict, importance_dict


def validate_model(X_train, X_test, X_valid, y_train, y_test, y_valid, folder_path, optimiser, loss, model_name,
                   folder_models_soft_sensors, model_type):
    model = keras.models.load_model(
        folder_models_soft_sensors + '{}.h5'.format(model_name), compile=False)
    model.compile(optimizer=optimiser, loss=loss)

    training_error = np.mean(np.abs(model.predict(X_train) - y_train), axis=0)
    testing_error = np.mean(np.abs(model.predict(X_test) - y_test), axis=0)
    validation_error = np.mean(
        np.abs(model.predict(X_valid) - y_valid), axis=0)

    std = dict()

    std['training'] = np.std(model.predict(X_train) - y_train)
    std['testing'] = np.std(model.predict(X_test) - y_test)
    std['validation'] = np.std(model.predict(X_valid) - y_valid)

    max_value = dict()

    max_value['training'] = np.max(np.abs(model.predict(X_train) - y_train))
    max_value['testing'] = np.max(np.abs(model.predict(X_test) - y_test))
    max_value['validation'] = np.max(np.abs(model.predict(X_valid) - y_valid))

    deviations = dict()

    deviations['training'] = model.predict(X_train) - y_train
    deviations['testing'] = model.predict(X_test) - y_test
    deviations['validation'] = model.predict(X_valid) - y_valid

    print('Performance: ')
    print('Training: {}'.format(training_error))
    print('Testing: {}'.format(testing_error))
    print('Validation: {}'.format(validation_error))

    return model, training_error, testing_error, validation_error, std, max_value, deviations


def optimise_model(X_train, X_test, X_valid, y_train, y_test, y_valid, n_hidden_layers, n_nodes, optimiser, loss,
                   kernel_initializer, bias_initializer, activation_function, model_name, folder_models_soft_sensors,
                   epochs, verbose):
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=folder_models_soft_sensors + '{}.h5'.format(model_name),
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    model = model_builder_adam(X_train.shape[1], y_train.shape[1], n_hidden_layers, n_nodes,
                               activation_function, optimiser, loss, kernel_initializer, bias_initializer)

    history = keras.callbacks.History()
    history = model.fit(X_train, y_train, epochs=epochs, verbose=verbose, batch_size=32,
                        validation_data=(X_test, y_test),
                        shuffle=True, callbacks=[history, model_checkpoint_callback])

    pd.DataFrame(history.history).to_csv(folder_models_soft_sensors + '{}.csv'.format(model_name))

    return pd.DataFrame(history.history)


def ARIMA_thresholds(data_LIMS, results_holder_dict):
    if np.array(results_holder_dict).all() == None:
        return np.array([None])

    if np.array(results_holder_dict).all() != None:
        arima = []

        for i in list(data_LIMS.columns):
            arima.append(int(np.where(results_holder_dict['ARIMA_threshold']['product'] == i)[0]))

        arima = np.array(results_holder_dict['ARIMA_threshold']['Value'][arima])

        return arima


def LIMS_analyse(folder_results_excel, value_riazi, folder_data):
    data_holder_dict = dict()
    results_holder_dict = dict()

    writer_LIMS = pd.ExcelWriter(folder_results_excel + '/output results - LIMS.xlsx', engine='xlsxwriter')

    print('LIMS analysis: Start')

    # ----- LOAD THE LIMS DATA -----
    print('LIMS data loading: start')
    data_date, data_LIMS, data_LIMS_separated = load_LIMS_data()
    print('LIMS data loading: completed')

    data_holder_dict['data_LIMS_separated'] = data_LIMS_separated
    data_holder_dict['data_date'] = data_date

    # ----- WRITE ORIGINAL DATA TO OUTPUT -----

    output_df = pd.DataFrame([])

    for i in data_LIMS_separated.keys():
        output_df = pd.concat([output_df, data_LIMS_separated[i]], axis=1)

    output_df.to_excel(writer_LIMS, sheet_name='Data - Original')

    # ----- REMOVE ZEROS AND NINES FROM LIMS DATA -----
    print('Removing zeros in LIMS: start')
    data_LIMS_separated_nozeros = remove_zeros(data_LIMS_separated)
    print('Removing zeros in LIMS: completed')

    data_holder_dict['data_LIMS_separated_nozeros'] = data_LIMS_separated_nozeros

    # ----- WRITE REMOVE ZEROS DATA TO OUTPUT -----

    output_df = pd.DataFrame([])

    for i in data_LIMS_separated_nozeros.keys():
        output_df = pd.concat(
            [output_df, data_LIMS_separated_nozeros[i]], axis=1)

    output_df.to_excel(writer_LIMS, sheet_name='Data - Zeros and -999 removed')

    # ----- REMOVE DUPLICATES -----

    print('Removing duplicates in LIMS: start')
    data_LIMS_separated_nozeros_noduplicates = remove_duplicates(
        data_LIMS_separated_nozeros)
    print('Removing duplicates in LIMS: completed')

    data_holder_dict['data_LIMS_separated_nozeros_noduplicates'] = data_LIMS_separated_nozeros_noduplicates

    # ----- REMOVE DUPLICATES TO OUTPUT -----

    output_df = pd.DataFrame([])

    for i in data_LIMS_separated_nozeros_noduplicates.keys():
        output_df = pd.concat(
            [output_df, data_LIMS_separated_nozeros_noduplicates[i]], axis=1)

    output_df.to_excel(writer_LIMS, sheet_name='Data - Duplicates removed')

    # ----- RIAZI MODELLING -----

    if value_riazi == True:

        print('Fitting Riazi model in LIMS: start')
        data_riazi, riazi_results_dict, deviations_riazi = fit_riazi(
            data_LIMS_separated_nozeros_noduplicates)
        print('Fitting Riazi model in LIMS: completed')

        output_df = pd.DataFrame([])

        for i in riazi_results_dict.keys():
            output_df = pd.concat(
                [output_df, riazi_results_dict[i].round(3)], axis=1)

        output_df.to_excel(writer_LIMS, sheet_name='Data - Riazi model')

        data_holder_dict['data_riazi'] = data_riazi
        results_holder_dict['riazi_results_dict'] = riazi_results_dict
        results_holder_dict['deviations_riazi'] = deviations_riazi

    if value_riazi == False:
        print('Fitting Riazi model in LIMS: disabled')
        data_holder_dict['data_riazi'] = 0

        # ----- ARIMA MODELLING -----

    print('Fitting ARIMA model in LIMS: start')
    data_ARIMA, results_ARIMA = fit_arima(data_LIMS_separated_nozeros_noduplicates)
    print('Fitting ARIMA model in LIMS: completed')

    # ----- ARIMA MODELLING TO OUTPUT-----

    output_df = pd.DataFrame([])

    for i in data_ARIMA.keys():
        output_df = pd.concat([output_df, data_ARIMA[i].round(3)], axis=1)

    output_df.to_excel(writer_LIMS, sheet_name='Data - ARIMA model')

    output_df = pd.DataFrame([])

    for i in results_ARIMA.keys():
        output_df = pd.concat([output_df, results_ARIMA[i]], axis=1)

    output_df = output_df.transpose()
    output_df = output_df.iloc[1:, :]
    output_df.columns = ['Value']
    output_df['product'] = output_df.index
    output_df = output_df.reset_index(drop=True)

    output_df.to_excel(writer_LIMS, sheet_name='Data - ARIMA model results', index=False)

    data_holder_dict['data_ARIMA'] = data_ARIMA
    results_holder_dict['results_ARIMA'] = results_ARIMA
    results_holder_dict['ARIMA_threshold'] = output_df

    # ----- REMOVE OUTLIERS FROM LIMS -----

    print('Removing outliers in LIMS: start')
    data_no_outliers, outliers, ARIMA_replace_outliers, limits = remove_outliers(
        data_LIMS_separated_nozeros_noduplicates, data_holder_dict['data_ARIMA'])
    print('Removing outliers in LIMS: completed')

    data_holder_dict['data_no_outliers'] = data_no_outliers
    results_holder_dict['outliers'] = outliers
    results_holder_dict['limits'] = limits

    # ----- WRITE REMOVE OUTLIERS FROM LIMS DATA TO OUTPUT -----

    output_df = pd.DataFrame([])

    for i in data_no_outliers.keys():
        output_df = pd.concat([output_df, data_no_outliers[i]], axis=1)

    output_df.to_excel(writer_LIMS, sheet_name='Data - No outliers')

    output_df = pd.DataFrame([])

    for i in data_no_outliers.keys():
        output_df = pd.concat([output_df,
                               pd.DataFrame(outliers[i], columns=['index']),
                               pd.DataFrame(
                                   data_LIMS_separated_nozeros_noduplicates[i].iloc[outliers[i], :]).reset_index(
                                   drop=True)], axis=1)

    output_df.to_excel(writer_LIMS, sheet_name='Data - Outliers', index=False)

    output_df = []

    for i in limits['upper'].keys():
        output_df.append([i, round(float(limits['lower'][i]), 2),
                          round(float(limits['upper'][i]), 2)])

    output_df = pd.DataFrame(np.array(output_df))

    output_df.columns = ['Product', 'Lower limit', 'Upper limit']

    output_df.to_excel(writer_LIMS, sheet_name='LIMS limits', index=False)

    # ----- CORRELATIONS IN LIMS -----

    print('Calculating correlations in LIMS: start')
    corr_map, knn_results_df, models_knn = correlations_LIMS(data_no_outliers)
    print('Calculating correlations in LIMS: completed')

    results_holder_dict['kNN models'] = models_knn
    results_holder_dict['corr_map'] = corr_map

    # ----- CORRELATIONS IN LIMS TO OUTPUT -----

    knn_results_df.round(3).to_excel(writer_LIMS, sheet_name='kNN regression results')
    corr_map.index = corr_map.columns
    corr_map.round(3).to_excel(writer_LIMS, sheet_name='LIMS - Correlations map')

    writer_LIMS.save()

    # ----- PRIORITISATION  -----

    print('Finding the best key - prioritisation: start')
    priority_key = prioritisation(data_no_outliers)
    print('Finding the best key - prioritisation: completed - ' + priority_key)

    results_holder_dict['priority_key'] = priority_key

    print('LIMS analysis: Completed')

    data_no_outliers[priority_key].to_csv(folder_data + '/LIMS_soft_sensors.csv', index = False)

    return data_holder_dict, results_holder_dict


def sensors_analyse(folder_results_excel, folder_data, data_date, write_sensors_csv_files, period_sensors):
    data_holder_dict = dict()
    results_holder_dict = dict()

    writer_sensors = pd.ExcelWriter(folder_results_excel + '/output results - sensors.xlsx', engine='xlsxwriter')

    feature_preselection_results_df = pd.DataFrame([])

    print('Sensors analysis: Start')

    print('Loading sensors data: start')
    data_sensors, labels, description = load_sensors_data(data_date, period_sensors)
    print('Loading sensors data: completed')

    if write_sensors_csv_files == True:
        data_sensors.to_csv(folder_data + 'data_raw.csv', index=False)
        print('Raw sensors data exported to: {}/data_raw.csv'.format(folder_data))

    feature_preselection_results_df = pd.concat([feature_preselection_results_df,
                                                 pd.DataFrame(data_sensors.drop('date', axis=1).columns)], axis=1)

    feature_preselection_results_df = pd.concat([feature_preselection_results_df,
                                                 pd.DataFrame(labels['Name'].loc[labels['Force'] == 'Yes'].reset_index(
                                                     drop=True))], axis=1)

    results_holder_dict['description'] = description

    # --- REMOVE SHORT TERM OUTLIERS FROM SENSORS ---

    print('Removing short-term outliers from sensors: start')
    data_sensors, data_no_short_outliers, data_smooth, outliers_list, STO_results = remove_short_term_outliers(
        data_sensors, data_sensors.drop('date', axis=1).columns)
    print('Removing short-term outliers from sensors: completed')

    STO_results.round(2).to_excel(writer_sensors, sheet_name='Short-term outliers - Summary')

    if write_sensors_csv_files == True:
        data_smooth.to_csv(folder_data + 'data_steady_states.csv', index=False)
        print('Steady states data exported to: {}/data_no_outliers.csv'.format(folder_data))

        data_no_short_outliers.to_csv(folder_data + '/' + 'data_no_short_outliers.csv', index=False)
        print('No short outliers data exported to: {}/data_no_short_outliers.csv'.format(folder_data))

    data_holder_dict['raw_sensors'] = data_sensors
    data_holder_dict['sensors_no_short_outliers'] = data_no_short_outliers
    data_holder_dict['sensors_smooth_data'] = data_smooth

    # --- REMOVE LONG TERM OUTLIERS FROM SENSORS ---
    print('Removing long-term outliers from sensors: start')
    data_sensors_no_outliers, long_term_outliers_dates, PCA_results = remove_long_term_outliers(data_no_short_outliers)
    print('Removing long-term outliers from sensors: completed')

    PCA_results['summary'].round(2).to_excel(writer_sensors, sheet_name='PCA summary', index=False)

    data_sensors.to_csv(folder_data + '/' + 'data_no_outliers.csv', index=False)
    print('Outliers-filtered sensors data exported to: {}/data_no_outliers.csv'.format(folder_data))

    data_holder_dict['sensors_no_outliers'] = data_sensors_no_outliers
    results_holder_dict['PCA'] = PCA_results

    if write_sensors_csv_files == True:
        results_holder_dict['PCA']['X_pca'].to_csv(folder_data + 'PCA.csv', index=False)
        print('Steady states data exported to: {}/PCA.csv'.format(folder_data))

    # --- FEATURES PRE-SELECTION ---
    print('Feature pre-selection with PCA: start')
    data_pre_selected, tags_accepted, tags_rejected, results_featurepreselection = feature_pre_selection(
        data_sensors_no_outliers, labels)
    print('Feature pre-selection with PCA: completed')

    if write_sensors_csv_files == True:
        data_pre_selected.to_csv(folder_data + 'data_pre_selected.csv', index=False)
        print('Sensors after pre-selection data exported to: {}/data_pre_selected.csv'.format(folder_data))

    feature_preselection_results_df = pd.concat([feature_preselection_results_df,
                                                 pd.DataFrame(tags_accepted,
                                                              columns=['Pre-selected features - accepted'])], axis=1)

    feature_preselection_results_df = pd.concat([feature_preselection_results_df,
                                                 pd.DataFrame(tags_rejected,
                                                              columns=['Pre-selected features - rejected'])], axis=1)

    feature_preselection_results_df = pd.concat([feature_preselection_results_df,
                                                 pd.DataFrame(data_pre_selected.drop('date', axis=1).columns,
                                                              columns=['Final features'])], axis=1)

    feature_preselection_results_df.columns = ['Original features', 'Forced to accept',
                                               'Pre-selected features - accepted', 'Pre-selected features - rejected',
                                               'Final features']

    feature_preselection_results_df.to_excel(writer_sensors, sheet_name='Features summary', index=False)

    results_featurepreselection.round(2).to_excel(writer_sensors, sheet_name='Feature Pre-Selection Matrix')

    data_holder_dict['sensors_pre_selected'] = data_pre_selected

    print('Correlations in sensors: start')
    corr_df = correlations_sensors(data_sensors_no_outliers)
    print('Correlations in sensors: completed')

    corr_df.index = corr_df.columns

    corr_df.round(3).to_excel(writer_sensors, sheet_name='Correlations map')

    results_holder_dict['corr_map_sensors'] = corr_df

    writer_sensors.save()

    print('Sensors analysis: Completed')

    return data_holder_dict, results_holder_dict


def load_aligned_data():
    data_LIMS = pd.read_csv(folder_path + '/y.csv')
    data_sensors = pd.read_csv(folder_path + '/X.csv')
    
    return data_sensors, data_LIMS


def soft_sensors_analyse(priority_key, data_LIMS, data_sensors, results_holder_dict, value_search, folder_results_excel,
                         folder_models_soft_sensors, description, align_in_time):
    writer_soft_sensors = pd.ExcelWriter(folder_results_excel + '/output results - soft sensors.xlsx',
                                         engine='xlsxwriter')
    
    if align_in_time == 'align':

        data_sensors, data_LIMS, data_date, description = time_alignment(priority_key, data_LIMS, data_sensors, description)
    
        pd.concat([data_date, data_sensors, data_LIMS], axis=1).to_excel(writer_soft_sensors,
                                                                         sheet_name='Processed & aligned data', index=False)
                                                                         
    if align_in_time == 'load_aligned':
        data_sensors, data_LIMS = load_aligned_data()
        data_date = np.arange(1, data_LIMS.shape[0])
        description = [None]
        
        
    print('Soft sensors analysis: Started')
                                    
    print('Correlations between LIMS and sensors data: start')
    data_corr = correlations_LIMS_sensors(data_LIMS, data_sensors)
    print('Correlations between LIMS and sensors data: completed')

    data_corr.index = data_corr.columns
    data_corr.to_excel(writer_soft_sensors, sheet_name='LIMS and sensors correlations')

    print('LIMS + sensors data splitting: start')
    X, y, X_train, X_test, X_valid, y_train, y_test, y_valid, X_scaled, tags, names = data_split(data_sensors,
                                                                                                 data_LIMS, 0.35, 0.3,
                                                                                                 15)
    print('LIMS + sensors data splitting: completed')

    pd.DataFrame(X_train, columns=tags).to_excel(writer_soft_sensors, sheet_name='Scaled input X - training',
                                                 index=False)
    pd.DataFrame(y_train, columns=names).to_excel(writer_soft_sensors, sheet_name='Scaled output y - training',
                                                  index=False)
    pd.DataFrame(X_test, columns=tags).to_excel(writer_soft_sensors, sheet_name='Scaled input X - testing', index=False)
    pd.DataFrame(y_test, columns=names).to_excel(writer_soft_sensors, sheet_name='Scaled output y - testing',
                                                 index=False)
    pd.DataFrame(X_valid, columns=tags).to_excel(writer_soft_sensors, sheet_name='Scaled input X - validation',
                                                 index=False)
    pd.DataFrame(y_valid, columns=names).to_excel(writer_soft_sensors, sheet_name='Scaled output y - validation',
                                                  index=False)

    thresholds = ARIMA_thresholds(data_LIMS, results_holder_dict)

    if value_search == False:
        print('Type of ANN search: Quick - start')

        performance_dict, importance_dict = quick_search(X_train, X_test, X_valid, y_train, y_test, y_valid, tags,
                                                         folder_models_soft_sensors)

        output_df = pd.DataFrame([])

        output_df['Sensor name'] = importance_dict['feature_imp_sorted_labels']
        output_df['Sensor importance'] = importance_dict['feature_imp_sorted']

        output_df.round(2).to_excel(writer_soft_sensors, sheet_name='Quick search - features', index=False)

        output_df = pd.DataFrame([])

        output_df['Product'] = names
        output_df['Training MAE'] = np.round(performance_dict['training_error'], 2)
        output_df['Testing MAE'] = np.round(performance_dict['testing_error'], 2)
        output_df['Validation MAE'] = np.round(performance_dict['validation_error'], 2)

        if priority_key != None:
            output_df['ARIMA threshold'] = np.array(thresholds, dtype = float)

        output_df.to_excel(writer_soft_sensors, sheet_name='Quick search - performance')

        print('Type of ANN search: Quick - completed')

    if value_search == True:
        print('Type of ANN search: Full - start')

        configuration, performance_dict, importance_dict = full_search(X_train, X_test, X_valid, y_train, y_test,
                                                                       y_valid, tags, folder_models_soft_sensors)

        workbook = writer_soft_sensors.book

        print(priority_key)

        if np.array(priority_key) == None:
            thresholds = np.zeros(len(names))

        for k in range(0, len(names)):

            output_df = np.array([])

            for i in configuration['Hidden layers']:
                for j in configuration['Number of nodes']:
                    output_df = np.append(output_df,
                                          np.array(['Layers: {} x Nodes: {}'.format(int(i), int(j)),
                                                    round(performance_dict['{}_{}'.format(int(i), int(j))][
                                                              'training_error'][k], 2),
                                                    round(performance_dict['{}_{}'.format(int(i), int(j))][
                                                              'testing_error'][k], 2),
                                                    round(performance_dict['{}_{}'.format(int(i), int(j))][
                                                              'validation_error'][k], 2),
                                                    thresholds[k]]))

            output_df = output_df.reshape(int(len(output_df) / 5), 5)

            output_df = pd.DataFrame(output_df)

            output_df.columns = ['Architecture', 'Training MAE', 'Testing MAE', 'Validation MAE', 'ARIMA threshold']

            output_df.to_excel(writer_soft_sensors, sheet_name='Full search - summary', startrow=1, startcol=k * 6,
                               index=False)

            worksheet = writer_soft_sensors.sheets['Full search - summary']

            worksheet.merge_range('{}1:{}1'.format(colnum_string(k * 6 + 1), colnum_string(k * 6 + 5)), str(names[k]))

        output_df = pd.DataFrame([])
        architecture = []

        for i in configuration['Hidden layers']:
            for j in configuration['Number of nodes']:
                tag = importance_dict['{}_{}'.format(int(i), int(j))]['feature_imp_sorted_labels']
                feature = np.round(importance_dict['{}_{}'.format(int(i), int(j))]['feature_imp_sorted'], 2)
                length = len(importance_dict['{}_{}'.format(int(i), int(j))]['feature_imp_sorted'])
                architecture.append('Hidden layers: {} x nodes: {}'.format(int(i), int(j)))

                feature_imp_df = pd.DataFrame(np.stack((tag, feature)).T, columns=['Tag', 'Importance'])

                output_df = pd.concat([output_df, feature_imp_df], axis=1)

        output_df.to_excel(writer_soft_sensors, sheet_name='Full search - feature', startrow=1, index=False)

        worksheet = writer_soft_sensors.sheets['Full search - feature']

        for i in range(0, len(architecture)):
            worksheet.merge_range('{}1:{}1'.format(colnum_string(i * 2 + 1), colnum_string(i * 2 + 2)), architecture[i])

    data_holder_dict = dict()
    results_holder_dict = dict()
    data_holder_dict['date_time'] = data_date
    data_holder_dict['description'] = description
    results_holder_dict['corr_map_combined'] = data_corr
    results_holder_dict['products'] = names
    results_holder_dict['sensors'] = tags
    data_holder_dict['X'] = X
    data_holder_dict['X_scaled'] = X_scaled
    data_holder_dict['y'] = y
    results_holder_dict['arima_thresholds'] = thresholds

    print('Type of ANN search: Full - completed')

    if value_search == True:
        results_holder_dict['soft_sensors_performance'] = performance_dict
        results_holder_dict['soft_sensors_configuration'] = configuration
        results_holder_dict['soft_sensors_feature_importance'] = importance_dict

    if value_search == False:
        results_holder_dict['soft_sensors_performance'] = performance_dict
        results_holder_dict['soft_sensors_feature_importance'] = importance_dict

    print('Soft sensors analysis: Completed')

    writer_soft_sensors.save()

    return data_holder_dict, results_holder_dict


def soft_sensors_visualisation(folder_visualisations, soft_sensors_plot_decision, results_holder_dict, data_holder_dict,
                               value_search, folder_models_soft_sensors):
    if soft_sensors_plot_decision == False:
        print('Visualisations for soft sensors: Disabled')

    if soft_sensors_plot_decision == True:

        print('Visualisations for soft sensors: Started')

        folder_visualisations_soft_sensors = folder_visualisations + '/Soft sensors/'

        try:
            os.makedirs(folder_visualisations_soft_sensors)

        except:
            pass

        buttons = []

        for idx, i in enumerate(results_holder_dict['sensors']):

            try:
                title = f"Sensor description: {results_holder_dict['description']['Description'][np.where(results_holder_dict['description']['Name'] == i)[0]]}"

            except:
                title = f"Sensor description: {i}"

            buttons.append(dict(label=str(i),
                                method="update",
                                args=[{"y": [data_holder_dict['X'][i]]},
                                      {'title': title}]))

        fig_X = go.Figure()

        fig_X.add_trace(go.Scatter(x=data_holder_dict['date_time'],
                                   y=data_holder_dict['X'][results_holder_dict['sensors'][0]],
                                   name=str(results_holder_dict['sensors'][0])))

        fig_X.update_layout(updatemenus=[dict(direction="down",
                                              x=0,
                                              y=1,
                                              showactive=True,
                                              buttons=list(buttons))])

        try:
            title = f"Sensor description: {results_holder_dict['description']['Description'][np.where(results_holder_dict['description']['Name'] == results_holder_dict['sensors'][0])[0]]}"

        except:
            title = f"Sensor description: {results_holder_dict['sensors'][0]}"

        fig_X.update_layout(xaxis_title='Date',
                            yaxis_title='Sensor measurement (a.u.)',
                            title=title)

        fig_X.write_html(folder_visualisations_soft_sensors + 'X.html')

        fig_y = go.Figure()

        for idx, i in enumerate(results_holder_dict['products']):
            fig_y.add_trace(go.Scatter(x=data_holder_dict['date_time'],
                                       y=data_holder_dict['y'][i],
                                       name=str(i)))

        fig_y.update_layout(xaxis_title='Date',
                            yaxis_title='Temperature (oC)',
                            title='Output data: Products')

        fig_y.write_html(folder_visualisations_soft_sensors + 'y.html')

        if bool(value_search) == False:

            fig_loss_function = go.Figure()

            fig_loss_function.add_trace(
                go.Scatter(x=np.arange(1, len(results_holder_dict['soft_sensors_performance']['history']['loss']) + 1),
                           y=results_holder_dict['soft_sensors_performance']['history']['loss'],
                           name='Training dataset'))

            fig_loss_function.add_trace(go.Scatter(
                x=np.arange(1, len(results_holder_dict['soft_sensors_performance']['history']['val_loss']) + 1),
                y=results_holder_dict['soft_sensors_performance']['history']['val_loss'],
                name='Testing dataset'))

            time_elapsed = int(np.linspace(1, results_holder_dict['soft_sensors_performance']['time_elapsed'], len(
                results_holder_dict['soft_sensors_performance']['history']['val_loss']))[
                                   np.argmin(results_holder_dict['soft_sensors_performance']['history']['val_loss'])])

            if time_elapsed < 60:
                text_time = f"{time_elapsed} seconds"

            if time_elapsed > 60 and time_elapsed < 3600:
                minutes = math.floor(time_elapsed / 60)
                seconds = time_elapsed - minutes * 60
                text_time = f"{int(minutes)} minutes and {int(seconds)} seconds"

            if time_elapsed > 3600:
                hours = math.floor(time_elapsed / 3600)
                minutes = math.floor((time_elapsed - hours * 3600) / 60)
                seconds = time_elapsed - minutes * 60 - hours * 3600
                text_time = f"{int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds"

            text = f"Products: {str(results_holder_dict['products'])} <br> Training errors: {str(np.array(results_holder_dict['soft_sensors_performance']['training_error']).round(2))} <br> Testing errors: {str(np.array(results_holder_dict['soft_sensors_performance']['testing_error']).round(2))} <br> Validation errors: {str(np.array(results_holder_dict['soft_sensors_performance']['validation_error']).round(2))} <br> Thresholds: {str(np.array(results_holder_dict['arima_thresholds']))} <br> Time to train the network: {text_time}"

            fig_loss_function.add_annotation(
                x=np.argmin(results_holder_dict['soft_sensors_performance']['history']['val_loss']) + 1,
                y=np.min(
                    results_holder_dict['soft_sensors_performance']['history']['val_loss']),
                axref="x",
                ayref="y",
                ax=int(np.argmin(
                    results_holder_dict['soft_sensors_performance']['history']['val_loss']) * 0.7),
                ay=np.max(
                    results_holder_dict['soft_sensors_performance']['history']['val_loss']),
                text=text,
                arrowhead=2)

            try:
                title = f"Configuration of the network for the quick search for the product {results_holder_dict['priority_key']}: <br> Number of hidden layers - {results_holder_dict['soft_sensors_performance']['configuration_layers']} <br> Number of nodes in each hidden layer - {results_holder_dict['soft_sensors_performance']['configuration_nodes']} <br> Activation functions: {results_holder_dict['soft_sensors_performance']['configuration_activation']} <br> Loss function: {results_holder_dict['soft_sensors_performance']['configuration_loss']}"

            except:
                title = f"Configuration of the network for the quick search for the product: <br> Number of hidden layers - {results_holder_dict['soft_sensors_performance']['configuration_layers']} <br> Number of nodes in each hidden layer - {results_holder_dict['soft_sensors_performance']['configuration_nodes']} <br> Activation functions: {results_holder_dict['soft_sensors_performance']['configuration_activation']} <br> Loss function: {results_holder_dict['soft_sensors_performance']['configuration_loss']}"

            fig_loss_function.update_layout(xaxis_title='Epochs',
                                            yaxis_title='Loss function',
                                            title=dict(text=title,
                                                       y=0.95,
                                                       x=0.5,
                                                       yanchor='bottom'),
                                            margin=dict(t=200, pad=4))

            fig_loss_function.write_html(folder_visualisations_soft_sensors + 'loss_function - quick search.html')

            fig_feature_imp = go.Figure()

            x_data = results_holder_dict['soft_sensors_feature_importance']['feature_imp_sorted_labels']
            y_data = results_holder_dict['soft_sensors_feature_importance']['feature_imp_sorted']
            y_data = (y_data - np.min(y_data)) / \
                     (np.max(y_data) - np.min(y_data))

            fig_feature_imp.add_trace(go.Scatter(x=x_data,
                                                 y=y_data))

            fig_feature_imp.update_layout(xaxis_title='Feature name',
                                          yaxis_title='Relative feature importance',
                                          title=dict(text=title,
                                                     y=0.95,
                                                     x=0.5,
                                                     yanchor='bottom'),
                                          margin=dict(t=200, pad=4))

            fig_feature_imp.update_xaxes(tickangle=90)

            fig_feature_imp.write_html(folder_visualisations_soft_sensors + 'feature_importance - quick search.html')

            model_name = 'quick_search_model'
            model = keras.models.load_model(
                folder_models_soft_sensors + '{}.h5'.format(model_name))
            deviations = model.predict(
                data_holder_dict['X_scaled']) - data_holder_dict['y']

            deviations_histogram = make_subplots(rows=3,
                                                 cols=1,
                                                 subplot_titles=(
                                                     "Histogram", "Boxplot", 'Deviations'),
                                                 vertical_spacing=0.1)

            for idx, i in enumerate(deviations.columns):
                deviations_histogram.add_trace(go.Histogram(x=deviations[i],
                                                            name=i,
                                                            marker_color=px.colors.qualitative.Plotly[idx]),
                                               row=1,
                                               col=1)

                deviations_histogram.add_trace(go.Box(y=deviations[i],
                                                      name=i,
                                                      marker_color=px.colors.qualitative.Plotly[idx],
                                                      showlegend=False,
                                                      hovertext=data_holder_dict['date_time']),
                                               row=2,
                                               col=1)

                deviations_histogram.add_trace(go.Scatter(x=data_holder_dict['date_time'],
                                                          y=deviations[i],
                                                          name=i,
                                                          marker_color=px.colors.qualitative.Plotly[idx],
                                                          showlegend=False, mode='markers',
                                                          text=data_holder_dict['date_time']),
                                               row=3,
                                               col=1)

            deviations_histogram.update_layout(
                barmode='overlay', title=title)

            deviations_histogram.write_html(
                folder_visualisations_soft_sensors + 'deviations_histogram - quick search.html')

        if bool(value_search) == True:

            fig_feature_imp_all = go.Figure()

            def annotation(results_holder_dict, i, j):

                time_elapsed = int(np.linspace(1, results_holder_dict['soft_sensors_performance'][
                    '{}_{}'.format(int(i), int(j))]['time_elapsed'],
                                               len(results_holder_dict['soft_sensors_performance']['{}_{}'.format(
                                                   int(i), int(j))]['history']['val_loss']))[np.argmin(
                    results_holder_dict['soft_sensors_performance']['{}_{}'.format(int(i), int(j))]['history'][
                        'val_loss'])])

                if time_elapsed < 60:
                    text_time = f"{time_elapsed} seconds"

                if time_elapsed > 60 and time_elapsed < 3600:
                    minutes = math.floor(time_elapsed / 60)
                    seconds = time_elapsed - minutes * 60
                    text_time = f"{int(minutes)} minutes and {int(seconds)} seconds"

                if time_elapsed > 3600:
                    hours = math.floor(time_elapsed / 3600)
                    minutes = math.floor(
                        (time_elapsed - hours * 3600) / 60)
                    seconds = time_elapsed - minutes * 60 - hours * 3600
                    text_time = f"{int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds"

                text = f"Products: {str(results_holder_dict['products'])} <br> Training errors: {str(results_holder_dict['soft_sensors_performance']['{}_{}'.format(int(i), int(j))]['training_error'].round(2))} <br> Testing errors: {str(results_holder_dict['soft_sensors_performance']['{}_{}'.format(int(i), int(j))]['testing_error'].round(2))} <br> Validation errors: {str(results_holder_dict['soft_sensors_performance']['{}_{}'.format(int(i), int(j))]['validation_error'].round(2))} <br> Thresholds: {str(results_holder_dict['arima_thresholds'].round(2))} <br> Time to train the network: {text_time}"

                annotation = dict(x=np.argmin(
                    results_holder_dict['soft_sensors_performance']['{}_{}'.format(int(i), int(j))]['history'][
                        'val_loss']) + 1,
                                  y=np.min(results_holder_dict['soft_sensors_performance']['{}_{}'.format(
                                      int(i), int(j))]['history']['val_loss']),
                                  axref="x",
                                  ayref="y",
                                  ax=int(np.argmin(results_holder_dict['soft_sensors_performance']['{}_{}'.format(
                                      int(i), int(j))]['history']['val_loss']) * 0.7),
                                  ay=np.max(results_holder_dict['soft_sensors_performance']['{}_{}'.format(
                                      int(i), int(j))]['history']['val_loss']),
                                  text=text,
                                  arrowhead=2)

                return annotation

            fig_loss_function = go.Figure()
            fig_feature_imp = go.Figure()
            fig_feature_imp_all = go.Figure()

            buttons_loss = []
            button_feature = []

            xticks_modified = []

            for i in results_holder_dict['soft_sensors_configuration']['Hidden layers']:
                for j in results_holder_dict['soft_sensors_configuration']['Number of nodes']:

                    model_name = 'quick_search_model'
                    model = keras.models.load_model(
                        folder_models_soft_sensors + 'layers_{}_nodes_{}.h5'.format(i, j))
                    deviations = model.predict(
                        data_holder_dict['X_scaled']) - data_holder_dict['y']

                    print(deviations)

                    try:
                        title = f"Configuration of the network the product {results_holder_dict['priority_key']}: <br> Number of hidden layers - {i} <br> Number of nodes in each hidden layer - {j} <br> Activation functions: {results_holder_dict['soft_sensors_configuration']['Activation function']} <br> Loss function: {results_holder_dict['soft_sensors_configuration']['Loss function']}"

                    except:
                        title = f"Configuration of the network the product: <br> Number of hidden layers - {i} <br> Number of nodes in each hidden layer - {j} <br> Activation functions: {results_holder_dict['soft_sensors_configuration']['Activation function']} <br> Loss function: {results_holder_dict['soft_sensors_configuration']['Loss function']}"

                    deviations_histogram = make_subplots(rows=3,
                                                         cols=1,
                                                         subplot_titles=(
                                                             "Histogram", "Boxplot", 'Deviations'),
                                                         vertical_spacing=0.1)

                    for kdx, k in enumerate(deviations.columns):
                        deviations_histogram.add_trace(go.Histogram(x=deviations[k],
                                                                    name=k,
                                                                    marker_color=px.colors.qualitative.Plotly[kdx]),
                                                       row=1,
                                                       col=1)

                        deviations_histogram.add_trace(go.Box(y=deviations[k],
                                                              name=k,
                                                              marker_color=px.colors.qualitative.Plotly[kdx],
                                                              showlegend=False,
                                                              hovertext=data_holder_dict['date_time']),
                                                       row=2,
                                                       col=1)

                        deviations_histogram.add_trace(go.Scatter(x=data_holder_dict['date_time'],
                                                                  y=deviations[k],
                                                                  name=k,
                                                                  marker_color=px.colors.qualitative.Plotly[kdx],
                                                                  showlegend=False, mode='markers',
                                                                  text=data_holder_dict['date_time']),
                                                       row=3,
                                                       col=1)

                    deviations_histogram.update_layout(
                        barmode='overlay', title=title)

                    deviations_histogram.write_html(
                        folder_visualisations_soft_sensors + 'histogram layers_{}_nodes_{}.html'.format(i, j))

                    xticks_modified.append(f"{i} x {j}")

                    buttons_loss.append(dict(label=f"Hidden layers: {i}, number of nodes: {j}",
                                             method="update",
                                             args=[{"x": [np.arange(1, len(
                                                 results_holder_dict['soft_sensors_performance'][
                                                     '{}_{}'.format(int(i), int(j))]['history']['loss']) + 1),
                                                          np.arange(1, len(
                                                              results_holder_dict['soft_sensors_performance'][
                                                                  '{}_{}'.format(int(i), int(j))]['history'][
                                                                  'val_loss']) + 1)],
                                                    'y': [results_holder_dict['soft_sensors_performance'][
                                                              '{}_{}'.format(int(i), int(j))]['history']['loss'],
                                                          results_holder_dict['soft_sensors_performance'][
                                                              '{}_{}'.format(int(i), int(j))]['history']['val_loss']]},
                                                   {'annotations': [annotation(results_holder_dict, i, j)],
                                                    'title': title}]))

                    x_data = results_holder_dict['soft_sensors_feature_importance']['{}_{}'.format(
                        int(i), int(j))]['feature_imp_sorted_labels']
                    y_data = results_holder_dict['soft_sensors_feature_importance']['{}_{}'.format(
                        int(i), int(j))]['feature_imp_sorted']
                    y_data = (y_data - np.min(y_data)) / \
                             (np.max(y_data) - np.min(y_data))

                    button_feature.append(dict(label=f"Hidden layers: {i}, number of nodes: {j}",
                                               method="update",
                                               args=[{"x": [x_data],
                                                      'y': [y_data]},
                                                     {'title': title}]))

                    x_data = results_holder_dict['soft_sensors_feature_importance']['{}_{}'.format(
                        int(i), int(j))]['labels']
                    y_data = results_holder_dict['soft_sensors_feature_importance']['{}_{}'.format(
                        int(i), int(j))]['feature_imp']
                    y_data = (y_data - np.min(y_data)) / \
                             (np.max(y_data) - np.min(y_data))

                    fig_feature_imp_all.add_trace(go.Scatter(x=x_data,
                                                             y=y_data,
                                                             name=f"Hidden layers: {i}, number of nodes: {j}"))

            fig_loss_function.add_trace(go.Scatter(x=np.arange(1, len(results_holder_dict['soft_sensors_performance'][
                                                                          list(results_holder_dict[
                                                                                   'soft_sensors_performance'].keys())[
                                                                              0]]['history']['loss']) + 1),
                                                   y=results_holder_dict['soft_sensors_performance'][list(
                                                       results_holder_dict['soft_sensors_performance'].keys())[0]][
                                                       'history']['loss'],
                                                   name='Training set'))

            fig_loss_function.add_trace(go.Scatter(x=np.arange(1, len(results_holder_dict['soft_sensors_performance'][
                                                                          list(results_holder_dict[
                                                                                   'soft_sensors_performance'].keys())[
                                                                              0]]['history']['val_loss']) + 1),
                                                   y=results_holder_dict['soft_sensors_performance'][list(
                                                       results_holder_dict['soft_sensors_performance'].keys())[0]][
                                                       'history']['val_loss'],
                                                   name='Testing set'))

            x_data = results_holder_dict['soft_sensors_feature_importance'][list(
                results_holder_dict['soft_sensors_feature_importance'].keys())[0]]['feature_imp_sorted_labels']
            y_data = results_holder_dict['soft_sensors_feature_importance'][list(
                results_holder_dict['soft_sensors_feature_importance'].keys())[0]]['feature_imp_sorted']
            y_data = (y_data - np.min(y_data)) / \
                     (np.max(y_data) - np.min(y_data))

            fig_feature_imp.add_trace(go.Scatter(x=x_data,
                                                 y=y_data))

            time_elapsed = int(np.linspace(1, results_holder_dict['soft_sensors_performance'][
                list(results_holder_dict['soft_sensors_feature_importance'].keys())[0]]['time_elapsed'],
                                           len(results_holder_dict['soft_sensors_performance'][list(
                                               results_holder_dict['soft_sensors_feature_importance'].keys())[0]][
                                                   'history']['val_loss']))[np.argmin(
                results_holder_dict['soft_sensors_performance'][
                    list(results_holder_dict['soft_sensors_feature_importance'].keys())[0]]['history']['val_loss'])])

            if time_elapsed < 60:
                text_time = f"{time_elapsed} seconds"

            if time_elapsed > 60 and time_elapsed < 3600:
                minutes = math.floor(time_elapsed / 60)
                seconds = time_elapsed - minutes * 60
                text_time = f"{int(minutes)} minutes and {int(seconds)} seconds"

            if time_elapsed > 3600:
                hours = math.floor(time_elapsed / 3600)
                minutes = math.floor((time_elapsed - hours * 3600) / 60)
                seconds = time_elapsed - minutes * 60 - hours * 3600
                text_time = f"{int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds"

            text = f"Products: {str(results_holder_dict['products'])} <br> Training errors: {str(results_holder_dict['soft_sensors_performance'][list(results_holder_dict['soft_sensors_performance'].keys())[0]]['training_error'].round(2))} <br> Testing errors: {str(results_holder_dict['soft_sensors_performance'][list(results_holder_dict['soft_sensors_performance'].keys())[0]]['testing_error'].round(2))} <br> Validation errors: {str(results_holder_dict['soft_sensors_performance'][list(results_holder_dict['soft_sensors_performance'].keys())[0]]['validation_error'].round(2))} <br> Thresholds: {str(results_holder_dict['arima_thresholds'].round(2))} <br> Time to train the network: {text_time}"

            fig_loss_function.add_annotation(x=np.argmin(results_holder_dict['soft_sensors_performance'][list(
                results_holder_dict['soft_sensors_performance'].keys())[0]]['history']['val_loss']) + 1,
                                             y=np.min(results_holder_dict['soft_sensors_performance'][list(
                                                 results_holder_dict['soft_sensors_performance'].keys())[0]]['history'][
                                                          'val_loss']),
                                             axref="x",
                                             ayref="y",
                                             ax=int(np.argmin(results_holder_dict['soft_sensors_performance'][list(
                                                 results_holder_dict['soft_sensors_performance'].keys())[0]]['history'][
                                                                  'val_loss']) * 0.7),
                                             ay=np.max(results_holder_dict['soft_sensors_performance'][list(
                                                 results_holder_dict['soft_sensors_performance'].keys())[0]]['history'][
                                                           'val_loss']),
                                             text=text,
                                             arrowhead=2)

            fig_loss_function.update_layout(updatemenus=[dict(direction="down",
                                                              x=0,
                                                              y=1.05,
                                                              xanchor='left',
                                                              yanchor='bottom',
                                                              showactive=True,
                                                              buttons=list(buttons_loss))])

            fig_feature_imp.update_layout(updatemenus=[dict(direction="down",
                                                            x=0,
                                                            y=1.05,
                                                            xanchor='left',
                                                            yanchor='bottom',
                                                            showactive=True,
                                                            buttons=list(button_feature))])

            title = f"Configuration of the network the product: <br> Number of hidden layers - {results_holder_dict['soft_sensors_configuration']['Hidden layers'][0]} <br> Number of nodes in each hidden layer - {results_holder_dict['soft_sensors_configuration']['Number of nodes'][0]} <br> Activation functions: {results_holder_dict['soft_sensors_configuration']['Activation function']} <br> Loss function: {results_holder_dict['soft_sensors_configuration']['Loss function']}"

            fig_feature_imp.update_layout(xaxis_title='Feature name',
                                          yaxis_title='Relative feature importance',
                                          title=dict(text=title,
                                                     y=0.95,
                                                     x=0.5,
                                                     yanchor='bottom'),
                                          margin=dict(t=200, pad=4))

            fig_feature_imp.update_xaxes(tickangle=90)

            fig_loss_function.update_layout(xaxis_title='Epochs',
                                            yaxis_title='Loss function',
                                            title=dict(text=title,
                                                       y=0.95,
                                                       x=0.5,
                                                       yanchor='bottom'),
                                            margin=dict(t=200, pad=4))

            fig_loss_function.write_html(
                folder_visualisations_soft_sensors + "loss_function - full search.html")

            fig_feature_imp.write_html(
                folder_visualisations_soft_sensors + 'feature_importance - full search.html')

            fig_feature_imp_all.update_xaxes(tickangle=90)
            fig_feature_imp_all.update_layout(xaxis_title='Feature name',
                                              yaxis_title='Relative feature importance (scaled between 0 and 1)')

            fig_feature_imp_all.write_html(
                folder_visualisations_soft_sensors + 'feature_importance_all - full search.html')

            # mean
            results_full_search_df_train = pd.DataFrame(
                np.zeros((int(len(results_holder_dict['soft_sensors_configuration']['Hidden layers']) * len(
                    results_holder_dict['soft_sensors_configuration']['Number of nodes'])),
                          len(results_holder_dict['products']))))
            results_full_search_df_train.index = results_holder_dict['soft_sensors_performance'].keys(
            )
            results_full_search_df_train.columns = results_holder_dict['products']

            results_full_search_df_test = pd.DataFrame(
                np.zeros((int(len(results_holder_dict['soft_sensors_configuration']['Hidden layers']) * len(
                    results_holder_dict['soft_sensors_configuration']['Number of nodes'])),
                          len(results_holder_dict['products']))))
            results_full_search_df_test.index = results_holder_dict['soft_sensors_performance'].keys(
            )
            results_full_search_df_test.columns = results_holder_dict['products']

            results_full_search_df_valid = pd.DataFrame(
                np.zeros((int(len(results_holder_dict['soft_sensors_configuration']['Hidden layers']) * len(
                    results_holder_dict['soft_sensors_configuration']['Number of nodes'])),
                          len(results_holder_dict['products']))))
            results_full_search_df_valid.index = results_holder_dict['soft_sensors_performance'].keys(
            )
            results_full_search_df_valid.columns = results_holder_dict['products']

            for i in results_full_search_df_train.index:
                results_full_search_df_train.loc[i,
                :] = results_holder_dict['soft_sensors_performance'][i]['training_error']
                results_full_search_df_test.loc[i,
                :] = results_holder_dict['soft_sensors_performance'][i]['testing_error']
                results_full_search_df_valid.loc[i,
                :] = results_holder_dict['soft_sensors_performance'][i]['validation_error']

            fig_full_search_performance = make_subplots(rows=3,
                                                        cols=1,
                                                        subplot_titles=(
                                                            "Training data set", "Testing data set",
                                                            "Validation data set"),
                                                        vertical_spacing=0.1)

            for idx, i in enumerate(results_full_search_df_train.columns):
                fig_full_search_performance.add_trace(go.Scatter(x=xticks_modified,
                                                                 y=results_full_search_df_train.loc[:, i],
                                                                 name=str(
                                                                     i),
                                                                 mode='lines+markers',
                                                                 marker=dict(
                                                                     color=px.colors.qualitative.Plotly[idx]),
                                                                 showlegend=False),
                                                      row=1, col=1)

                fig_full_search_performance.add_trace(go.Scatter(x=xticks_modified,
                                                                 y=results_full_search_df_test.loc[:, i],
                                                                 name=str(
                                                                     i),
                                                                 mode='lines+markers',
                                                                 marker=dict(
                                                                     color=px.colors.qualitative.Plotly[idx]),
                                                                 showlegend=False),
                                                      row=2, col=1)

                fig_full_search_performance.add_trace(go.Scatter(x=xticks_modified,
                                                                 y=results_full_search_df_valid.loc[:, i],
                                                                 name=str(
                                                                     i),
                                                                 mode='lines+markers',
                                                                 marker=dict(color=px.colors.qualitative.Plotly[idx])),
                                                      row=3, col=1)

            fig_full_search_performance.update_yaxes(
                title_text="Mean absolute deviation MAD", row=1, col=1)
            fig_full_search_performance.update_yaxes(
                title_text="Mean absolute deviation MAD", row=2, col=1)
            fig_full_search_performance.update_yaxes(
                title_text="Mean absolute deviation MAD", row=3, col=1)

            fig_full_search_performance.update_xaxes(
                title_text="Architecture", row=1, col=1)
            fig_full_search_performance.update_xaxes(
                title_text="Architecture", row=2, col=1)
            fig_full_search_performance.update_xaxes(
                title_text="Architecture", row=3, col=1)

            fig_full_search_performance.update_layout(
                title='Mean absolute deviation between predicted output product and actual values')

            fig_full_search_performance.write_html(
                folder_visualisations_soft_sensors + 'performance_comparison_mean - full search.html')

            # std

            results_full_search_df_train = pd.DataFrame(
                np.zeros((int(len(results_holder_dict['soft_sensors_configuration']['Hidden layers']) * len(
                    results_holder_dict['soft_sensors_configuration']['Number of nodes'])),
                          len(results_holder_dict['products']))))
            results_full_search_df_train.index = results_holder_dict['soft_sensors_performance'].keys(
            )
            results_full_search_df_train.columns = results_holder_dict['products']

            results_full_search_df_test = pd.DataFrame(
                np.zeros((int(len(results_holder_dict['soft_sensors_configuration']['Hidden layers']) * len(
                    results_holder_dict['soft_sensors_configuration']['Number of nodes'])),
                          len(results_holder_dict['products']))))
            results_full_search_df_test.index = results_holder_dict['soft_sensors_performance'].keys(
            )
            results_full_search_df_test.columns = results_holder_dict['products']

            results_full_search_df_valid = pd.DataFrame(
                np.zeros((int(len(results_holder_dict['soft_sensors_configuration']['Hidden layers']) * len(
                    results_holder_dict['soft_sensors_configuration']['Number of nodes'])),
                          len(results_holder_dict['products']))))
            results_full_search_df_valid.index = results_holder_dict['soft_sensors_performance'].keys(
            )
            results_full_search_df_valid.columns = results_holder_dict['products']

            for i in results_full_search_df_train.index:
                results_full_search_df_train.loc[i,
                :] = results_holder_dict['soft_sensors_performance'][i]['std']['training']
                results_full_search_df_test.loc[i,
                :] = results_holder_dict['soft_sensors_performance'][i]['std']['testing']
                results_full_search_df_valid.loc[i,
                :] = results_holder_dict['soft_sensors_performance'][i]['std']['validation']

            fig_full_search_performance = make_subplots(rows=3,
                                                        cols=1,
                                                        subplot_titles=(
                                                            "Training data set", "Testing data set",
                                                            "Validation data set"),
                                                        vertical_spacing=0.1)

            for idx, i in enumerate(results_full_search_df_train.columns):
                fig_full_search_performance.add_trace(go.Scatter(x=xticks_modified,
                                                                 y=results_full_search_df_train.loc[:, i],
                                                                 name=str(
                                                                     i),
                                                                 mode='lines+markers',
                                                                 marker=dict(
                                                                     color=px.colors.qualitative.Plotly[idx]),
                                                                 showlegend=False),
                                                      row=1, col=1)

                fig_full_search_performance.add_trace(go.Scatter(x=xticks_modified,
                                                                 y=results_full_search_df_test.loc[:, i],
                                                                 name=str(
                                                                     i),
                                                                 mode='lines+markers',
                                                                 marker=dict(
                                                                     color=px.colors.qualitative.Plotly[idx]),
                                                                 showlegend=False),
                                                      row=2, col=1)

                fig_full_search_performance.add_trace(go.Scatter(x=xticks_modified,
                                                                 y=results_full_search_df_valid.loc[:, i],
                                                                 name=str(
                                                                     i),
                                                                 mode='lines+markers',
                                                                 marker=dict(color=px.colors.qualitative.Plotly[idx])),
                                                      row=3, col=1)

            fig_full_search_performance.update_yaxes(
                title_text="Standard deviation", row=1, col=1)
            fig_full_search_performance.update_yaxes(
                title_text="Standard deviation", row=2, col=1)
            fig_full_search_performance.update_yaxes(
                title_text="Standard deviation", row=3, col=1)

            fig_full_search_performance.update_xaxes(
                title_text="Architecture", row=1, col=1)
            fig_full_search_performance.update_xaxes(
                title_text="Architecture", row=2, col=1)
            fig_full_search_performance.update_xaxes(
                title_text="Architecture", row=3, col=1)

            fig_full_search_performance.update_layout(
                title='Standard deviation between predicted output product and actual values')

            fig_full_search_performance.write_html(
                folder_visualisations_soft_sensors + 'performance_comparison_std - full search.html')

            # --- maximum

            for i in results_full_search_df_train.index:
                results_full_search_df_train.loc[i,
                :] = results_holder_dict['soft_sensors_performance'][i]['max']['training']
                results_full_search_df_test.loc[i,
                :] = results_holder_dict['soft_sensors_performance'][i]['max']['testing']
                results_full_search_df_valid.loc[i,
                :] = results_holder_dict['soft_sensors_performance'][i]['max']['validation']

            fig_full_search_performance = make_subplots(rows=3,
                                                        cols=1,
                                                        subplot_titles=(
                                                            "Training data set", "Testing data set",
                                                            "Validation data set"),
                                                        vertical_spacing=0.1)

            for idx, i in enumerate(results_full_search_df_train.columns):
                fig_full_search_performance.add_trace(go.Scatter(x=xticks_modified,
                                                                 y=results_full_search_df_train.loc[:, i],
                                                                 name=str(
                                                                     i),
                                                                 mode='lines+markers',
                                                                 marker=dict(
                                                                     color=px.colors.qualitative.Plotly[idx]),
                                                                 showlegend=False),
                                                      row=1, col=1)

                fig_full_search_performance.add_trace(go.Scatter(x=xticks_modified,
                                                                 y=results_full_search_df_test.loc[:, i],
                                                                 name=str(
                                                                     i),
                                                                 mode='lines+markers',
                                                                 marker=dict(
                                                                     color=px.colors.qualitative.Plotly[idx]),
                                                                 showlegend=False),
                                                      row=2, col=1)

                fig_full_search_performance.add_trace(go.Scatter(x=xticks_modified,
                                                                 y=results_full_search_df_valid.loc[:, i],
                                                                 name=str(
                                                                     i),
                                                                 mode='lines+markers',
                                                                 marker=dict(color=px.colors.qualitative.Plotly[idx])),
                                                      row=3, col=1)

            fig_full_search_performance.update_yaxes(
                title_text="Maximum deviation", row=1, col=1)
            fig_full_search_performance.update_yaxes(
                title_text="Maximum deviation", row=2, col=1)
            fig_full_search_performance.update_yaxes(
                title_text="Maximum deviation", row=3, col=1)

            fig_full_search_performance.update_xaxes(
                title_text="Architecture", row=1, col=1)
            fig_full_search_performance.update_xaxes(
                title_text="Architecture", row=2, col=1)
            fig_full_search_performance.update_xaxes(
                title_text="Architecture", row=3, col=1)

            fig_full_search_performance.update_layout(
                title='Standard deviation between predicted output product and actual values')

            fig_full_search_performance.write_html(
                folder_visualisations_soft_sensors + 'performance_comparison_max - full search.html')

        print('Visualisations for soft sensors: Completed')


def LIMS_visualisation(LIMS_plot_decision, data_holder_dict, results_holder_dict, folder_visualisations):
    if bool(LIMS_plot_decision) == False:
        print('Visualisations for LIMS: Disabled')

    if bool(LIMS_plot_decision) == True:

        print('Visualisations for LIMS: Started')

        folder_visualisations_LIMS = folder_visualisations + '/LIMS/'

        print(folder_visualisations)
        print(folder_visualisations_LIMS)

        try:
            os.makedirs(folder_visualisations_LIMS)

        except:
            pass

        # Define figures holder
        figures_no_zeros = []
        figures_no_zeros_histogram_with_outliers = []
        figures_no_zeros_histogram_without_outliers = []
        figures_boxplots = []
        figures_outliers = []

        for i in data_holder_dict['data_LIMS_separated_nozeros'].keys():

            # Define single figure
            figure_no_zeros = go.Figure()
            figure_raw = go.Figure()
            figure_boxplot = go.Figure()
            figure_outliers = go.Figure()
            figure_no_duplicates = go.Figure()

            # Select data for i
            data = data_holder_dict['data_LIMS_separated_nozeros_noduplicates'][i].drop(
                'time', axis=1)

            for position_j, j in enumerate(data.drop('date', axis=1).columns):
                # All data - raw

                x_data = data_holder_dict['data_LIMS_separated'][i]['date']
                y_data = data_holder_dict['data_LIMS_separated'][i].drop('time', axis=1)[
                    j]
                figure_raw.add_trace(
                    go.Scatter(x=x_data,
                               y=y_data,
                               mode='markers',
                               name='%s' % (str(j)),
                               showlegend=True,
                               marker_color=px.colors.qualitative.Plotly[position_j]))

                # All data - no zeros
                x_data = data_holder_dict['data_LIMS_separated_nozeros'][i]['date']
                y_data = data_holder_dict['data_LIMS_separated_nozeros'][i].drop('time', axis=1)[
                    j]
                figure_no_zeros.add_trace(go.Scatter(x=x_data,
                                                     y=y_data,
                                                     mode='markers',
                                                     name='%s' % (str(j)),
                                                     showlegend=True,
                                                     marker_color=px.colors.qualitative.Plotly[position_j]))

                # All data - no duplicates
                x_data = data_holder_dict['data_LIMS_separated_nozeros_noduplicates'][i]['date']
                y_data = data_holder_dict['data_LIMS_separated_nozeros_noduplicates'][i].drop(
                    'time', axis=1)[j]
                figure_no_duplicates.add_trace(go.Scatter(x=x_data,
                                                          y=y_data,
                                                          mode='markers',
                                                          name='%s' % (
                                                              str(j)),
                                                          showlegend=True,
                                                          marker_color=px.colors.qualitative.Plotly[position_j]))

                # Boxplots
                y_data = data_holder_dict['data_LIMS_separated_nozeros_noduplicates'][i].drop(
                    'time', axis=1)[j]
                figure_boxplot.add_trace(go.Box(y=y_data,
                                                name='%s' % (str(j)),
                                                boxmean='sd',
                                                boxpoints='all'))

                # Outliers
                x_data = data_holder_dict['data_LIMS_separated_nozeros_noduplicates'][i]['date']
                y_data = data_holder_dict['data_LIMS_separated_nozeros_noduplicates'][i].drop(
                    'time', axis=1)[j]
                figure_outliers.add_trace(go.Scatter(x=x_data,
                                                     y=y_data,
                                                     mode='markers',
                                                     name='%s' % (str(j)),
                                                     showlegend=True,
                                                     marker_color=px.colors.qualitative.Plotly[position_j],
                                                     marker=dict(
                                                         opacity=0.5),
                                                     marker_symbol='circle-open'))

                x_data = data_holder_dict['data_LIMS_separated_nozeros_noduplicates'][i]['date'].loc[data[j]
                                                                                                     >
                                                                                                     results_holder_dict[
                                                                                                         'limits'][
                                                                                                         'upper'][j]]
                y_data = data[j].loc[data[j] >
                                     results_holder_dict['limits']['upper'][j]]
                figure_outliers.add_trace(go.Scatter(x=x_data,
                                                     y=y_data,
                                                     mode='markers',
                                                     name='%s' % (str(j)),
                                                     showlegend=False,
                                                     marker_color=px.colors.qualitative.Plotly[position_j],
                                                     marker_symbol='x',
                                                     marker=dict(size=14)))

                x_data = data_holder_dict['data_LIMS_separated_nozeros_noduplicates'][i]['date'].loc[data[j]
                                                                                                     <
                                                                                                     results_holder_dict[
                                                                                                         'limits'][
                                                                                                         'lower'][j]]
                y_data = data[j].loc[data[j] <
                                     results_holder_dict['limits']['lower'][j]]

                figure_outliers.add_trace(go.Scatter(x=x_data,
                                                     y=y_data,
                                                     mode='markers',
                                                     name='%s' % (str(j)),
                                                     showlegend=False,
                                                     marker_color=px.colors.qualitative.Plotly[position_j],
                                                     marker_symbol='x',
                                                     marker=dict(size=14)))

            # Update layouts

            figure_raw.update_layout(showlegend=True,
                                     title_text="LIMS all data - raw: %s" % (
                                         str(i)),
                                     margin=dict(
                                         l=1, r=1, b=50, t=50, pad=1),
                                     xaxis_title="Date",
                                     yaxis_title="Measurement value (a.u.)")

            figure_no_zeros.update_layout(showlegend=True,
                                          title_text="LIMS all data - no zeros: %s" % (
                                              str(i)),
                                          margin=dict(
                                              l=1, r=1, b=50, t=50, pad=1),
                                          xaxis_title="Date",
                                          yaxis_title="Measurement value (a.u.)")

            figure_no_duplicates.update_layout(showlegend=True,
                                               title_text="LIMS all data - no zeros & no duplicates: %s" % (
                                                   str(i)),
                                               margin=dict(
                                                   l=1, r=1, b=50, t=50, pad=1),
                                               xaxis_title="Date",
                                               yaxis_title="Measurement value (a.u.)")

            figure_outliers.update_layout(title_text="LIMS: Outliers based on box plots: %s" % (str(i)),
                                          margin=dict(
                                              l=1, r=1, b=50, t=50, pad=1),
                                          xaxis_title="Date",
                                          yaxis_title="Measurement value (a.u.)")

            figure_boxplot.update_layout(showlegend=True,
                                         title_text="LIMS: Box plots for outliers removal: %s" % (
                                             str(i)),
                                         margin=dict(
                                             l=1, r=1, b=50, t=50, pad=1),
                                         xaxis_title="Property",
                                         yaxis_title="Measurement value (a.u.)")

            figure_raw.write_html(
                folder_visualisations_LIMS + '1. data - all raw - %s.html' % (i))
            figure_no_zeros.write_html(
                folder_visualisations_LIMS + '2. data - no zeros and 999 - %s.html' % (i))
            figure_no_duplicates.write_html(
                folder_visualisations_LIMS + '3. data - no zeros and 999 & no duplicates- %s.html' % (i))
            figure_outliers.write_html(
                folder_visualisations_LIMS + '4. outliers - %s.html' % (i))
            figure_boxplot.write_html(
                folder_visualisations_LIMS + '5. boxplots - %s.html' % (i))

            # Propagate in figures holders
            figures_no_zeros.append(figure_no_zeros)
            figures_boxplots.append(figure_boxplot)
            figures_outliers.append(figure_outliers)

            # Histogram
            figure_no_zeros_histogram_with_outliers = ff.create_distplot(
                [data[j] for j in data.drop('date', axis=1).columns[::-1]],
                data.drop(
                    'date', axis=1).columns[::-1],
                colors=px.colors.qualitative.Plotly[
                       :len(data.drop('date', axis=1).columns)][::-1])

            figure_no_zeros_histogram_with_outliers.update_layout(showlegend=True,
                                                                  title_text="LIMS histogram: %s" % (
                                                                      str(i)),
                                                                  margin=dict(
                                                                      l=1, r=1, b=50, t=50, pad=1),
                                                                  yaxis_title="Count of measurement value (a.u.)")

            figure_no_zeros_histogram_with_outliers.write_html(
                folder_visualisations_LIMS + '6. histogram with outliers - %s.html' % (i))

            figures_no_zeros_histogram_with_outliers.append(
                figure_no_zeros_histogram_with_outliers)

            # Correlation map

            figure_corr_map = ff.create_annotated_heatmap(z=np.array(results_holder_dict['corr_map']),
                                                          annotation_text=np.around(
                                                              np.array(
                                                                  results_holder_dict['corr_map']),
                                                              decimals=2),
                                                          x=list(
                                                              results_holder_dict['corr_map'].columns),
                                                          y=list(
                                                              results_holder_dict['corr_map'].columns),
                                                          colorscale='Jet',
                                                          showscale=True,
                                                          zmax=1,
                                                          zmin=-1,
                                                          zmid=0)

            figure_corr_map.update_layout(title_text="LIMS correlations map",
                                          margin=dict(l=1, r=1, b=50, t=50, pad=1))

            figure_corr_map.write_html(
                folder_visualisations_LIMS + '7. Correlations map.html')

            if data_holder_dict['data_riazi'] != 0:

                try:

                    figure_riazi = go.Figure()

                    for column in data_holder_dict['data_riazi'][i].drop('volume', axis=1).columns.to_list():
                        figure_riazi.add_trace(go.Scatter(x=data_holder_dict['data_riazi'][i]['volume'],
                                                          y=data_holder_dict['data_riazi'][i][column],
                                                          name=str(column)))

                        figure_riazi.add_trace(go.Scatter(x=volumes_dict[i].flatten(),
                                                          y=np.array(data_LIMS_separated_nozeros_noduplicates[i].drop(
                                                              ['date', 'time'], axis=1).iloc[np.where(
                                                              data_LIMS_separated_nozeros_noduplicates[i][
                                                                  'date'] == column)[0], :]).flatten(),
                                                          name=str(column),
                                                          mode='markers'))

                    button_all = dict(label='All',
                                      method='update',
                                      args=[{'visible': data_holder_dict['data_riazi'][i].drop('volume',
                                                                                               axis=1).columns.isin(
                                          data_holder_dict['data_riazi'][i].drop('volume', axis=1).columns),
                                          'title': 'All',
                                          'showlegend': True}])

                    def create_layout_button(column):
                        return dict(label=str(column),
                                    method='update',
                                    args=[{'visible': data_holder_dict['data_riazi'][i].drop('volume',
                                                                                             axis=1).columns.isin(
                                        [column]),
                                        'title': str(column),
                                        'showlegend': True}])

                    figure_riazi.update_layout(updatemenus=[go.layout.Updatemenu(active=0,
                                                                                 buttons=([button_all] * True) + list(
                                                                                     data_holder_dict['data_riazi'][
                                                                                         i].drop(
                                                                                         'volume', axis=1).columns.map(
                                                                                         lambda
                                                                                             column: create_layout_button(
                                                                                             column))))],
                                               margin=dict(
                                                   l=1, r=1, b=50, t=50, pad=1),
                                               title_text="Distillation curves obtained from Riazi model",
                                               yaxis_title="Measurement value (a.u.)",
                                               xaxis_title="Distillation volume fraction")

                    figure_riazi.write_html(
                        folder_visualisations_LIMS + '8. Riazi distillation curve - %s.html' % (i))

                    volumes_dict = dict()

                    volume_array = []

                    for volume in data_holder_dict['data_LIMS_separated_nozeros_noduplicates'][i].drop(['date', 'time'],
                                                                                                       axis=1).iloc[
                                  np.where(
                                      data_holder_dict['data_LIMS_separated_nozeros_noduplicates'][i]['date'] == i)[0],
                                  :].columns:
                        volume_array.append(
                            volume.split('_')[1].split('%')[0])

                    volumes_dict[i] = np.array(
                        volume_array, dtype=int) / 100

                    figure_riazi_deviations = go.Figure()

                    for position_volume, volume in enumerate(volumes_dict[i]):
                        figure_riazi_deviations.add_trace(
                            go.Box(y=results_holder_dict['deviations_riazi'][i].iloc[position_volume, :],
                                   name=str(i) + '_' + str(int(volume * 100)) + '%', boxmean='sd', boxpoints='all',
                                   jitter=0))

                        figure_riazi_deviations.update_layout(
                            title_text="Boxplots representing deviations between actual values of LIMS and those predicted with optimised Riazi model",
                            margin=dict(
                                l=1, r=1, b=50, t=50, pad=1),
                            yaxis_title="Deviation between actual values and those predicted with Riazi model (a.u.)",
                            xaxis_title="Property")

                    figure_riazi_deviations.write_html(
                        folder_visualisations_LIMS + '8. Riazi deviations - %s.html' % (i))

                    figure_riazi_parameters = make_subplots(rows=3,
                                                            cols=2,
                                                            horizontal_spacing=0.05,
                                                            specs=[[{}, {}],
                                                                   [{}, {}],
                                                                   [{"colspan": 2}, None]],
                                                            subplot_titles=(
                                                                "Parameter A", "Parameter B", "Parameter T0",
                                                                "Calculated T at 99 vol% (T99)",
                                                                "Coefficient of determination R2 for actual and predicted values")
                                                            )

                    figure_riazi_parameters.add_trace(go.Scatter(x=results_holder_dict['riazi_results_dict'][i]['date'],
                                                                 y=results_holder_dict['riazi_results_dict'][i][
                                                                     'popt1_riazi'],
                                                                 name=str(
                                                                     i),
                                                                 mode='markers',
                                                                 showlegend=False), row=1, col=1)

                    figure_riazi_parameters.add_trace(go.Scatter(x=results_holder_dict['riazi_results_dict'][i]['date'],
                                                                 y=results_holder_dict['riazi_results_dict'][i][
                                                                     'popt2_riazi'],
                                                                 name=str(
                                                                     i),
                                                                 mode='markers',
                                                                 showlegend=False), row=1, col=2)

                    figure_riazi_parameters.add_trace(go.Scatter(x=results_holder_dict['riazi_results_dict'][i]['date'],
                                                                 y=results_holder_dict['riazi_results_dict'][i][
                                                                     'popt3_riazi'],
                                                                 name=str(
                                                                     i),
                                                                 mode='markers',
                                                                 showlegend=False), row=2, col=1)

                    figure_riazi_parameters.add_trace(go.Scatter(x=results_holder_dict['riazi_results_dict'][i]['date'],
                                                                 y=results_holder_dict['riazi_results_dict'][i][
                                                                     'T99_riazi'],
                                                                 name=str(
                                                                     i),
                                                                 mode='markers',
                                                                 showlegend=False), row=2, col=2)

                    figure_riazi_parameters.add_trace(go.Scatter(x=results_holder_dict['riazi_results_dict'][i]['date'],
                                                                 y=results_holder_dict['riazi_results_dict'][i][
                                                                     'R2_riazi'],
                                                                 name=str(
                                                                     i),
                                                                 mode='markers',
                                                                 showlegend=False), row=3, col=1)

                    figure_riazi_parameters.update_yaxes(
                        title_text="Parameter A value", row=1, col=1)
                    figure_riazi_parameters.update_yaxes(
                        title_text="Parameter B value", row=1, col=2)
                    figure_riazi_parameters.update_yaxes(
                        title_text="Parameter T0 (oC)", row=2, col=1)
                    figure_riazi_parameters.update_yaxes(
                        title_text="Calculated T99 (oC)", row=2, col=2)
                    figure_riazi_parameters.update_yaxes(
                        title_text="R2", row=2, col=1)

                    figure_riazi_parameters.update_xaxes(
                        title_text="Date", row=1, col=1)
                    figure_riazi_parameters.update_xaxes(
                        title_text="Date", row=1, col=2)
                    figure_riazi_parameters.update_xaxes(
                        title_text="Date", row=2, col=1)
                    figure_riazi_parameters.update_xaxes(
                        title_text="Date", row=2, col=2)
                    figure_riazi_parameters.update_xaxes(
                        title_text="Date", row=2, col=1)

                    figure_riazi_parameters.write_html(
                        folder_visualisations_LIMS + '8. Riazi parameters - %s.html' % (i))

                except:
                    pass

            if data_holder_dict['data_ARIMA'] != 0:

                figure_ARIMA = make_subplots(rows=2,
                                             cols=1,
                                             vertical_spacing=0.05)

                for position_j, j in enumerate(data_holder_dict['data_ARIMA'][i].drop('date', axis=1).columns):
                    figure_ARIMA.add_trace(go.Scatter(x=data_holder_dict['data_ARIMA'][i]['date'].iloc[6:],
                                                      y=data_holder_dict['data_ARIMA'][i][j].iloc[6:],
                                                      showlegend=False,
                                                      marker_color=px.colors.qualitative.Plotly[position_j]), row=1,
                                           col=1)

                    figure_ARIMA.add_trace(
                        go.Scatter(x=data_holder_dict['data_LIMS_separated_nozeros_noduplicates'][i]['date'].iloc[6:],
                                   y=data_holder_dict['data_LIMS_separated_nozeros_noduplicates'][i][j].iloc[6:],
                                   mode='markers',
                                   showlegend=False,
                                   marker_color=px.colors.qualitative.Plotly[position_j]), row=1, col=1)

                    figure_ARIMA.add_trace(go.Box(y=data_holder_dict['data_ARIMA'][i][j].iloc[6:] -
                                                    data_holder_dict['data_LIMS_separated_nozeros_noduplicates'][i][
                                                        j].iloc[6:],
                                                  boxmean='sd',
                                                  boxpoints='all',
                                                  jitter=0), row=2, col=1)

                figure_ARIMA.write_html(
                    folder_visualisations_LIMS + '9. ARIMA - %s.html' % (i))

        print('Visualisations for LIMS: Completed')


def y_axis_name(tag):
    if tag.startswith('T') == True:
        return 'Temperature (oC)'

    if tag.startswith('F') == True:
        return 'Flow rate (T/h)'

    if tag.startswith('T') == True:
        return 'Temperature (oC)'


def sensors_visualisation(folder_visualisations, sensors_plot_decision, data_holder_dict, results_holder_dict):
    if bool(sensors_plot_decision) == False:
        print('Visualisations for sensors: Disabled')

    if bool(sensors_plot_decision) == True:
        print('Visualisations for sensors: Started')

        folder_visualisations_sensors = folder_visualisations + 'Sensors/'

        try:
            os.makedirs(folder_visualisations_sensors)

        except:
            pass

        # Plotting sensors - raw

        for position_j, j in enumerate(data_holder_dict['raw_sensors'].drop('date', axis=1).columns):

            sensors_raw = go.Figure()
            years = np.unique(
                pd.to_datetime(data_holder_dict['raw_sensors']['date']).dt.year)
            buttons = []
            years_id = dict()

            if np.min(data_holder_dict['raw_sensors'].drop('date', axis=1)[j]) < 0:
                y_min = 1.2 * \
                        np.min(data_holder_dict['raw_sensors'].drop(
                            'date', axis=1)[j])

            if np.min(data_holder_dict['raw_sensors'].drop('date', axis=1)[j]) == 0:
                y_min = - \
                    np.std(data_holder_dict['raw_sensors'].drop(
                        'date', axis=1)[j])

            if np.min(data_holder_dict['raw_sensors'].drop('date', axis=1)[j]) > 0:
                y_min = 0.8 * \
                        np.min(data_holder_dict['raw_sensors'].drop(
                            'date', axis=1)[j])

            y_max = 1.2 * \
                    np.max(data_holder_dict['raw_sensors'].drop(
                        'date', axis=1)[j])

            for position_i, i in enumerate(years):
                visible = [False] * len(years)
                visible[position_i] = True

                years_id[i] = np.where(
                    pd.to_datetime(data_holder_dict['raw_sensors']['date']).dt.year == i)[0]

                buttons.append(dict(label=str(i),
                                    method="update",
                                    args=[{"x": [data_holder_dict['raw_sensors']['date'].iloc[years_id[i]]],
                                           "y": [data_holder_dict['raw_sensors'].iloc[years_id[i], position_j]],
                                           'yaxis': {'title': y_axis_name(str(j))}}],
                                    ))

            sensors_raw.add_trace(
                go.Scatter(x=data_holder_dict['raw_sensors']['date'].iloc[years_id[list(years_id.keys())[0]]],
                           y=data_holder_dict['raw_sensors'].iloc[years_id[list(years_id.keys())[
                               0]], position_j],
                           visible=True,
                           name='Raw data'))

            menu = [dict(type='buttons',
                         active=0,
                         buttons=list(buttons))]

            sensors_raw.update_layout(yaxis_title=y_axis_name(str(j)),
                                      updatemenus=[dict(
                                          type="buttons",
                                          direction="right",
                                          x=0.7,
                                          y=1.2,
                                          showactive=True,
                                          buttons=list(buttons))])

            sensors_raw.update_yaxes(range=[y_min, y_max])

            sensors_raw.write_html(folder_visualisations_sensors + '{} - raw.html'.format(j))

        # Plotting sensors - no short outliers
        for position_j, j in enumerate(data_holder_dict['sensors_no_short_outliers'].drop('date', axis=1).columns):

            sensors_no_short_outliers = go.Figure()
            years = np.unique(
                pd.to_datetime(data_holder_dict['sensors_no_short_outliers']['date']).dt.year)
            buttons = []
            years_id = dict()

            if np.min(data_holder_dict['raw_sensors'].drop('date', axis=1)[j]) < 0:
                y_min = 1.2 * \
                        np.min(data_holder_dict['raw_sensors'].drop(
                            'date', axis=1)[j])

            if np.min(data_holder_dict['raw_sensors'].drop('date', axis=1)[j]) == 0:
                y_min = - \
                    np.std(data_holder_dict['raw_sensors'].drop(
                        'date', axis=1)[j])

            if np.min(data_holder_dict['raw_sensors'].drop('date', axis=1)[j]) > 0:
                y_min = 0.8 * \
                        np.min(data_holder_dict['raw_sensors'].drop(
                            'date', axis=1)[j])

            y_max = 1.2 * \
                    np.max(data_holder_dict['raw_sensors'].drop(
                        'date', axis=1)[j])

            for position_i, i in enumerate(years):
                years_id[i] = np.where(
                    pd.to_datetime(data_holder_dict['sensors_no_short_outliers']['date']).dt.year == i)[0]

                buttons.append(dict(label=str(i),
                                    method="update",
                                    args=[{"x": [data_holder_dict['raw_sensors']['date'].iloc[years_id[i]],
                                                 data_holder_dict['sensors_no_short_outliers']['date'].iloc[
                                                     years_id[i]],
                                                 data_holder_dict['sensors_smooth_data']['date'].iloc[years_id[i]]],
                                           "y": [data_holder_dict['raw_sensors'].iloc[years_id[i], position_j],
                                                 data_holder_dict['sensors_no_short_outliers'].iloc[
                                                     years_id[i], position_j],
                                                 data_holder_dict['sensors_smooth_data'].iloc[years_id[i], position_j]],
                                           'yaxis': {'title': y_axis_name(str(j))}}],
                                    ))

            sensors_no_short_outliers.add_trace(
                go.Scatter(x=data_holder_dict['raw_sensors']['date'].iloc[years_id[list(years_id.keys())[0]]],
                           y=data_holder_dict['raw_sensors'].iloc[years_id[list(years_id.keys())[
                               0]], position_j],
                           visible=True,
                           name='Raw data'))

            sensors_no_short_outliers.add_trace(
                go.Scatter(
                    x=data_holder_dict['sensors_no_short_outliers']['date'].iloc[years_id[list(years_id.keys())[0]]],
                    y=data_holder_dict['sensors_no_short_outliers'].iloc[years_id[list(years_id.keys())[
                        0]], position_j],
                    visible=True,
                    showlegend=True,
                    name='No short-term outliers'))

            sensors_no_short_outliers.add_trace(
                go.Scatter(x=data_holder_dict['sensors_smooth_data']['date'].iloc[years_id[list(years_id.keys())[0]]],
                           y=data_holder_dict['sensors_smooth_data'].iloc[years_id[list(years_id.keys())[
                               0]], position_j],
                           visible=True,
                           showlegend=True,
                           name='Steady states'))

            sensors_no_short_outliers.update_layout(yaxis_title=y_axis_name(str(j)),
                                                    updatemenus=[dict(
                                                        type="buttons",
                                                        direction="right",
                                                        x=0.7,
                                                        y=1.2,
                                                        showactive=True,
                                                        buttons=list(buttons))])

            sensors_no_short_outliers.update_yaxes(range=[y_min, y_max])

            sensors_no_short_outliers.write_html(
                folder_visualisations_sensors + '{} - no short term outliers.html'.format(j))

        # Plotting sensors - hotelling's statistics

        figure_hotteling = go.Figure()
        years = np.unique(
            pd.to_datetime(data_holder_dict['sensors_no_short_outliers']['date']).dt.year)
        buttons = []
        years_id = dict()

        for position_i, i in enumerate(years):
            years_id[i] = np.where(
                pd.to_datetime(data_holder_dict['sensors_no_short_outliers']['date']).dt.year == i)[0]

            buttons.append(dict(label=str(i),
                                method="update",
                                args=[{"x": [data_holder_dict['sensors_no_short_outliers']['date'][years_id[i]],
                                             [np.min(
                                                 data_holder_dict['sensors_no_short_outliers']['date'][years_id[i]]),
                                              np.max(
                                                  data_holder_dict['sensors_no_short_outliers']['date'][years_id[i]])]],
                                       "y": [results_holder_dict['PCA']['T2'][years_id[i]],
                                             [f.ppf(0.95, 2, 100000) * 2,
                                              f.ppf(0.95, 2, 100000) * 2]]}],
                                ))

        figure_hotteling.add_trace(
            go.Scatter(x=data_holder_dict['sensors_no_short_outliers']['date'][years_id[list(years_id.keys())[0]]],
                       y=results_holder_dict['PCA']['T2'][years_id[list(years_id.keys())[
                           0]]],
                       mode='lines',
                       name="Hotelling's T2 statistics"))

        figure_hotteling.add_trace(
            go.Scatter(
                x=[np.min(data_holder_dict['sensors_no_short_outliers']['date'][years_id[list(years_id.keys())[0]]]),
                   np.max(data_holder_dict['sensors_no_short_outliers']['date'][years_id[list(years_id.keys())[0]]])],
                y=[f.ppf(0.95, 2, 100000) * 2,
                   f.ppf(0.95, 2, 100000) * 2],
                mode='lines',
                name="Upper confidence level"))

        menu = [dict(type='buttons',
                     active=0,
                     buttons=list(buttons))]

        figure_hotteling.update_layout(yaxis_title="T2 statistics",
                                       updatemenus=[dict(type="buttons",
                                                         direction="right",
                                                         x=0.7,
                                                         y=1.2,
                                                         showactive=True,
                                                         buttons=list(buttons))])

        figure_hotteling.write_html(folder_visualisations_sensors + 'hottelings_statistics.html')

        # data sensors no outliers at all

        for position_j, j in enumerate(data_holder_dict['sensors_no_outliers'].drop('date', axis=1).columns):

            sensors_no_outliers = go.Figure()
            years = np.unique(
                pd.to_datetime(data_holder_dict['sensors_no_outliers']['date']).dt.year)
            buttons = []
            years_id = dict()

            if np.min(data_holder_dict['raw_sensors'].drop('date', axis=1)[j]) < 0:
                y_min = 1.2 * \
                        np.min(data_holder_dict['raw_sensors'].drop(
                            'date', axis=1)[j])

            if np.min(data_holder_dict['raw_sensors'].drop('date', axis=1)[j]) == 0:
                y_min = - \
                    np.std(data_holder_dict['raw_sensors'].drop(
                        'date', axis=1)[j])

            if np.min(data_holder_dict['raw_sensors'].drop('date', axis=1)[j]) > 0:
                y_min = 0.8 * \
                        np.min(data_holder_dict['raw_sensors'].drop(
                            'date', axis=1)[j])

            y_max = 1.2 * \
                    np.max(data_holder_dict['raw_sensors'].drop(
                        'date', axis=1)[j])

            for position_i, i in enumerate(years):
                years_id[i] = np.where(
                    pd.to_datetime(data_holder_dict['sensors_no_outliers']['date']).dt.year == i)[0]

                buttons.append(dict(label=str(i),
                                    method="update",
                                    args=[{"x": [data_holder_dict['raw_sensors']['date'].iloc[years_id[i]],
                                                 data_holder_dict['sensors_no_outliers']['date'].iloc[years_id[i]]],
                                           "y": [data_holder_dict['raw_sensors'].iloc[years_id[i], position_j],
                                                 data_holder_dict['sensors_no_outliers'].iloc[
                                                     years_id[i], position_j]]}],
                                    ))

        sensors_no_outliers.add_trace(
            go.Scatter(x=data_holder_dict['raw_sensors']['date'].iloc[years_id[list(years_id.keys())[0]]],
                       y=data_holder_dict['raw_sensors'].iloc[years_id[list(years_id.keys())[
                           0]], position_j],
                       visible=True,
                       name='Raw data'))

        sensors_no_outliers.add_trace(
            go.Scatter(x=data_holder_dict['sensors_no_outliers']['date'].iloc[years_id[list(years_id.keys())[0]]],
                       y=data_holder_dict['sensors_no_outliers'].iloc[
                           years_id[list(years_id.keys())[0]], position_j],
                       visible=True,
                       showlegend=True,
                       name='No outliers'))

        sensors_no_outliers.update_layout(updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.2,
            showactive=True,
            buttons=list(buttons))])

        sensors_no_outliers.update_yaxes(range=[y_min, y_max])

        sensors_no_outliers.write_html(folder_visualisations_sensors + '{} - no outliers.html'.format(j))

        # --- correlations map

        figure_corr_map = ff.create_annotated_heatmap(z=np.array(results_holder_dict['corr_map_sensors']),
                                                      annotation_text=np.around(
                                                          np.array(results_holder_dict['corr_map_sensors']),
                                                          decimals=2),
                                                      x=list(results_holder_dict['corr_map_sensors'].columns),
                                                      y=list(results_holder_dict['corr_map_sensors'].columns),
                                                      colorscale='Jet',
                                                      showscale=True,
                                                      zmax=1,
                                                      zmin=-1,
                                                      zmid=0)

        figure_corr_map.update_layout(title_text="Sensors correlations map",
                                      margin=dict(l=1, r=1, b=50, t=50, pad=1))

        figure_corr_map.write_html(folder_visualisations_sensors + 'Correlations map.html')

        print('Visualisations for sensors: Completed')


def perform_auto_search(value_riazi, value_search, folder_results, main_decision, write_sensors_csv_files,
                        period_sensors, soft_sensors_plot_decision, LIMS_plot_decision, sensors_plot_decision, align_in_time):
    data_holder_dict = dict()
    results_holder_dict = dict()

    data_LIMS_separated = None
    data_LIMS_separated_nozeros = None
    data_LIMS_separated_nozeros_noduplicates = None
    data_no_outliers = None
    data_sensors = None
    data_smooth = None
    data_no_short_outliers = None
    outliers = None
    limits = None
    models_knn = None
    corr_map = None
    corr_df = None
    PCA_results = None

    folder_results_excel = folder_results + '/Excel output files/'
    folder_data = folder_results + '/Data files/'
    folder_visualisations = folder_results + '/Visualisations/'

    try:
        os.makedirs(folder_results_excel)
        os.makedirs(folder_data)
        os.makedirs(folder_visualisations)

    except:
        pass
    
    if align_in_time == 'align':
        print('do alignment')
        
    if align_in_time == 'load_aligned':
        print('load aligned')

    if main_decision == 'full':

        folder_models_soft_sensors = folder_results + '/Models/'

        try:
            os.makedirs(folder_models_soft_sensors)

        except:
            pass

        print('Full analysis')
        data_holder_dict_LIMS, results_holder_dict_LIMS = LIMS_analyse(folder_results_excel, value_riazi, folder_data)

        for i in data_holder_dict_LIMS.keys():
            data_holder_dict[i] = data_holder_dict_LIMS[i]

        for i in results_holder_dict_LIMS.keys():
            results_holder_dict[i] = results_holder_dict_LIMS[i]

        data_holder_dict_LIMS = None
        results_holder_dict_LIMS = None

        data_holder_dict_sensors, results_holder_dict_sensors = sensors_analyse(folder_results_excel, folder_data,
                                                                                data_holder_dict['data_date'],
                                                                                write_sensors_csv_files, None)

        for i in data_holder_dict_sensors.keys():
            data_holder_dict[i] = data_holder_dict_sensors[i]

        for i in results_holder_dict_sensors.keys():
            results_holder_dict[i] = results_holder_dict_sensors[i]

        data_holder_dict_sensors = None
        results_holder_dict_sensors = None

        data_holder_dict_soft_sensors, results_holder_dict_soft_sensors = soft_sensors_analyse(
            results_holder_dict['priority_key'], data_holder_dict['data_no_outliers'],
            data_holder_dict['sensors_pre_selected'], results_holder_dict, value_search, folder_results_excel,
            folder_models_soft_sensors, results_holder_dict['description'], align_in_time)

        for i in data_holder_dict_soft_sensors.keys():
            data_holder_dict[i] = data_holder_dict_soft_sensors[i]

        for i in results_holder_dict_soft_sensors.keys():
            results_holder_dict[i] = results_holder_dict_soft_sensors[i]

        data_holder_dict_soft_sensors = None
        results_holder_dict_soft_sensors = None

        soft_sensors_visualisation(folder_visualisations, soft_sensors_plot_decision, results_holder_dict,
                                   data_holder_dict, value_search, folder_models_soft_sensors)

        LIMS_visualisation(LIMS_plot_decision, data_holder_dict, results_holder_dict, folder_visualisations)

        sensors_visualisation(folder_visualisations, sensors_plot_decision, data_holder_dict, results_holder_dict)

    if main_decision == 'LIMS':
        print('LIMS analysis')

        data_holder_dict_LIMS, results_holder_dict_LIMS = LIMS_analyse(folder_results_excel, value_riazi, folder_data)

        for i in data_holder_dict_LIMS.keys():
            data_holder_dict[i] = data_holder_dict_LIMS[i]

        for i in results_holder_dict_LIMS.keys():
            results_holder_dict[i] = results_holder_dict_LIMS[i]

        data_holder_dict_LIMS = None
        results_holder_dict_LIMS = None

        LIMS_visualisation(LIMS_plot_decision, data_holder_dict, results_holder_dict, folder_visualisations)

    if main_decision == 'sensors':
        print('Sensors analysis')

        data_holder_dict_sensors, results_holder_dict_sensors = sensors_analyse(folder_results_excel, folder_data,
                                                                                np.array([None]),
                                                                                write_sensors_csv_files, period_sensors)

        for i in data_holder_dict_sensors.keys():
            data_holder_dict[i] = data_holder_dict_sensors[i]

        for i in results_holder_dict_sensors.keys():
            results_holder_dict[i] = results_holder_dict_sensors[i]

        data_holder_dict_sensors = None
        results_holder_dict_sensors = None

        sensors_visualisation(folder_visualisations, sensors_plot_decision, data_holder_dict, results_holder_dict)

    if main_decision == 'soft_sensors':
        print('Soft sensors analysis')

        folder_models_soft_sensors = folder_results + '/Models/'

        try:
            os.makedirs(folder_models_soft_sensors)

        except:
            pass

        data_holder_dict_soft_sensors, results_holder_dict_soft_sensors = soft_sensors_analyse(np.array([None]),
                                                                                               np.array([None]),
                                                                                               np.array([None]),
                                                                                               np.array([None]),
                                                                                               value_search,
                                                                                               folder_results_excel,
                                                                                               folder_models_soft_sensors,
                                                                                               [None],
                                                                                              align_in_time)

        for i in data_holder_dict_soft_sensors.keys():
            data_holder_dict[i] = data_holder_dict_soft_sensors[i]

        for i in results_holder_dict_soft_sensors.keys():
            results_holder_dict[i] = results_holder_dict_soft_sensors[i]

        data_holder_dict_soft_sensors = None
        results_holder_dict_soft_sensors = None

        soft_sensors_visualisation(folder_visualisations, soft_sensors_plot_decision, results_holder_dict,
                                   data_holder_dict, value_search, folder_models_soft_sensors)

    return data_holder_dict, results_holder_dict


# Initiate Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

app.config.suppress_callback_exceptions = True

# App layout
app.layout = html.Div([html.H1('Self-learning Models'),  # Title
                       dcc.Tabs(id='main_tabs',
                                value='Fully_automatic_search',
                                children=[dcc.Tab(id='sub_tab_fully_automatic',  # LIMS: Analytics tab
                                                  label='Fully automated optimisation',
                                                  value='Fully_automatic_search'),
                                          dcc.Tab(id='sub_tab_analytical_LIMS',
                                                  label='Semi-manual soft sensors optimisation',
                                                  value='Analytics_LIMS')
                                          ]
                                )
                       ])


# --- Main tabs: LIMS
@app.callback([Output('sub_tab_fully_automatic', 'children'),
               Output('sub_tab_analytical_LIMS', 'children')],
              [Input('main_tabs', 'value')])
def update_sub_tabs(value):
    load_the_data_info = 'You need to provide a folder path to where your input files are place. \n ' \
                         'When you click on the button "Select a directory with data", a new window will pop up' \
                         ' - You will need to go to the folder where your input files are located.' \
                         ' More information on how to prepare your input files can be found in the attached manual, as well as example files' \
                         ' If the folder is successfully loaded - the Status should change appropriately.' \
                         'In the folder you need the following files: 1) labels.csv; 2) LIMS.csv; 3) h5 files with sensors data'

    choose_main_option_info = 'Please choose what you want this software to do. You have 3 options:' \
                              'Full analysis (LIMS, sensors and soft sensors), LIMS only, Sensors only and Soft Sensors only' \
                              'If you want to rely fully on the automation, choose option 1.' \
                              'Option LIMS only will only process LIMS: it will produce excel output file with visualisations.' \
                              'You can then use these eexcel output files for other processing (for example in Soft sensors only). It gives your more flexibility.' \
                              'The same can be done with sensors only. It will process sensors and produce excel output files, files with processed data and visualisations' \
                              'If you already have processed the data (both LIMS and sensors), formatted them appropriately, you can just perform' \
                              ' Soft sensor only option: it will take input and output (X and y), and fit artificial neural network models as required.'

    if value == 'Fully_automatic_search':
        content = [html.Div([html.Hr(),
                             html.Div([html.Div([html.H2('Step 1: Load the data')], style={'display': 'inline-block'}),
                                       html.Div([dbc.Button('i', id="button_step1",
                                                            size="sm",
                                                            style={'padding': '1px 8px',
                                                                   'border-radius': '100%',
                                                                   'font-size': '12px',
                                                                   'font-style': 'italic',
                                                                   'background-color': 'white',
                                                                   'font-color': '#696969',
                                                                   'border': '2px dark-grey',
                                                                   'margin-bottom': '60%',
                                                                   'color': '#696969',
                                                                   'white-space': 'pre-line'})],
                                                style={'display': 'inline-block'}),
                                       dbc.Modal([dbc.ModalHeader("Step 1: Load the data"),
                                                  dbc.ModalBody(load_the_data_info),
                                                  dbc.ModalFooter(
                                                      dbc.Button("Close", id="close_step1", className="ml-auto"))],
                                                 id="modal_step1")]),
                             html.Div([html.Div([dbc.Button('Select a directory with data', id='select_directory')]),
                                       html.Div([html.H1('Load the data', id='select_directory_output')],
                                                )]),
                             ], style={'padding-left': '15px'}),
                   html.Div([html.Hr(),
                             html.Div([html.H2('Step 2: What do you want to do?', style={'display': 'inline-block'}),
                                       html.Div([dbc.Button('i', id="button_step2",
                                                            size="sm",
                                                            style={'padding': '1px 8px',
                                                                   'border-radius': '100%',
                                                                   'font-size': '12px',
                                                                   'font-style': 'italic',
                                                                   'background-color': 'white',
                                                                   'font-color': '#696969',
                                                                   'border': '2px dark-grey',
                                                                   'margin-bottom': '60%',
                                                                   'color': '#696969'})],
                                                style={'display': 'inline-block'}),
                                       dbc.Modal([dbc.ModalHeader("Step 2: What do you wanna do?"),
                                                  dbc.ModalBody(choose_main_option_info),
                                                  dbc.ModalFooter(
                                                      dbc.Button("Close", id="close_step2", className="ml-auto"))],
                                                 id="modal_step2")]),
                             dcc.RadioItems(id='choose_main_option', options=[
                                 {'label': 'Full analysis - LIMS, sensors and soft sensors', 'value': 'full'},
                                 {'label': 'LIMS only', 'value': 'LIMS'},
                                 {'label': 'Sensors only', 'value': 'sensors'},
                                 {'label': 'Soft sensors only', 'value': 'soft_sensor'}],
                                            value='full',
                                            labelStyle={'display': 'block'})
                             ], style={'padding-left': '15px'}),
                   html.Div([html.Hr(),
                             html.Div(id='further_options', children=[])]),
                   html.Div([html.Hr(),
                             html.H2('Step 4: Start analysis')]),
                   html.Div([dbc.Button('Start fully automatic search', id='button_start_automatic')]),
                   html.Div(id='container'),
                   html.Hr(),
                   html.Div(id='results_output_LIMS',
                            children=[html.H1('Results of the analysis')])]

        return content, None

    if value == 'Analytics_LIMS':
        content = [html.Div(html.H1(
            'Nothing to see here yet: Work to be commenced in August-September 2021'))]

        return None, content


@app.callback(Output("further_options", "children"),
              Input("choose_main_option", "value"))
def main_option(value):
    if value == 'full':
        title_1 = 'Chosen option: Full analysis (LIMS, sensors and soft sensors).'
        title_2 = 'Step 3: Choose other functionalities:'
        body_information = 'You also need to choose functionalities for the full analysis.' \
                           'Riazi modelling: if your data does not contain distillation curves, select No - skip Riazi' \
                           '. Otherwise, it will produce an error whilst trying to process the data. ' \
                           'Visualisations can take some space on your disc - so if you dont need them - choose to skip them.' \
                           'It is set as default to not produce visualisations. ' \
                           'NOTE: Visualisation for sensors can take a very long time to be produced (especially if there are lots of sensors)' \
                           ' and they can be quite large files. Only create sensors visualisations if you have to.' \
                           'If you want to have a look at the data produced from sensors analysis - they can be saved as csv files ' \
                           '(for every step of processing of outliers) - they can be large files so create them if needed.' \
                           'The last option is to choose quick or full training of ANN models. Quick search: ' \
                           'Number of nodes = number of sensors (round to the closest 10), number of hidden layers = 2.' \
                           'Full search: number of nodes = from 5 to the number of sensors (round to the closest 10) with interval of 5, ' \
                           'number of hidden layers = 1, 2, 3.'

        content = html.Div([html.H4('Do you want to include Riazi modelling'),
                            dcc.RadioItems(id='riazi_radio_choice',
                                           options=[{'label': 'Yes - include Riazi', 'value': 1},
                                                    {'label': 'No - skip Riazi', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}),
                            html.H4('Create LIMS visualisations?'),
                            dcc.RadioItems(id='LIMS_plot_decision',
                                           options=[{'label': 'Yes - create visualisation', 'value': 1},
                                                    {'label': 'No - skip visualisation', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}),
                            html.H4('Create sensors visualisations?'),
                            dcc.RadioItems(id='sensors_plot_decision',
                                           options=[{'label': 'Yes - create visualisation', 'value': 1},
                                                    {'label': 'No - skip visualisation', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}),
                            html.H4('Create soft sensors visualisations?'),
                            dcc.RadioItems(id='soft_sensors_plot_decision',
                                           options=[{'label': 'Yes - create visualisation', 'value': 1},
                                                    {'label': 'No - skip visualisation', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}),
                            html.H4('Do you want to write processed sensors data to .csv files?'),
                            dcc.RadioItems(id='write_sensors_csv_files',
                                           options=[{'label': 'Yes - write them to .csv files', 'value': 1},
                                                    {'label': 'No - do not write them to .csv files', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}),
                            html.H4('What is the period to be taken for sensors', style={'display': 'none'}),
                            html.Div([dcc.Slider(id='period_sensors',
                                                 min=1,
                                                 max=60,
                                                 step=1,
                                                 value=36, marks={i: str(i) for i in range(-6, 61, 6)})],
                                     style={'display': 'none',
                                            'width': '50%'}),
                            html.H4('Quick or full model search?'),
                            dcc.RadioItems(id='search_radio_choice',
                                           options=[{'label': 'Quick search', 'value': 0},
                                                    {'label': 'Full search', 'value': 1}],
                                           value=0, labelStyle={'display': 'block'}),
                            dcc.RadioItems(id='main_decision',
                                           options=[{'label': 'Full', 'value': 'full'},
                                                    {'label': 'LIMS', 'value': 'LIMS'},
                                                    {'label': 'Sensors', 'value': 'sensors'},
                                                    {'label': 'Soft sensors', 'value': 'Soft sensors'}],
                                           value='full',
                                           labelStyle={'display': 'block'}, style={'display': 'none'}),
                            html.H4('Do you want the software to align sensors with LIMS or load already aligned data?', style={'display': 'none'}),
                           dcc.RadioItems(id='align_in_time',
                                           options=[{'label': 'Align in time', 'value': 'align'},
                                                    {'label': 'Load aligned data (X & y)', 'value': 'load_aligned'}],
                                           value='align',
                                           labelStyle={'display': 'block'}, style={'display': 'none'})])

        return [html.Div([html.H2(title_1), html.Div(
            [html.H2(title_2, style={'display': 'inline-block'}), html.Div([dbc.Button('i', id="button_step3_full",
                                                                                       size="sm",
                                                                                       style={'padding': '1px 8px',
                                                                                              'border-radius': '100%',
                                                                                              'font-size': '12px',
                                                                                              'font-style': 'italic',
                                                                                              'background-color': 'white',
                                                                                              'font-color': '#696969',
                                                                                              'border': '2px dark-grey',
                                                                                              'margin-bottom': '60%',
                                                                                              'color': '#696969'})],
                                                                           style={'display': 'inline-block'}),
             dbc.Modal([dbc.ModalHeader("Step 3: Choose other functionalities"),
                        dbc.ModalBody(body_information),
                        dbc.ModalFooter(dbc.Button("Close", id="close_step3_full", className="ml-auto"))],
                       id="modal_step3_full")]), content], style={'padding-left': '15px'})]

    if value == 'LIMS':
        title_1 = 'Chosen option: LIMS only'
        title_2 = 'Step 3: Choose other functionalities:'
        body_information = 'You also need to choose functionalities for the full analysis.' \
                           'Riazi modelling: if your data does not contain distillation curves, select No - skip Riazi' \
                           '. Otherwise, it will produce an error whilst trying to process the data. ' \
                           'Visualisations can take some space on your disc - so if you dont need them - choose to skip them.' \
                           'It is set as default to not produce visualisations. '

        content = html.Div([html.H4('Do you want to include Riazi modelling'),
                            dcc.RadioItems(id='riazi_radio_choice',
                                           options=[{'label': 'Yes - include Riazi', 'value': 1},
                                                    {'label': 'No - skip Riazi', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}),
                            html.H4('Create LIMS visualisations?'),
                            dcc.RadioItems(id='LIMS_plot_decision',
                                           options=[{'label': 'Yes - create visualisation', 'value': 1},
                                                    {'label': 'No - skip visualisation', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}),
                            html.H4('Create sensors visualisations?', style={'display': 'none'}),
                            dcc.RadioItems(id='sensors_plot_decision',
                                           options=[{'label': 'Yes - create visualisation', 'value': 1},
                                                    {'label': 'No - skip visualisation', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'},
                                           style={'display': 'none'}),
                            html.H4('Create soft sensors visualisations?', style={'display': 'none'}),
                            dcc.RadioItems(id='soft_sensors_plot_decision',
                                           options=[{'label': 'Yes - create visualisation', 'value': 1},
                                                    {'label': 'No - skip visualisation', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}, style={'display': 'none'}),
                            html.H4('Do you want to write processed sensors data to .csv files?',
                                    style={'display': 'none'}),
                            dcc.RadioItems(id='write_sensors_csv_files',
                                           options=[{'label': 'Yes - write them to .csv files', 'value': 1},
                                                    {'label': 'No - do not write them to .csv files', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}, style={'display': 'none'}),
                            html.H4('Quick or full model search?', style={'display': 'none'}),
                            dcc.RadioItems(id='search_radio_choice',
                                           options=[{'label': 'Quick search', 'value': 0},
                                                    {'label': 'Full search', 'value': 1}],
                                           value=0, labelStyle={'display': 'block'}, style={'display': 'none'}),
                            html.H4('What is the period to be taken for sensors', style={'display': 'none'}),
                            html.Div([dcc.Slider(id='period_sensors',
                                                 min=1,
                                                 max=60,
                                                 step=1,
                                                 value=36, marks={i: str(i) for i in range(-6, 61, 6)})],
                                     style={'display': 'none',
                                            'width': '50%'}),
                            dcc.RadioItems(id='main_decision',
                                           options=[{'label': 'Full', 'value': 'full'},
                                                    {'label': 'LIMS', 'value': 'LIMS'},
                                                    {'label': 'Sensors', 'value': 'sensors'},
                                                    {'label': 'Soft sensors', 'value': 'soft_sensors'}],
                                           value='LIMS',
                                           labelStyle={'display': 'block'}, style={'display': 'none'}),
                            html.H4('Do you want the software to align sensors with LIMS or load already aligned data?', style={'display': 'none'}),
                           dcc.RadioItems(id='align_in_time',
                                           options=[{'label': 'Align in time', 'value': 'align'},
                                                    {'label': 'Load aligned data (X & y)', 'value': 'load_aligned'}],
                                           value='align',
                                           labelStyle={'display': 'block'}, style={'display': 'none'})])

        return [html.Div([html.H2(title_1), html.Div(
            [html.H2(title_2, style={'display': 'inline-block'}), html.Div([dbc.Button('i', id="button_step3_LIMS",
                                                                                       size="sm",
                                                                                       style={'padding': '1px 8px',
                                                                                              'border-radius': '100%',
                                                                                              'font-size': '12px',
                                                                                              'font-style': 'italic',
                                                                                              'background-color': 'white',
                                                                                              'font-color': '#696969',
                                                                                              'border': '2px dark-grey',
                                                                                              'margin-bottom': '60%',
                                                                                              'color': '#696969'})],
                                                                           style={'display': 'inline-block'}),
             dbc.Modal([dbc.ModalHeader("Step 3: Choose other functionalities"),
                        dbc.ModalBody(body_information),
                        dbc.ModalFooter(dbc.Button("Close", id="close_step3_LIMS", className="ml-auto"))],
                       id="modal_step3_LIMS")]), content], style={'padding-left': '15px'})]

    if value == 'sensors':
        title_1 = 'Chosen option: sensors only'
        title_2 = 'Step 3: Choose other functionalities:'
        body_information = 'You also need to choose functionalities for the full analysis.' \
                           'Visualisations can take some space on your disc - so if you dont need them - choose to skip them.' \
                           'It is set as default to not produce visualisations. ' \
                           'NOTE: Visualisation for sensors can take a very long time to be produced (especially if there are lots of sensors)' \
                           ' and they can be quite large files. Only create sensors visualisations if you have to.' \
                           'If you want to have a look at the data produced from sensors analysis - they can be saved as csv files ' \
                           '(for every step of processing of outliers) - they can be large files so create them if needed.' \
                           'The last option is to choose the period for sensors. In full analysis, the first step is LIMS which also ' \
                           'gives information about the earliest and the latest dates for LIMS. These are used to inform what are the ' \
                           'start and end point in time for sensors. Thus, for sensors only option, the user need to narrow down the ' \
                           'period taken: start point is from the start in .h5 files, while end is start + period specified in the option.' \
                           'NOTE: more than 36 months can result in your computer crashing: too much data loaded at once.'

        content = html.Div([html.H4('Do you want to include Riazi modelling', style={'display': 'none'}),
                            dcc.RadioItems(id='riazi_radio_choice',
                                           options=[{'label': 'Yes - include Riazi', 'value': 1},
                                                    {'label': 'No - skip Riazi', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}, style={'display': 'none'}),
                            html.H4('Create LIMS visualisations?', style={'display': 'none'}),
                            dcc.RadioItems(id='LIMS_plot_decision',
                                           options=[{'label': 'Yes - create visualisation', 'value': 1},
                                                    {'label': 'No - skip visualisation', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}, style={'display': 'none'}),
                            html.H4('Create sensors visualisations?'),
                            dcc.RadioItems(id='sensors_plot_decision',
                                           options=[{'label': 'Yes - create visualisation', 'value': 1},
                                                    {'label': 'No - skip visualisation', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}),
                            html.H4('Create soft sensors visualisations?', style={'display': 'none'}),
                            dcc.RadioItems(id='soft_sensors_plot_decision',
                                           options=[{'label': 'Yes - create visualisation', 'value': 1},
                                                    {'label': 'No - skip visualisation', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}, style={'display': 'none'}),
                            html.H4('Do you want to write processed sensors data to .csv files?'),
                            dcc.RadioItems(id='write_sensors_csv_files',
                                           options=[{'label': 'Yes - write them to .csv files', 'value': 1},
                                                    {'label': 'No - do not write them to .csv files', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}),
                            html.H4('Quick or full model search?', style={'display': 'none'}),
                            dcc.RadioItems(id='search_radio_choice',
                                           options=[{'label': 'Quick search', 'value': 0},
                                                    {'label': 'Full search', 'value': 1}],
                                           value=0, labelStyle={'display': 'block'}, style={'display': 'none'}),
                            html.H4('What is the period to be taken for sensors'),
                            html.Div([dcc.Slider(id='period_sensors',
                                                 min=1,
                                                 max=60,
                                                 step=1,
                                                 value=36, marks={i: str(i) for i in range(-6, 61, 6)})],
                                     style={'display': 'block',
                                            'width': '50%'}),
                            dcc.RadioItems(id='main_decision',
                                           options=[{'label': 'Full', 'value': 'full'},
                                                    {'label': 'LIMS', 'value': 'LIMS'},
                                                    {'label': 'Sensors', 'value': 'sensors'},
                                                    {'label': 'Soft sensors', 'value': 'soft_sensors'}],
                                           value='sensors',
                                           labelStyle={'display': 'block'}, style={'display': 'none'}),
                            html.H4('Do you want the software to align sensors with LIMS or load already aligned data?', style={'display': 'none'}),
                           dcc.RadioItems(id='align_in_time',
                                           options=[{'label': 'Align in time', 'value': 'align'},
                                                    {'label': 'Load aligned data (X & y)', 'value': 'load_aligned'}],
                                           value='align',
                                           labelStyle={'display': 'block'}, style={'display': 'none'})])

        return [html.Div([html.H2(title_1), html.Div([html.H2(title_2, style={'display': 'inline-block'}),
                                                      html.Div([dbc.Button('i', id="button_step3_softsensors",
                                                                           size="sm",
                                                                           style={'padding': '1px 8px',
                                                                                  'border-radius': '100%',
                                                                                  'font-size': '12px',
                                                                                  'font-style': 'italic',
                                                                                  'background-color': 'white',
                                                                                  'font-color': '#696969',
                                                                                  'border': '2px dark-grey',
                                                                                  'margin-bottom': '60%',
                                                                                  'color': '#696969'})],
                                                               style={'display': 'inline-block'}),
                                                      dbc.Modal(
                                                          [dbc.ModalHeader("Step 3: Choose other functionalities"),
                                                           dbc.ModalBody(body_information),
                                                           dbc.ModalFooter(
                                                               dbc.Button("Close", id="close_step3_softsensors",
                                                                          className="ml-auto"))],
                                                          id="modal_step3_softsensors")]), content],
                         style={'padding-left': '15px'})]

    if value == 'soft_sensor':
        title_1 = 'Chosen option: soft sensors only'
        title_2 = 'Step 3: Choose other functionalities:'
        body_information = 'You also need to choose functionalities for the full analysis.' \
                           'This options will only perform optimisation of ANN (either quick or full, as described below).' \
                           'NOTE: Details what the format of input files must be can be found in manual. Quick search: ' \
                           'Number of nodes = number of sensors (round to the closest 10), number of hidden layers = 2.' \
                           'Full search: number of nodes = from 5 to the number of sensors (round to the closest 10) with interval of 5, ' \
                           'number of hidden layers = 1, 2, 3.'

        content = html.Div([html.H4('Do you want to include Riazi modelling', style={'display': 'none'}),
                            dcc.RadioItems(id='riazi_radio_choice',
                                           options=[{'label': 'Yes - include Riazi', 'value': 1},
                                                    {'label': 'No - skip Riazi', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}, style={'display': 'none'}),
                            html.H4('Create LIMS visualisations?', style={'display': 'none'}),
                            dcc.RadioItems(id='LIMS_plot_decision',
                                           options=[{'label': 'Yes - create visualisation', 'value': 1},
                                                    {'label': 'No - skip visualisation', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}, style={'display': 'none'}),
                            html.H4('Create sensors visualisations?', style={'display': 'none'}),
                            dcc.RadioItems(id='sensors_plot_decision',
                                           options=[{'label': 'Yes - create visualisation', 'value': 1},
                                                    {'label': 'No - skip visualisation', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}, style={'display': 'none'}),
                            html.H4('Create soft sensors visualisations?'),
                            dcc.RadioItems(id='soft_sensors_plot_decision',
                                           options=[{'label': 'Yes - create visualisation', 'value': 1},
                                                    {'label': 'No - skip visualisation', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}),
                            html.H4('Do you want to write processed sensors data to .csv files?',
                                    style={'display': 'none'}),
                            dcc.RadioItems(id='write_sensors_csv_files',
                                           options=[{'label': 'Yes - write them to .csv files', 'value': 1},
                                                    {'label': 'No - do not write them to .csv files', 'value': 0}],
                                           value=0, labelStyle={'display': 'block'}, style={'display': 'none'}),
                            html.H4('Quick or full model search?'),
                            dcc.RadioItems(id='search_radio_choice',
                                           options=[{'label': 'Quick search', 'value': 0},
                                                    {'label': 'Full search', 'value': 1}],
                                           value=0, labelStyle={'display': 'block'}),
                            html.H4('What is the period to be taken for sensors', style={'display': 'none'}),
                            html.Div([dcc.Slider(id='period_sensors',
                                                 min=1,
                                                 max=60,
                                                 step=1,
                                                 value=36, marks={i: str(i) for i in range(-6, 61, 6)})],
                                     style={'display': 'none',
                                            'width': '50%'}),
                            dcc.RadioItems(id='main_decision',
                                           options=[{'label': 'Full', 'value': 'full'},
                                                    {'label': 'LIMS', 'value': 'LIMS'},
                                                    {'label': 'Sensors', 'value': 'sensors'},
                                                    {'label': 'Soft sensors', 'value': 'soft_sensors'}],
                                           value='soft_sensors',
                                           labelStyle={'display': 'block'}, style={'display': 'none'}),
                            html.H4('Do you want the software to align sensors with LIMS or load already aligned data?'),
                           dcc.RadioItems(id='align_in_time',
                                           options=[{'label': 'Align in time', 'value': 'align'},
                                                    {'label': 'Load aligned data (X & y)', 'value': 'load_aligned'}],
                                           value='align',
                                           labelStyle={'display': 'block'})])

        return [html.Div([html.H2(title_1), html.Div(
            [html.H2(title_2, style={'display': 'inline-block'}), html.Div([dbc.Button('i', id="button_step3_sensors",
                                                                                       size="sm",
                                                                                       style={'padding': '1px 8px',
                                                                                              'border-radius': '100%',
                                                                                              'font-size': '12px',
                                                                                              'font-style': 'italic',
                                                                                              'background-color': 'white',
                                                                                              'font-color': '#696969',
                                                                                              'border': '2px dark-grey',
                                                                                              'margin-bottom': '60%',
                                                                                              'color': '#696969'})],
                                                                           style={'display': 'inline-block'}),
             dbc.Modal([dbc.ModalHeader("Step 3: Choose other functionalities"),
                        dbc.ModalBody(body_information),
                        dbc.ModalFooter(dbc.Button("Close", id="close_step3_sensors", className="ml-auto"))],
                       id="modal_step3_sensors")]), content], style={'padding-left': '15px'})]


def toggle_modal(n1, n2, is_open):
    print(n1, n2, is_open)
    if n1 or n2:
        return not is_open
    return is_open


app.callback(
    Output("modal_step1", "is_open"),
    [Input("button_step1", "n_clicks"), Input("close_step1", "n_clicks")],
    [State("modal_step1", "is_open")],
)(toggle_modal)

app.callback(
    Output("modal_step2", "is_open"),
    [Input("button_step2", "n_clicks"), Input("close_step2", "n_clicks")],
    [State("modal_step2", "is_open")],
)(toggle_modal)

app.callback(
    Output("modal_step3_full", "is_open"),
    [Input("button_step3_full", "n_clicks"), Input("close_step3_full", "n_clicks")],
    [State("modal_step3_full", "is_open")],
)(toggle_modal)

app.callback(
    Output("modal_step3_LIMS", "is_open"),
    [Input("button_step3_LIMS", "n_clicks"), Input("close_step3_LIMS", "n_clicks")],
    [State("modal_step3_LIMS", "is_open")],
)(toggle_modal)

app.callback(
    Output("modal_step3_sensors", "is_open"),
    [Input("button_step3_sensors", "n_clicks"), Input("close_step3_sensors", "n_clicks")],
    [State("modal_step3_sensors", "is_open")],
)(toggle_modal)

app.callback(
    Output("modal_step3_softsensors", "is_open"),
    [Input("button_step3_softsensors", "n_clicks"), Input("close_step3_softsensors", "n_clicks")],
    [State("modal_step3_softsensors", "is_open")],
)(toggle_modal)


@app.callback(Output('select_directory_output', 'children'),
              Input('select_directory', 'n_clicks'))
def select_directory(btn1):
    global folder_path
    value = [p['value'] for p in dash.callback_context.triggered][0]

    if value == None:
        return [html.Div([html.H3('Status: Please load the directory with input files')], style={'color': 'red'})]

    if value > 0:

        try:
            root.destroy()
        except:
            pass

        try:
            root = Tk()
            folder_path = filedialog.askdirectory()
            root.destroy()

        except:
            pass

        return [html.Div([html.H3('Status: Directory with input files loaded successfully')], style={'color': 'green'})]


def parse_content(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            data = pd.read_csv(io.BytesIO(decoded))

            return data

    except Exception as e:
        return html.Div(['There was an error processing this file.'])


@app.callback(Output('results_output_LIMS', 'children'),
              Input('button_start_automatic', 'n_clicks'),
              [State('riazi_radio_choice', 'value'),
               State('search_radio_choice', 'value'),
               State('LIMS_plot_decision', 'value'),
               State('sensors_plot_decision', 'value'),
               State('soft_sensors_plot_decision', 'value'),
               State('write_sensors_csv_files', 'value'),
               State('main_decision', 'value'),
               State('period_sensors', 'value'),
               State('align_in_time', 'value')])
def automatic_search(btn1, value_riazi, value_search, LIMS_plot_decision, sensors_plot_decision,
                     soft_sensors_plot_decision, write_sensors_csv_files, main_decision, period_sensors, align_in_time):
    value = [p['value'] for p in dash.callback_context.triggered][0]

    if value == None:
        return [html.H1('Results of the analysis')]

    if value > 0:

        current_time = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

        print('Started automated soft sensors development')

        name = None

        if main_decision == 'full':
            print('Option chosen: Full search')
            name = 'Full search'

        if main_decision == 'LIMS':
            print('Option chosen: LIMS only')
            name = 'LIMS only'

        if main_decision == 'sensors':
            print('Option chosen: Sensors only')
            name = 'Sensors only'

        if main_decision == 'soft_sensors':
            print('Option chosen: Soft sensors only')
            name = 'Soft sensors only'

        folder_results = folder_path + '/Results/{} - {}'.format(current_time, name)

        try:
            os.makedirs(folder_results)

        except:
            pass

        data_holder_dict, results_holder_dict = perform_auto_search(bool(value_riazi), bool(value_search),
                                                                    folder_results, main_decision,
                                                                    bool(write_sensors_csv_files), int(period_sensors),
                                                                    bool(soft_sensors_plot_decision),
                                                                    bool(LIMS_plot_decision),
                                                                    bool(sensors_plot_decision),
                                                                    align_in_time)

        # return results_content
        print('Everything completed!')
        return [html.H1('Results of the analysis')]

    return [html.H1('Results of the analysis')]


# --- Run
if __name__ == '__main__':
    app.run_server()