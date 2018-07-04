#pylint: disable-msg=too-many-arguments
'''Utilities for building stock data for training and testing'''
import os
import warnings

import dill as pickle
import numpy as np
import pandas as pd
import pandas_talib as ta
import peakutils
import quandl
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

CACHE_DIR = 'stock_data/'
HI_LO_DIFF = 0.03
MIN_MAX_PERIOD = 8

with open('quandl_key.txt') as key_file:
    quandl.ApiConfig.api_key = key_file.readline().strip()

np.random.seed(1337) # for reproducibility
warnings.simplefilter(action='ignore', category=FutureWarning)


def build_train_data(stock_code, build_labels=True, build_test=False, test_ratio=0.1,
                     start_date=None, end_date=None, cache=True, shuffle_data=False):

    '''Build training data and indicators for given stocks.

    Parameters
    ----------
    stock_code: Quandl-compatible stock code string
    build_for_train: whether to build labels
    build_test: whether to build test data. Must be used with build_labels
    test_ratio: ratio of test data to train data. Only used if `build_for_train==True`.
    start_date: Start date in YYYY-MM-DD format. `None` = start from earliest
    end_date: Start date in YYYY-MM-DD format. `None` = end at latest
    cache: Whether to cache the data to avoid re-building it later. If requested data is already in\
         cache,return cached data instead of rebuilding.
    shuffle_data: Whether to randomly shuffle data to avoid chronological dependencies

    Returns
    -------
    Dictionary containing a Pandas dataframe and the labels, one for train data, \
        one for test.
    Dictionary keys: train_x, train_y, test_x, test_y
    i.e. `[{'train_x': train1_data, 'train_y': train1_labels,
            'test_x': test1_data, 'test_y': test1_labels}]`
    '''
    print('Building', stock_code)
    output = {}
    data = _get_cached_data(stock_code, build_labels, start_date, end_date) if cache else None
    need_to_cache = True
    if data is None:
        try:
            data = quandl.get(stock_code, start_date=start_date, end_date=end_date)

        except quandl.errors.quandl_error.NotFoundError:
            # invalid stock code
            print('Invalid code: {}'.format(stock_code))
            return None

        data = _rename_columns(data)
        data.reset_index(inplace=True, drop=True)
        data = build_indicators(data)

        if build_labels:
            data['Y'] = compute_labels(data['Close'])

        data = data.replace([np.inf, -np.inf], np.nan)
        data.dropna(inplace=True)
        # If target Y data is requested, build it
        if build_labels:
            labels = data.pop('Y')

    else:
        need_to_cache = False
        labels = data.pop('Y')

    data = normalize(data)
    if build_labels and build_test:
        if shuffle_data:
            train_x, test_x, train_y, test_y = train_test_split(data, labels,
                                                      test_size=test_ratio,
                                                      random_state=0)

        else: 		# just clips the test data off the end
            length = len(data)
            test_size = int(test_ratio*length)
            train_x, test_x = data[:-test_size], data[-test_size:]
            train_y, test_y = labels[:-test_size], labels[-test_size:]

        output['train_x'], output['test_x'] = train_x, test_x
        output['train_y'] = np.array(train_y.values.tolist())
        output['test_y'] = np.array(test_y.values.tolist())

    output['X'] = data
    if build_labels:
        output['Y'] = labels.values

    output['price'] = data['Close']

    if cache and need_to_cache:
        full_data = data.assign(Y=labels)
        _cache_data(full_data, stock_code, build_labels, start_date, end_date)

    return output


def build_train_data_multiple(stock_codes, **kwargs):
    '''Build training data and indicators for given stocks.

    Parameters
    ----------
    stock_codes: list of Quandl-compatible stock code strings
    build_labels: whether to set aside portion of data as test
    build_test: whether to build test data. Must be used with build_labels
    test_ratio: ratio of test data to train data. Only used if `build_for_train==True`.
    start_date: Start date in YYYY-MM-DD format. `None` = start from earliest
    end_date: Start date in YYYY-MM-DD format. `None` = end at latest
    cache: Whether to cache the data to avoid re-building it later. If requested data is already \
        in cache, return cached data instead of rebuilding.
    shuffle_data: Whether to randomly shuffle data to avoid chronological dependencies

    Returns
    -------
    List of dictionaries containing Pandas dataframes of the training and testing data \
        and the labels, one for train data, one for test.
    Dictionary keys: train_x, train_y, test_x, test_y
    i.e. `[{'train_x': train1_data, 'train_y': train1_labels,
            'test_x': test1_data, 'test_y': test1_labels},
           {...}]`
    '''
    # TODO: Parallelize
    for stock_code in stock_codes:
        yield build_train_data(stock_code, **kwargs)

def build_indicators(data):
    '''
    Builds technical indicators on dataframe
    '''
    no_args = ['PPSR', 'STOK', 'MassI', 'Chaikin', 'ULTOSC', ]
    one_args = ["MA", "MOM", "ROC", "ATR", "BBANDS", "STO", "TRIX", "Vortex",
                "RSI", "ACCDIST", "MFI", "OBV", "FORCE", "EOM", "CCI", "COPP", "KELCH",
                "STDDEV"]
    no_args_normalize = ["Chaikin"]
    one_arg_normalize = ["MA", "ATR", "CCI", "Force", "KelChD", "KelChM", "KelChU", "Momentum", "OBV", "Ultimate_Osc"]
    # TODO: KST, TSI
    for indicator in no_args:
        data = getattr(ta, indicator)(data)
    for period in range(2, 40, 8):
        for indicator in one_args:
            data = getattr(ta, indicator)(data, period)
        data = ta.MACD(data, period*2, period)
        period_suffix = "_%d_%d" % (period*2, period)
        data.drop(["MACD"+period_suffix, "MACDsign" + period_suffix])
        data = ta.ADX(data, period, period)
    data['Volume'] = data['Volume'].apply(lambda x: x/10000)

    return data

def compute_labels(close_series):
    '''
    Compute direction of price movement for the next day
    0 = current day is a max/going toward a min/sell signal
    1 = current day is a min/going toward a max/buy signal
    '''
    price = close_series.values
    min_idxs = peakutils.indexes(-price, HI_LO_DIFF)
    max_idxs = peakutils.indexes(price, HI_LO_DIFF)

    labels = pd.Series(name="signal", dtype=float,
                       index=range(0, len(price)))

    for idx in min_idxs:
        labels.set_value(idx, 1)

    for idx in max_idxs:
        labels.set_value(idx, 0)

    # print("MAXS:", n)
    _min_idx, _max_idx = 0, 0
    for i, label in np.ndenumerate(labels.values):
        if label == 1:
            _min_idx = i[0]
        elif label == 0:
            _max_idx = i[0]
        else:
            if _min_idx > _max_idx:
                intermediate_label = 1
            elif _max_idx > _min_idx:
                intermediate_label = 0
            else:
                # no action taken
                # only occurs at the beginnings of datasets, afaik
                intermediate_label = None

            labels.set_value(i, intermediate_label)

    return labels

def _get_cached_data(stock_code, build_for_train, start_date, end_date):
    sec = stock_code.split("/")[1]	# Just the ticker, not the DB code
    pickle_name = _build_pickle_name(sec, build_for_train, start_date, end_date)
    if os.path.isfile(CACHE_DIR + pickle_name + ".pkl"):
        with open(CACHE_DIR + pickle_name + ".pkl", 'rb') as file:
            return pickle.load(file)
    return None

def _cache_data(dataframe, stock_code, build_for_train, start_date, end_date):
    sec = stock_code.split("/")[1]	# Just the ticker, not the DB code
    pickle_name = _build_pickle_name(sec, build_for_train, start_date, end_date)
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    with open(CACHE_DIR + pickle_name + ".pkl", 'wb') as file:
        pickle.dump(dataframe, file)

def _rename_columns(data):
    if 'Adj. Close' in data.columns:
        data = data[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
        # Remove the 'Adj. ' and make lowercase
        data.rename(columns=lambda x: x[5:], inplace=True)
    elif 'Close' in data.columns:
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        # df.rename(columns=lambda x: x.lower(), inplace=True)    # make lowercase

    return data

def _build_pickle_name(sec, build_for_train, start_date, end_date):
    pickle_name = sec
    if build_for_train:
        pickle_name += "_withtest"
    if start_date and end_date:
        pickle_name += start_date + "to" + end_date
    elif start_date:
        pickle_name += start_date
    elif end_date:
        pickle_name += "to" + end_date
    return pickle_name

def get_num_features():
    data = quandl.get('WIKI/AAPl', start_date='2016-01-04', end_date='2016-01-08')
    data = _rename_columns(data)
    data.reset_index(inplace=True, drop=True)
    data = build_indicators(data)
    return len(data.iloc[-1].values)

def normalize(data):
    '''Normalizes given dataframe column-wise'''
    # x = data.values
    # min_max_scaler = preprocessing.MinMaxScaler()
    # return pd.DataFrame(min_max_scaler.fit_transform(x.T).T, columns=data.columns, index=data.index)
    return data
