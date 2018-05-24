import os
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

tf.app.flags.DEFINE_string('equities', '', 'Equities')
tf.app.flags.DEFINE_string('all_equities', 'No', 'All equities')
tf.app.flags.DEFINE_string('use_adjusted', 'Yes', 'Use adjusted OHLC data')
FLAGS = tf.app.flags.FLAGS

def _read_data(equity_file):
    """Read pre-processed data for specified equity

    Args:
        equity_file: string denoting file path for equity to collect data for

    Returns:
        pandas DataFrame containing data for equity (if available)
    """
    if not os.path.isfile(equity_file):
        print("No file at path %s" % equity_file)
    else:
        data = pd.read_csv(equity_file)
        return data

def _price_change(equity, periods=1):
    """Calculate absolute change in price

    Args:
        equity: pandas DataFrame containing equity data
        periods: number of periods to calculate price change for

    Returns: None - appends 'Change' column to DataFrame
    """
    equity['Change_%dd' % periods] = equity['Close'].shift(-periods) - equity['Close']

def _percent_change(equity, periods=1):
    """Calculate percent change in price

    Args:
        equity: pandas DataFrame containing equity data
        periods: number of periods to calculate percent change for

    Returns: None - appends 'Pct_Change' column to DataFrame
    """
    equity['Pct_Change_%dd' % periods] = \
                                    (equity['Close'].shift(-periods) -
                                     equity['Close']) / equity['Close']

def _sma(equity, periods, input_column_name='Close', output_column_name=''):
    """Calculate Simple Moving Average (SMA)

    Args:
        equity: string denoting equity
        periods: number of periods to calculate simple moving average over
        input_column_name: name of column to take sma of
        output_column_name: optional name for new column

    Returns: None - appends 'SMA' column to DataFrame
    """

    if output_column_name == '':
        equity['SMA_%dd' % periods] = \
            equity[input_column_name].rolling(window=periods,
                                              min_periods=periods).mean()
    else:
        equity[output_column_name] = \
            equity[input_column_name].rolling(window=periods,
                                              min_periods=periods).mean()

def _ema(equity, periods, input_column_name='Close', output_column_name=''):
    """Calculate Exponential Moving Average (EMA)

    Args:
        equity: pandas DataFrame containing equity data
        periods: number of periods to calculate exponential moving average over
        input_column_name: name of column to take ema of
        output_column_name: optional name for new column

    Returns: None - appends 'EMA' column to DataFrame
    """

    if output_column_name == '':
        equity['EMA_%dd' % periods] = \
            equity[input_column_name].ewm(span=periods, adjust=False,
                                          min_periods=periods).mean()
    else:
        equity[output_column_name] = \
            equity[input_column_name].ewm(span=periods, adjust=False,
                                          min_periods=periods).mean()

def _rsi(equity, look_back_period=14, dow=False):
    """Calculate Relative Strength Index (RSI)

    Args:
        equity: pandas DataFrame containing equity data
        look_back_period: number of periods to calculate rsi over

    Returns: None - appends 'RSI' column to DataFrame
    """
    if 'Change_1d' not in equity.columns:
        _price_change(equity, 1)
    equity['Gain_1d'] = [x if x >= 0 else 0 for x in equity['Change_1d'].values]
    equity['Loss_1d'] = [-x if x < 0 else 0 for x in equity['Change_1d'].values]

    equity['Avg_Gain'] = equity['Gain_1d'].rolling(window=look_back_period,
                            min_periods=look_back_period).sum() / look_back_period
    equity['Avg_Loss'] = equity['Loss_1d'].rolling(window=look_back_period,
                            min_periods=look_back_period).sum() / look_back_period

    equity['RS'] = equity['Avg_Gain'] / equity['Avg_Loss']
    if not dow:
        equity['RSI_%dd' % look_back_period] = 100 - (100/(1 + (equity['RS'])))
    else:
        equity['DOW_RSI_%dd' % look_back_period] = 100 - (100/(1 + (equity['RS'])))

    # Drop unnecessary columns
    equity.drop(['Avg_Gain', 'Avg_Loss', 'Gain_1d', 'Change_1d',
                 'Loss_1d', 'RS'], axis=1, inplace=True)

def _ppo(equity, short=12, long=26, signal=9, dow=False):
    """Calculate Percentage Price Oscillator (PPO)

    Args:
        equity: pandas DataFrame containing equity data
        short: number of periods for shorter exponential moving average
        long: number of periods for longer exponential moving average
        signal: number of periods for exponential moving average of macd line

    Returns: None - appends columns 'PPO', 'PPO_Signal', and 'PPO_Hist' to
             DataFrame
    """

    # Calculate necessary exponential moving averages if necessary
    if ('EMA_%dd' % short) not in equity.columns: _ema(equity, short)
    if ('EMA_%dd' % long) not in equity.columns: _ema(equity, long)

    # Calculate DataFrame columns for MACD, MACD_Signal, and MACD_Hist
    if not dow:
        equity['PPO_%d_%d_%d' % (short, long, signal)] = ((equity['EMA_%dd' % short] -
                                                        equity['EMA_%dd' % long]) / \
                                                        equity['EMA_%dd' % long]) * 100.0
        equity['PPO_%d_%d_%d_Signal' % (short, long, signal)] = \
                equity['PPO_%d_%d_%d' % (short, long, signal)].ewm(span=signal,
                                    adjust=False, min_periods=signal).mean()
        equity['PPO_%d_%d_%d_Hist' % (short, long, signal)] = \
                equity['PPO_%d_%d_%d' % (short, long, signal)] - \
                    equity['PPO_%d_%d_%d_Signal' % (short, long, signal)]
    else:
        equity['DOW_PPO_%d_%d_%d' % (short, long, signal)] = ((equity['EMA_%dd' % short] -
                                                        equity['EMA_%dd' % long]) / \
                                                        equity['EMA_%dd' % long]) * 100.0
        equity['DOW_PPO_%d_%d_%d_Signal' % (short, long, signal)] = \
                equity['DOW_PPO_%d_%d_%d' % (short, long, signal)].ewm(span=signal,
                                    adjust=False, min_periods=signal).mean()
        equity['DOW_PPO_%d_%d_%d_Hist' % (short, long, signal)] = \
                equity['DOW_PPO_%d_%d_%d' % (short, long, signal)] - \
                    equity['DOW_PPO_%d_%d_%d_Signal' % (short, long, signal)]

    # Drop unnecessary columns
    equity.drop([('EMA_%dd' % short), ('EMA_%dd' % long)], axis=1, inplace=True)

def _full_stochastic_oscillator(equity, look_back_period=14, k_smoothing=3,
                               d_ma=3, dow=False):
    """Calculate Full Stochastics

    Args:
        equity: pandas DataFrame containing equity data
        look_back_period: number of periods to calculate lowest low and highest high
        k_smoothing: number of periods for sma of fast %K
        d_sma: number of periods for sma of %K

    Returns: None - appends columns '%K' and '%D' to DataFrame
    """

    # Determine lowest low and highest high for look_back_period
    equity['Lowest_Low'] = \
            equity['Low'].rolling(window=look_back_period, \
                                  min_periods=look_back_period).min()
    equity['Highest_High'] = \
            equity['High'].rolling(window=look_back_period, \
                                   min_periods=look_back_period).max()

    # Calculate %K and %D
    equity['K_Fast'] = ((equity['Close'] - equity['Lowest_Low']) /
                        (equity['Highest_High'] - equity['Lowest_Low'])) * 100.0
    if not dow:
        _sma(equity, k_smoothing, input_column_name='K_Fast',
            output_column_name='K_%d_%d_%d' % (look_back_period, k_smoothing, d_ma))

        _sma(equity, d_ma, input_column_name='K_%d_%d_%d' % (look_back_period, k_smoothing, d_ma),
            output_column_name='D_%d_%d_%d' % (look_back_period, k_smoothing, d_ma))
    else:
        _sma(equity, k_smoothing, input_column_name='K_Fast',
         output_column_name='DOW_K_%d_%d_%d' % (look_back_period, k_smoothing, d_ma))

        _sma(equity, d_ma, input_column_name='DOW_K_%d_%d_%d' % (look_back_period, k_smoothing, d_ma),
         output_column_name='DOW_D_%d_%d_%d' % (look_back_period, k_smoothing, d_ma))

    # Drop unnecessary columns
    equity.drop(['Lowest_Low', 'Highest_High', 'K_Fast'], axis=1, inplace=True)

def _determine_label(equity, periods):
    """Determine target label based on percent change

    Args:
        equity: DataFrame containing equity data
        periods: number of periods to calculate percent change over

    Returns: None - appends column 'Label' with value either 1(buy) or -1(sell)
    """

    pct_change_col = 'Pct_Change_%dd' % periods
    if pct_change_col not in equity.columns:
        _percent_change(equity, periods)
    equity['Label'] = [2 if x > 0.05 else 1 if x < -.05 else 0 for x in equity[pct_change_col].values]
    equity['Label'] = equity['Label'].astype('int64')

    # Drop unnecessary columns
    equity.drop([pct_change_col], axis=1, inplace=True)

def _normalize(equity):
    """Normalize input

    Args:
        equity: DataFrame containing equity data
    
    Returns: None - normalizes data in DataFrame
    """
    for column in equity.columns:
        if column != 'Close':
            equity[column] = (equity[column] - equity[column].mean()) / equity[column].std()


def main(unused_arg):
    equity_files = []

    # Determine which equities are to be processed based on command line flags
    if FLAGS.all_equities.lower() == 'yes':
        if FLAGS.use_adjusted.lower() == 'yes':
            equity_files = glob('data/equities/adjusted-pre-processed/*.csv')
        else:
            equity_files = glob('data/equities/pre-processed/*.csv')
    else:
        equities = FLAGS.equities.upper().split(',')
        if FLAGS.use_adjusted.lower() == 'yes':
            for equity in equities:
                equity_files.append(('data/equities/adjusted-pre-processed/%s.csv'
                                     % equity))
        else:
            for equity in equities:
                equity_files.append(('data/equities/pre-processed/%s.csv'
                                     % equity))

    # Process data for Dow
    if FLAGS.use_adjusted.lower() == 'yes':
        dow = _read_data('data/equities/adjusted-pre-processed/DJI.csv')
    else:
        dow = _read_data('data/equities/pre-processed/DJI.csv')
    _rsi(dow, dow=True)
    _ppo(dow, dow=True)
    _full_stochastic_oscillator(dow, dow=True)
    dow.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
    
    # Master DataFrame to hold all data
    master = pd.DataFrame()

    for file in equity_files:
        equity = _read_data(file)

        # Check to make sure file has information
        if equity.empty:
            continue

        # Perform processing
        _rsi(equity)
        _ppo(equity)
        _full_stochastic_oscillator(equity)

        # Merge in Dow data and drop unecessary columns
        equity = pd.merge(equity, dow, on='Date')
        equity.drop(['Open', 'High', 'Low', 'Date', 'Volume'], axis=1, inplace=True)
        equity.dropna(axis=0, how='any', inplace=True)
        equity.reset_index(drop=True, inplace=True)

        # Normalize data
        _normalize(equity)

        # Determine label
        _determine_label(equity, 20)

        # Remove any rows containing missing values and reset index column
        equity.dropna(axis=0, how='any', inplace=True)
        equity.reset_index(drop=True, inplace=True)

        # Drop Close column
        equity.drop(['Close'], axis=1, inplace=True)

        # Append data to master DataFrame
        master = pd.concat([master, equity], ignore_index=True)

    # Split into test and train
    master = shuffle(master)
    total_buys = 0
    total_sells = 0
    total_holds = 0
    for entry in master['Label'].values:
        if entry == 0:
            total_holds += 1
        elif entry == 1:
            total_sells += 1
        else:
            total_buys += 1
    
    print("Buys: %d\nSells: %d\nHolds: %d\n" % (total_buys, total_sells, total_holds))
    train, test = train_test_split(master, test_size=0.2)

    train.to_csv('data/equities/post-processed/train.csv', index=False)
    test.to_csv('data/equities/post-processed/test.csv', index=False)

if __name__ == '__main__':
    tf.app.run()
