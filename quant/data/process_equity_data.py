import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

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
    equity['Change_%dd' % periods] = equity['Close'].diff(periods=periods)

def _percent_change(equity, periods=1):
    """Calculate percent change in price

    Args:
        equity: pandas DataFrame containing equity data
        periods: number of periods to calculate percent change for

    Returns: None - appends 'Pct_Change' column to DataFrame
    """
    equity['Pct_Change_%dd' % periods] = \
                                    equity['Close'].pct_change(periods=periods)

def _sma(equity, periods, input_column_name='Close', output_column_name=''):
    """Calculate simple moving average

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
    """Calculate exponential moving average

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

def _rsi(equity, look_back_period=14):
    """Calculate relative strength index

    Args:
        equity: pandas DataFrame containing equity data
        look_back_period: number of periods to calculate rsi over

    Returns: None - appends 'RSI' column to DataFrame
    """
    if 'Change_1d' not in equity.columns:
        price_change(equity, 1)
    equity['Gain_1d'] = [x if x >= 0 else 0 for x in equity['Change_1d'].values]
    equity['Loss_1d'] = [-x if x < 0 else 0 for x in equity['Change_1d'].values]

    equity['Avg_Gain'] = equity['Gain_1d'].rolling(window=look_back_period,
                            min_periods=look_back_period).sum() / look_back_period
    equity['Avg_Loss'] = equity['Loss_1d'].rolling(window=look_back_period,
                            min_periods=look_back_period).sum() / look_back_period

    equity['RS'] = equity['Avg_Gain'] / equity['Avg_Loss']
    equity['RSI_%dd' % look_back_period] = 100 - (100/(1 + (equity['RS'])))

    # Drop unnecessary columns
    equity.drop(['Avg_Gain', 'Avg_Loss', 'Gain_1d', 'Loss_1d', 'RS'], axis=1,
                inplace=True)

def _macd(equity, short=12, long=26, signal=9):
    """Calculate Moving Average Convergence Divergence (MACD)

    Args:
        equity: pandas DataFrame containing equity data
        short: number of periods for shorter exponential moving average
        long: number of periods for longer exponential moving average
        signal: number of periods for exponential moving average of macd line

    Returns: None - appends columns 'MACD', 'MACD_Signal', and 'MACD_Hist' to
             DataFrame
    """

    # Calculate necessary exponential moving averages if necessary
    if ('EMA_%dd' % short) not in equity.columns: _ema(equity, short)
    if ('EMA_%dd' % long) not in equity.columns: _ema(equity, long)

    # Calculate DataFrame columns for MACD, MACD_Signal, and MACD_Hist
    equity['MACD_%d_%d_%d' % (short, long, signal)] = equity['EMA_%dd' % short] - \
                                                        equity['EMA_%dd' % long]
    equity['MACD_%d_%d_%d_Signal' % (short, long, signal)] = \
            equity['MACD_%d_%d_%d' % (short, long, signal)].ewm(span=signal,
                                adjust=False, min_periods=signal).mean()
    equity['MACD_%d_%d_%d_Hist' % (short, long, signal)] = \
            equity['MACD_%d_%d_%d' % (short, long, signal)] - \
                equity['MACD_%d_%d_%d_Signal' % (short, long, signal)]

    # Drop unnecessary columns
    equity.drop([('EMA_%dd' % short), ('EMA_%dd' % long)], axis=1, inplace=True)

def _full_stochastic_oscillator(equity, look_back_period=14, k_smoothing=3,
                               d_ma=3):
    """Calculate Moving Average Convergence Divergence (MACD)

    Args:
        equity: pandas DataFrame containing equity data
        look_back_period: number of periods to calculate lowest low and highest high
        k_smoothing: number of periods for sma of fast %K
        d_sma: number of periods for sma of %K

    Returns: None - appends columns '%K' and '%D' to DataFrame
    """

    # Determine lowest low and highest high for look_back_period
    equity['Lowest_Low'] = \
            equity['Low'].rolling(window=look_back_period,
                                  min_periods=look_back_period).min()
    equity['Highest_High'] = \
            equity['High'].rolling(window=look_back_period,
                                  min_periods=look_back_period).max()

    # Calculate %K and %D
    equity['K_Fast'] = (equity['Close'] - equity['Lowest_Low']) / \
                   (equity['Highest_High'] - equity['Lowest_Low']) * 100
    _sma(equity, k_smoothing, input_column_name='K_Fast',
         output_column_name='K_%d_%d_%d' % (look_back_period, k_smoothing, d_ma))

    _sma(equity, d_ma, input_column_name='K_%d_%d_%d' % (look_back_period, k_smoothing, d_ma),
         output_column_name='D_%d_%d_%d' % (look_back_period, k_smoothing, d_ma))

    # Drop unnecessary columns
    equity.drop(['Lowest_Low', 'Highest_High', 'K_Fast'], axis=1, inplace=True)

def main(unused_arg):
    equity_files = []

    # Determine which equities are to be processed based on command line flags
    if FLAGS.all_equities.lower() == 'yes':
        if FLAGS.use_adjusted.lower() == 'yes':
            equity_files = os.listdir('data/equities/adjusted-pre-processed')
        else:
            equity_files = os.listdir('data/equities/pre-processed')
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

    for file in equity_files:
        equity = _read_data(file)

        # Perform processing
        _price_change(equity)
        _percent_change(equity)
        _ema(equity, 10)
        _sma(equity, 10)
        _rsi(equity)
        _macd(equity)
        _full_stochastic_oscillator(equity)

        # Drop unnecessary columns
        equity.drop(['Date', 'Open', 'High', 'Low', 'Close'], axis=1, inplace=True)

        # Remove any rows containing missing values and reset index column
        equity.dropna(axis=0, how='any', inplace=True)
        equity.reset_index(drop=True, inplace=True)

        print(equity)

if __name__ == '__main__':
    tf.app.run()
