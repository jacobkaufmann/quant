import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import pandas as pd

from pandas_datareader import data as web
import fix_yahoo_finance as yf
import wget

tf.app.flags.DEFINE_string('equity', '', 'equity')
tf.app.flags.DEFINE_string('all_equities', 'No', 'All equities')
tf.app.flags.DEFINE_string('adjust', 'No', 'Adjust OHLC')
FLAGS = tf.app.flags.FLAGS

def make_directories():
    """Create directory structure for equities data.
    """
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/equities'):
        os.makedirs('data/equities')
        os.makedirs('data/equities/pre-processed')
        os.makedirs('data/equities/adjusted-pre-preprocessed')
        os.makedirs('data/equities/post-processed')

def get_data_for_equity(equity, start, end=None, adjusted=False):
    """Collect ohlc (open, high, low, close) and volume data for equity.

    Args:
        equity: string denoting the equity to get data for
        start: datetime for the beginning of term to collect data for
        end: datetime for the end of term to collect data for

    Returns:
        pandas DataFrame containing the available data for the specified term
    """
    if end == None:
        data = web.get_data_yahoo(equity, start=start, auto_adjust=adjusted)
    else:
        data = web.get_data_yahoo(equity, start=start, end=end,
                                  auto_adjust=adjusted)
    return data

def main(unused_arg):
    yf.pdr_override()

    """
    Make directories for pre-processed and post-processed data
    """
    make_directories()
    start = datetime(1980, 1, 1)
    equities = []

    """
    If all equities flag set to 'yes', download csv files of all current
    US equities on the following exchanges: Nasdaq, NYSE, AMEX
    """
    if FLAGS.all_equities.lower() == 'yes':
        if not os.path.isfile('data/nasdaq-symbols.csv'):
            wget.download('https://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download',
                      'data/nasdaq-symbols.csv')
        if not os.path.isfile('data/nyse-symbols.csv'):
            wget.download('https://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nyse&render=download',
                      'data/nyse-symbols.csv')
        if not os.path.isfile('data/amex-symbols.csv'):
            wget.download('https://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=amex&render=download',
                      'data/amex-symbols.csv')
        nasdaq_symbols = pd.read_csv('data/nasdaq-symbols.csv').values
        nyse_symbols = pd.read_csv('data/nyse-symbols.csv').values
        amex_symbols = pd.read_csv('data/amex-symbols.csv').values
        all_symbols = np.concatenate((nasdaq_symbols, nyse_symbols,
                                     amex_symbols), axis=0)
        equities = all_symbols[:,0]

    """
    Otherwise use the single equity passed in
    """
    else:
        equities = [FLAGS.equity]

    """
    Loop over all equities and write pre-processed data to csv files in
    appropriate directory (may be adjusted or non-adjusted depending on flag)
    """
    for equity in equities:
        equity_file = 'data/equities/adjusted-pre-processed/%s.csv' % equity
        if not os.path.isfile(equity_file):
            if FLAGS.adjust.lower() == yes:
                data = get_data_for_equity(equity, start, adjusted=True)
                data.to_csv()
            else:
                equity_file = 'data/equities/pre-processed/%s.csv' % equity
                data = get_data_for_equity(equity, start)
                data.to_csv(equity_file)

if __name__ == '__main__':
    tf.app.run()
