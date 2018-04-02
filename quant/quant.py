# Author: Jacob Kaufmann
# Purpose: Gather Data on equities in information technology sector
# Date: 10/12/17

# Import required libraries
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import pickle
import tensorflow as tf
import tflearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import technical_indicators as ti

# Flag for desired security to train and make predictions for
tf.app.flags.DEFINE_string('security', '', 'Security')

# Flag for desired prediction interval
tf.app.flags.DEFINE_string('interval', 10, 'Interval')

# Flag for desired return threshold (what threshold of return is necessary to consider something a 'sell' or 'buy')
tf.app.flags.DEFINE_float('return_threshold', .075, 'Return threshold')

FLAGS = tf.app.flags.FLAGS

# Create directory for data
if not os.path.exists('data'):
    os.makedirs('data')

# Create directory for time series data
if not os.path.exists('data/equities'):
    os.makedirs('data/equities')
    
# Create directory for preprocessed time series data
if not os.path.exists('data/equities/preprocessed'):
    os.makedirs('data/equities/preprocessed')
    
# Create directory for processed time series data
if not os.path.exists('data/equities/processed'):
    os.makedirs('data/equities/processed')

    
# Function to calculate technical indicators for an security
def calculate_indicators_for_security(security):
    ti.calculate_avg_vol_for_interval(security, 3)
    ti.calculate_avg_vol_for_interval(security, 5)
    ti.calculate_avg_vol_for_interval(security, 10)
    ti.calculate_avg_vol_for_interval(security, 30)
    ti.calculate_avg_vol_for_interval(security, 90)

    ti.calculate_sma_for_interval(security, 15)
    ti.calculate_sma_for_interval(security, 60)
    ti.calculate_sma_for_interval(security, 100)

    ti.calculate_ema_for_interval(security, 10)
    ti.calculate_ema_for_interval(security, 15)
    ti.calculate_ema_for_interval(security, 30)
    ti.calculate_ema_for_interval(security, 45)

    ti.calculate_macd(security, 12, 26, 9)

    ti.calculate_ratio(security, '5d_Avg_Vol', '30d_Avg_Vol')
    ti.calculate_ratio(security, '10d_Avg_Vol', '90d_Avg_Vol')

    ti.calculate_low_for_interval(security, 'Adj Close', 30)
    ti.calculate_high_for_interval(security, 'Adj Close', 30)
    ti.calculate_low_for_interval(security, 'Adj Close', 200)
    ti.calculate_high_for_interval(security, 'Adj Close', 200)

    ti.calculate_per_day_price_change(security)

    ti.calculate_rsi(security, 14)

    ti.calculate_full_stochs(security, 14, 3, 3)

    ti.calculate_adl(security)

    ti.calculate_chaikin_osc(security)

    ti.calculate_ratio(security, 'Adj Close', '30d_Low')
    ti.calculate_ratio(security, 'Adj Close', '30d_High')
    ti.calculate_ratio(security, 'Adj Close', '200d_Low')
    ti.calculate_ratio(security, 'Adj Close', '200d_High')
    ti.calculate_ratio(security, 'Adj Close', '60d_SMA')
    ti.calculate_ratio(security, 'Adj Close', '15d_EMA')
    ti.calculate_ratio(security, 'Adj Close', '30d_EMA')
    ti.calculate_ratio(security, 'Adj Close', '45d_EMA')
    ti.calculate_ratio(security, '10d_EMA', '45d_EMA')
    ti.calculate_ratio(security, '10d_EMA', '60d_SMA')
    ti.calculate_ratio(security, '15d_EMA', '60d_SMA')
    ti.calculate_ratio(security, '15d_EMA', '100d_SMA')


# Define function to calculate response and create column for target label
def calculate_target_label_for_interval(security, days, threshold):
    security['{}d_Pct_Change'.format(days)] = \
        (security['Adj Close'].shift(-days) - security['Adj Close']) / security['Adj Close']
    security['{}d_{}%_Label'.format(days, threshold * 100)] = \
        ['buy' if x > threshold else 'sell' if x < -threshold else 'hold' for x in security['{}d_Pct_Change'.format(days)]]


# Window of time to grab security data
start = dt.datetime(1977, 1, 1)
end = dt.datetime(2017, 7, 31)

# Set flags to vars
security = FLAGS.security
interval = FLAGS.interval
threshold = FLAGS.threshold

# Collect time series data for all possible symbols and store it locally so it does not have to be re-pulled from yahoo
if not os.path.exists('data/equities/preprocessed/' + security + '.csv'):
    try:
        df = web.DataReader(security, 'yahoo', start, end)
        df.to_csv('data/equities/preprocessed/' + security + '.csv')
    except ValueError:
        print("Error")


# For each symbol, look for file of time series data in preprocessed directory
# If exists, retrieve the data, calculate technical indicators and response label
# Store updated data into the processed directory
df = pd.DataFrame()
if os.path.exists('data/equities/preprocessed/' + security + '.csv'):
    # Read in pre-procesesd data
    df = pd.read_csv('data/equities/preprocessed/' + security + '.csv')

    # Calculate technical indicators for security
    calculate_indicators_for_security(df)

    # Calculate target labels for security
    calculate_target_label_for_interval(df, interval, threshold)

    # Drop rows containing missing values and columns not necessary for training
    df = df.drop(df.tail(interval).index, inplace=False)
    df = df.drop(['Date', 'Open'], axis=1)
    df = df.dropna(how='any')

    # Reset index column after the dropping of rows
    df = df.reset_index(drop=True)
    df.to_csv('data/equities/processed/' + security + '.csv', index=False)


# Pre-process data and fit to model
features = []
response = '20d_10.0%_Label'

X = df.drop(['High', 'Low', 'Volume'], axis=1)
X = X.drop(['{}d_{}%_Label'.format(interval, threshold * 100)], axis=1)
        
X_Scaled = preprocessing.scale(X)

y = df.filter([response])

X_train, X_test, y_train, y_test = train_test_split(X[features], y['20d_10.0%_Label'], test_size=0.33, random_state=42)
