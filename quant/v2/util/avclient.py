import sys
import json
import requests

import numpy as np
import pandas as pd

import avtoken

API_TOKEN = avtoken.get_token()
BASE_URL = "https://www.alphavantage.co/query?"

data_dir = "../data/"
intervals = {"1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"}
series_types = {"close", "open", "high", "low"}
data_types = {"json", "csv"}
output_sizes = {"full", "compact"}

class AlphaVantageClient(object):
    def __init__(self, max_attempts=3):
        self.token = API_TOKEN
        self.max_attempts = max_attempts

    def ts_daily(self, symbol, adjusted=True, output_size="full", data_type="csv"):
        assert(output_size in output_sizes)
        assert(data_type in data_types)

        func = "TIME_SERIES_DAILY"
        if adjusted:
            func = func + "_ADJUSTED"
        url = "{}function={}&symbol={}&outputsize={}&apikey={}&datatype={}".format(
            BASE_URL, func, symbol, output_size, self.token, data_type
        )
        if data_type == "csv":
            try:
                attempt = 0
                while attempt < self.max_attempts:
                    data = pd.read_csv(url)
                    columns = data.columns.values
                    columns[0] = "time"
                    data.columns = columns
                    return data
            except requests.exceptions.RequestException as e:
                print(e)
        else:
            pass
            
    def sma(self, symbol, interval="daily", time_period=15, series_type="close", data_type="csv"):
        assert(interval in intervals)
        assert(series_type in series_types)
        assert(data_type in data_types)

        url = "{}function=SMA&symbol={}&interval={}&time_period={}&series_type={}&apikey={}&datatype={}".format(
            BASE_URL, symbol, interval, time_period, series_type, self.token, data_type
        )
        if data_type == "csv":
            try:
                attempt = 0
                while attempt < self.max_attempts:
                    return pd.read_csv(url)
            except requests.exceptions.RequestException as e:
                print(e)
        else:
            pass

    def ema(self, symbol, interval="daily", time_period=15, series_type="close", data_type="csv"):
        assert(interval in intervals)
        assert(series_type in series_types)
        assert(data_type in data_types)

        url = "{}function=EMA&symbol={}&interval={}&time_period={}&series_type={}&apikey={}&datatype={}".format(
            BASE_URL, symbol, interval, time_period, series_type, self.token, data_type
        )
        if data_type == "csv":
            try:
                attempt = 0
                while attempt < self.max_attempts:
                    return pd.read_csv(url)
            except requests.exceptions.RequestException as e:
                print(e)
        else:
            pass

    def wma(self, symbol, interval="daily", time_period=15, series_type="close", data_type="csv"):
        assert(interval in intervals)
        assert(series_type in series_types)
        assert(data_type in data_types)

        url = "{}function=WMA&symbol={}&interval={}&time_period={}&series_type={}&apikey={}&datatype={}".format(
            BASE_URL, symbol, interval, time_period, series_type, self.token, data_type
        )
        if data_type == "csv":
            try:
                attempt = 0
                while attempt < self.max_attempts:
                    return pd.read_csv(url)
            except requests.exceptions.RequestException as e:
                print(e)
        else:
            pass

    def macd(self, symbol, interval="daily", series_type="close", data_type="csv", fast=12, slow=26, signal=9):
        assert(interval in intervals)
        assert(series_type in series_types)
        assert(data_type in data_types)

        url = "{}function=MACD&symbol={}&interval={}&series_type={}&apikey={}&fastperiod={}&slowperiod={}&signalperiod={}&datatype={}".format(
            BASE_URL, symbol, interval, series_type, self.token, fast, slow, signal, data_type
        )
        if data_type == "csv":
            try:
                attempt = 0
                while attempt < self.max_attempts:
                    return pd.read_csv(url)
            except requests.exceptions.RequestException as e:
                print(e)
        else:
            pass

    def stoch(self, symbol, interval="daily", data_type="csv", fastk=5, slowk=3, slowd=3):
        assert(interval in intervals)
        assert(data_type in data_types)

        url = "{}function=STOCH&symbol={}&interval={}&apikey={}&fastkperiod={}&slowkperiod={}&slowdperiod={}&datatype={}".format(
            BASE_URL, symbol, interval, self.token, fastk, slowk, slowd, data_type
        )
        if data_type == "csv":
            try:
                attempt = 0
                while attempt < self.max_attempts:
                    return pd.read_csv(url)
            except requests.exceptions.RequestException as e:
                print(e)
        else:
            pass

    def rsi(self, symbol, interval="daily", time_period=14, series_type="close", data_type="csv"):
        assert(interval in intervals)
        assert(series_type in series_types)
        assert(data_type in data_types)

        url = "{}function=RSI&symbol={}&interval={}&time_period={}&series_type={}&apikey={}&datatype={}".format(
            BASE_URL, symbol, interval, time_period, series_type, self.token, data_type
        )
        if data_type == "csv":
            try:
                attempt = 0
                while attempt < self.max_attempts:
                    return pd.read_csv(url)
            except requests.exceptions.RequestException as e:
                print(e)
        else:
            pass

    def ppo(self, symbol, interval="daily", series_type="close", data_type="csv", fast=12, slow=26):
        assert(interval in intervals)
        assert(series_type in series_types)
        assert(data_type in data_types)

        url = "{}function=PPO&symbol={}&interval={}&series_type={}&fastperiod={}&slowperiod={}&apikey={}&datatype={}".format(
            BASE_URL, symbol, interval, series_type, slow, fast, self.token, data_type
        )
        if data_type == "csv":
            try:
                attempt = 0
                while attempt < self.max_attempts:
                    return pd.read_csv(url)
            except requests.exceptions.RequestException as e:
                print(e)
        else:
            pass

    def ad(self, symbol, interval="daily", data_type="csv"):
        assert(interval in intervals)
        assert(data_type in data_types)

        url = "{}function=AD&symbol={}&interval={}&apikey={}&datatype={}".format(
            BASE_URL, symbol, interval, self.token, data_type
        )
        if data_type == "csv":
            try:
                attempt = 0
                while attempt < self.max_attempts:
                    return pd.read_csv(url)
            except requests.exceptions.RequestException as e:
                print(e)
        else:
            pass


equity = "DBX"
client = AlphaVantageClient()
daily = client.ts_daily(equity)
sma = client.sma(equity)
ema = client.ema(equity)
wma = client.wma(equity)
macd = client.macd(equity)
stoch = client.stoch(equity)
rsi = client.rsi(equity)
ppo = client.ppo(equity)
ad = client.ad(equity)

data = daily
measures = [sma, ema, wma, macd, stoch, rsi, ppo, ad]
for i, measure in enumerate(measures):
    if len(measure) < 50:
        print("Fail on ", i)
    data = pd.merge(data, measure, on="time")
