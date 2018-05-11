import numpy as np
import tensorflow as tf
import tflearn
import pandas as pd

from data import process_equity_data

stock = 'AAPL'
data = process_equity_data.read_data(stock)
print(data)
