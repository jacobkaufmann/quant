import os

import numpy as np
import pandas as pd

import tensorflow as tf

def compose_company_matrix(files, columns=["Symbol", "Name", "MarketCap", "Sector", "Industry"]):
    companies = []
    for file in files:
        data = pd.read_csv(file)
        data = data[columns]
        if len(companies) == 0:
            companies = data.values
        else:
            companies = np.append(companies, data.values, axis=0)
    return companies

def compose_company_record(symbol, interval="daily"):
    pass

compose_company_matrix([])
