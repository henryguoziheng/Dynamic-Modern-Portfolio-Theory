import time
import numpy as np
from numpy import power, prod, dot, cumprod, multiply, divide, square, exp, transpose, ones
from numpy.linalg import inv, multi_dot
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import pandas as pd
import datetime as dt


symbol = 'WIKI/AAPL'  # or 'AAPL.US'

df = web.DataReader(symbol, 'quandl', '2015-01-01', '2015-01-05')

print (df.head())