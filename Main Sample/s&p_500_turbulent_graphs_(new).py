# -*- coding: utf-8 -*-
"""S&P 500 Turbulent graphs (new).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Oa32R-7YfmQMU5Qah2ggpCKO08h8zkT-
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_excel("Forecasts turbulent.1.xlsx",index_col='Date')

plt.plot(df.index, df["Actuals"])
plt.plot(df.index, df["Forecast without PH"])
plt.ylabel("S&P 500 Realized Volatility")
plt.xticks(rotation=50)
plt.show()

plt.plot(df.index, df["Actuals"])
plt.plot(df.index, df["Forecast with PH"])
plt.ylabel("S&P 500 Realized Volatility")
plt.xticks(rotation=50)
plt.show()

plt.plot(df.index, df["Actuals"])
plt.plot(df.index, df["HAR"])
plt.ylabel("S&P 500 Realized Volatility")
plt.xticks(rotation=50)
plt.show()

plt.plot(df.index, df["Actuals"])
plt.plot(df.index, df["HARX"])
plt.ylabel("S&P 500 Realized Volatility")
plt.xticks(rotation=50)
plt.show()

plt.plot(df.index, df["Actuals"])
plt.plot(df.index, df["HARST - PH"])
plt.ylabel("S&P 500 Realized Volatility")
plt.xticks(rotation=50)
plt.show()

plt.plot(df.index, df["Actuals"])
plt.plot(df.index, df["HARST (week)"])
plt.ylabel("S&P 500 Realized Volatility")
plt.xticks(rotation=50)
plt.show()

