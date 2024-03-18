"""

@author: Hugo Gobato Souto
International School of Business at HAN University of Applied Sciences, Ruitenberglaan 31, 6826 CC Arnhem, the Netherlands; 
H.GobatoSouto@han.nl; https://orcid.org/0000-0002-7039-0572
"""

# This algorithm requires the installment of ripster library
# One can do so by !pip install ripser

import yfinance as yf
import numpy as np
import pandas as pd

from ripser import Rips
import persim

import matplotlib.pyplot as plt

import os, pickle, base64

# index names: e.g. ^GSPC = S&P 500, ^DJI = DOW Jones, ^RUT = Russell 2000
def fetch_raw_data(
        index_names = ['^GSPC', '^DJI', '^RUT'],
        start_date_string = "2000-01-01",
        end_date_string = "2022-03-30"):

    cache_name = '+'.join(sorted(index_names)) + f'+{start_date_string}+{end_date_string}'
    cache_name = base64.b64encode(bytes(cache_name, 'utf8'), b'+-')
    cache_name = cache_name + b'.pickle'
    cache_name = cache_name.decode('utf8')

    if os.path.exists(cache_name):
        print('* cache hit')
        with open(cache_name, 'rb') as file:
            raw_data = pickle.load(file)
    else:
        print('* fetching data from yahoo finance')
        raw_data = yf.download(index_names, start=start_date_string, end=end_date_string)
        with open(cache_name, 'wb') as file:
            pickle.dump(raw_data, file)

    return raw_data

raw_data = fetch_raw_data()
print(raw_data)

exit()

#Using own stock data in Excel
#raw_data = pd.read_excel("Doc.xlsx", index_col='Date')
#or CSV
#raw_data = pd.read_csv("Doc.csv", index_col='Date')

# cleaning data and keeping only adjusted closing prices
df_close = raw_data['Adj Close'].dropna(axis='rows')

# define array of adjusted closing prices
P = df_close.to_numpy()
# define array of log-returns defined as the log of the ratio between closing values of two subsequent days
r = np.log(np.divide(P[1:],P[:len(P)-1]))

# Instantiate Vietoris-Rips solver
rips = Rips(maxdim = 2)

#D=number of considered days for Homology Persistence Diagram

dgm = rips.fit_transform(r[0:D])

plt.figure(figsize=(5, 5), dpi=80)
plt.rcParams.update({'font.size': 10})
persim.plot_diagrams(dgm)

#plt.savefig("homology_persistence-diagram.png", dpi='figure', format=None, metadata=None,
#        bbox_inches=None, pad_inches=0.1,
#       facecolor='white', edgecolor='auto')

# Instantiate Vietoris-Rips solver again if you wish to change the number of considered dimensions
#rips = Rips(maxdim = 2)

# some parameters
w = 20 # time window size
n = len(df_close)-(2*w)+1 # number of time segments. Here a whole business month was chosen
wasserstein_dists = np.zeros((n,1)) # initialize array for wasserstein distances

# compute wasserstein distances between persistence diagrams for subsequent time windows
for i in range(n):

    # Compute persistence diagrams for adjacent time windows
    dgm1 = rips.fit_transform(r[i:i+w])
    dgm2 = rips.fit_transform(r[i+w+1:i+(2*w)+1])
    
    # Compute wasserstein distance between diagrams
    wasserstein_dists[i] = persim.wasserstein(dgm1[0], dgm2[0], matching=False)
    
# plot wasserstein distances over time if you wish. Here they are plotted together with S&P 500 scaled prices
plt.figure(figsize=(18, 8), dpi=80)
plt.rcParams.update({'font.size': 16})

plt.plot(raw_data.index[w:n+w],wasserstein_dists)
plt.plot(raw_data.index[w:n+w],df_close.iloc[w:n+w,0]/max(df_close.iloc[w:n+w,0]))
plt.legend(['wasserstein distances', 'S&P 500 (scaled)', 'Crash of 2020'])
plt.xlabel('Date')
plt.title('Homology Changes')
plt.show()

# Save data
Data={"Date": df_close.index[w:n+w],
      "wasserstein_dists":wasserstein_dists.reshape(len(wasserstein_dists))}
df=pd.DataFrame(data=Data)
df.to_csv("wasserstein_dists.csv")
