# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 15:02:11 2016

@author: Raymond
"""

import pandas as pd
import numpy as np
import pandas_datareader.data as web

Tickers = ['MMM','AXP','AAPL','BA','CAT','CVX','CSCO','KO','DIS','DD','XOM','GE',
           'GS','HD','IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE','PG',
           'TRV','UTX','UNH','VZ','WMT']

df = pd.DataFrame(columns = Tickers)

for Ticker in Tickers:
    df[Ticker] = web.DataReader(Ticker,'yahoo','1950-01-01')['Adj Close']
    #print web.DataReader(Ticker,'yahoo','1950-01-01')['Adj Close'].index[0],Ticker
df = df.dropna()

Return_df = df.pct_change().dropna()

Volatility_df = Return_df.resample('QS',how = np.std)*(250.**0.5)
Volatility_df_Rank = Volatility_df.rank(axis = 1)

for each in Volatility_df_Rank.columns:
    Volatility_df_Rank[each] = np.where(Volatility_df_Rank[each] <= len(Volatility_df_Rank.columns)/2.,1,0)
    
print Volatility_df_Rank
print Volatility_df_Rank.mean()

    
    
    













