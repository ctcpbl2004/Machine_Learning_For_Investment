# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:25:30 2016

@author: Raymond
"""

import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime
import matplotlib

matplotlib.style.use('ggplot')




def Portfolio(Tickers,start = '2000-01-01'):
    df = pd.DataFrame(columns = Tickers, index = pd.date_range(start = start, end = datetime.datetime.today()))    
    
    for each in Tickers:
        df[each] = web.DataReader(str(each),'yahoo',start)['Adj Close']
    
    return df.dropna()


def Performance_Comparision(df,Period):
    
    def Performance_Calculate(Series,Period):
        Period_Close = Series.resample(Period,how = 'last')
        
        Period_Return = (Period_Close/Period_Close.shift(1) - 1).ix[:-1]
        Period_Return.name = 'Return'
        
        Period_Volatility = ((Series/Series.shift(1) - 1).resample(Period,how = np.std)*250.**0.5).ix[:-1]
        Period_Volatility.name = 'Volatility'
        
        Mean_Variance_df = pd.concat([Period_Return,Period_Volatility],axis = 1)

        return Mean_Variance_df
    
    def Equity_Curve(df):
        Cumulative_Return = df/df.ix[0] - 1
        Cumulative_Return.plot()
    
    
    
    

    if Period == 'Monthly'or Period == 'monthly' or Period == 'Month' or Period == 'month' or Period == 'M' or Period == 'm':
        Settlement_Period = 'M'
    elif Period == 'Quarterly' or Period == 'quarterly' or Period == 'Quarter' or Period == 'quarter' or Period == 'Q' or Period == 'q':
        Settlement_Period = 'Q'
    else:
        print 'Error !! You got the wrong period input.'
   
    color_list = ['blue','red','green','yellow','black','cyan','magenta']    
    
    
    count = 0
    for each in df.columns:
        Performance = Performance_Calculate(Series = df[each],Period = Settlement_Period) * 100.
        plt.scatter(x = Performance['Volatility'], y = Performance['Return'], color = color_list[count],label = each, lw = 2)
        count = count + 1
        

    plt.xlabel('Volatility(%)', fontsize = 20)

    plt.ylabel('Return(%)', fontsize = 20)
    plt.legend()
    plt.show()    
    Equity_Curve(df)
    
    
    
    
Data = Portfolio(Tickers = ['SPY','EEM','VNQ','SHY'],start = '2000-01-01')
  
Performance_Comparision(Data, 'M')


'''
Settlement_Period = 'M'
Period_Close = data.resample(Settlement_Period,how = 'last')
Period_Return = (Period_Close/Period_Close.shift(1) - 1).ix[:-1]
Period_Return.name = 'Return'
Period_Volatility = ((data/data.shift(1) - 1).resample(Settlement_Period,how = np.std)*250.**0.5).ix[:-1]
Period_Volatility.name = 'Volatility'
Mean_Variance_df = pd.concat([Period_Return,Period_Volatility],axis = 1)

plt.scatter(x = Mean_Variance_df['Volatility'], y = Mean_Variance_df['Return'])
plt.show()

'''


