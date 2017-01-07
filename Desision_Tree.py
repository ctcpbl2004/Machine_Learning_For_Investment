# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 16:45:43 2017

@author: Raymond
"""

import pandas as pd
import pandas_datareader.data as web
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import subprocess
import matplotlib

matplotlib.style.use('ggplot')

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
#===============================================================================
def Prepare_Data(df,train_test_ratio,Predict_label,data_type = 'Time_Series'):
    features_df = df[df.columns.drop(Predict_label)]

    features_list = []
    for i in range(len(features_df)):
        features_list.append(features_df.ix[i].tolist())
    
    predict_list = df[Predict_label].tolist()

    if len(features_list) == len(predict_list):
        Data_length = len(features_list)
    else:
        print 'Error with number of data.'
    
    if data_type == 'Time_Series':
        Train_number = int(Data_length * train_test_ratio)
        Test_number = int(Data_length * (1-train_test_ratio))
        print Train_number,Test_number
        X_train = features_list[:Train_number]
        Y_train = predict_list[:Train_number]
        X_test = features_list[Train_number:]
        Y_test = predict_list[Train_number:]
        
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(features_list, predict_list, test_size=0.3)
    #print len(X_train),len(Y_train),len(X_test),len(Y_test)
    return X_train,Y_train,X_test,Y_test

Tickers = ['SPY','IWM','VGK','EWJ','EEM','SHY','IEF' ,'TLT' ,'TIP' ,'AGG' ,'HYG' ,'EMB' ,'VNQ' ,'RWX' ,'RWX' ,'GLD' ,'USO' ,'DBA' ]


#Shift_para = 20
#Frequency = 'MS'
#Ticker = '^GSPC'

def Decision_Tree_Prediction(Ticker,Frequency,Shift_para):
    Stock_raw = web.DataReader(Ticker,'yahoo','1950-01-01')
    
    Stock = Stock_raw.drop(['Open','High','Low','Close'],axis = 1)
    
    Stock['Return'] = Stock['Adj Close'].pct_change()
    #Stock['Max'] = pd.rolling_max(Stock['Adj Close'],Shift_para)
    #Stock['Min'] = pd.rolling_min(Stock['Adj Close'],Shift_para)
    #Stock['Change'] = Stock['Max'] - Stock['Min']
    Stock['Range'] = (pd.rolling_max(Stock['Adj Close'],Shift_para) - pd.rolling_min(Stock['Adj Close'],Shift_para))/Stock['Adj Close'].shift(Shift_para)
    Stock['Volatility']  = pd.rolling_std(Stock['Return'],Shift_para).diff()
    Stock['Momentum'] = Stock['Adj Close'].pct_change(Shift_para)#pd.rolling_sum(Stock['Return'],Shift_para)
    #Stock['MACD'] = (pd.rolling_mean(Stock['Adj Close'],12) - pd.rolling_mean(Stock['Adj Close'],26))/Stock['Adj Close']
    Stock['Volume_shock'] = pd.rolling_mean(Stock['Volume'],Shift_para) / pd.rolling_mean(Stock['Volume'],Shift_para*3)
    Stock['Bias'] = (Stock['Adj Close'] / pd.rolling_mean(Stock['Adj Close'],Shift_para) -1.)#.diff()
    Stock['Forward'] = np.where(Stock['Adj Close'].shift(-Shift_para) > Stock['Adj Close'],1,0)
    
    #Prepare features
    #Stock = Stock.dropna()
    Stock = Stock.resample(Frequency, how = 'first').dropna()
    
    del Stock['Adj Close']
    del Stock['Return']
    del Stock['Volume']
    
    #del Stock['Range']
    
    Train_Predict = Prepare_Data(df = Stock,train_test_ratio = 0.7,Predict_label = 'Forward',data_type = 'Time_Series')
    clf_tree = DecisionTreeClassifier()
    clf_tree.fit(Train_Predict[0],Train_Predict[1])
    #print Ticker, clf_tree.score(Train_Predict[2],Train_Predict[3])
    
    print Ticker + "'s predict accuracy = %0.4f" % clf_tree.score(Train_Predict[2],Train_Predict[3])
    
    
    
    Stock['Signal'] = np.nan
    Stock['Signal'][-len(clf_tree.predict(Train_Predict[2])):] = clf_tree.predict(Train_Predict[2])
    
    Signal_df = pd.DataFrame()
    Signal_df['Adj Close'] = Stock_raw['Adj Close']
    Signal_df['Signal'] = Stock['Signal'].dropna()
    Signal_df = Signal_df.resample('D', how = 'last').fillna(method = 'ffill')
    
    Signal_df = Signal_df.dropna()
    #Signal_df['Signal'] = Signal_df['Signal'].replace(0,-1)
    Signal_df['Return'] = Signal_df['Adj Close'].pct_change()
    Signal_df['Strategic_Return'] = Signal_df['Return'] * Signal_df['Signal'].shift(1) + 1.
    Signal_df['Buy & Hold'] = (Signal_df['Return'] +1.).cumprod()
    Signal_df['Decision_Tree_Strategy'] = Signal_df['Strategic_Return'].cumprod()
    #print Signal_df[['Decision_Tree_Strategy','Buy & Hold']]
    Signal_df[['Decision_Tree_Strategy','Buy & Hold']].plot()

#==============================================================================
Decision_Tree_Prediction('^GSPC','W-MON',5)



