# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 17:06:09 2016

@author: Raymond
"""

import pandas as pd
import numpy as np
import pandas_datareader.data as web
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

TWSE = web.DataReader('^TWII','yahoo','2008-01-01')
TWSE.resample('W-WED',how = 'last')

TWSE['diff'] = abs(TWSE['Adj Close'].diff())
TWSE['diff_shift'] = TWSE['diff'].shift(-1)
TWSE['Volatility_trend'] = np.where(TWSE['diff'] > TWSE['diff'].shift(1),1,0)
TWSE['X'] = TWSE['Volatility_trend'].shift(-1)
TWSE = TWSE.dropna()


Y = np.array(TWSE['Volatility_trend'].tolist(),dtype = float)
X = []
for i in range(len(TWSE['diff_shift'])):
    X.append(np.array([TWSE['diff_shift'][i]]))



X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3)


model = SVC()
model.fit(X_train,y_train)
print model.score(X_test,y_test)




