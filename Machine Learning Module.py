# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:01:29 2016

@author: Raymond
"""
#-------------------Data Analysis-------------------
import pandas as pd
import pandas_datareader.data as web
import numpy as np
#-------------------Machine Learning Package-------------------
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
#-------------------Classfier---------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#==============================================================================

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
        X_train = features_list[:Train_number]
        Y_train = predict_list[:Train_number]
        X_test = features_list[Test_number:]
        Y_test = predict_list[Test_number:]
        
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(features_list, predict_list, test_size=0.3)
    
    return X_train,Y_train,X_test,Y_test





class Machine_Learning(object):
    
    def __init__(self,df,Predict_label,Model,train_test_ratio,data_type):
        self.df = df.dropna()
        self.Model = Model
        self.Data =  self.Prepare_Data(self.df,train_test_ratio,Predict_label,data_type)
        #global Predict_model        
        self.Models(self.Data)


    def Prepare_Data(self,df,train_test_ratio,Predict_label,data_type = 'Time_Series'):
        features_df = df[df.columns.drop(Predict_label)]
        
        features_list = []
        for i in range(len(features_df)):
            features_list.append(features_df.ix[i].tolist())
            
        features_list = preprocessing.scale(features_list)
            
        predict_list = df[Predict_label].tolist()
    
        if len(features_list) == len(predict_list):
            Data_length = len(features_list)
        else:
            print 'Error with number of data.'
        
        if data_type == 'Time_Series':
            Train_number = int(Data_length * train_test_ratio)
            X_train = features_list[:Train_number]
            Y_train = predict_list[:Train_number]
            X_test = features_list[Train_number:]
            Y_test = predict_list[Train_number:]
            
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(features_list, predict_list, test_size= 1-train_test_ratio)
        
        return X_train,Y_train,X_test,Y_test

    def Models(self,data):
        if self.Model == 'SVM':
            model = SVC()
        elif self.Model == 'KNN':
            model = KNeighborsClassifier()
        else:
            print 'Error model input !!'

        model.fit(data[0],data[1])
        print model.score(data[2],data[3])
        
        global outsample_data        
        outsample_data = data[2],data[3]
        
        
        #Output predict model to global
        global Prediction_Model
        Prediction_Model = model
        
        









data = web.DataReader('^TWII','yahoo','2000-01-01')
data['past_return'] = data['Adj Close'].pct_change().shift(1)
data['Average_Return'] = pd.rolling_mean(data['Adj Close'].pct_change(),5)
data['Deviation'] = pd.rolling_mean(data['Adj Close'],5)/data['Adj Close'] - 1.
data['Volatility'] = pd.rolling_std(data['Adj Close'].pct_change(),5)

data = data.dropna()


Training = pd.DataFrame()
Training['Adj Close'] = data['Adj Close']
Training['past_return'] = data['past_return']
Training['Average_Return'] = data['Average_Return']
Training['Deviation'] = data['Deviation']
Training['Volatility'] = data['Volatility']

Training = Training.resample('W-WED',how = 'last')
Training['label'] = np.nan
Training['label'] = np.where(Training['Adj Close'].pct_change().shift(-1) < -0.01,1,0)
#Training['label'] = np.where(Training['Adj Close'].pct_change().shift(-1) < -0.01,-1,Training['label'])
#Training['label'].fillna(0)
del Training['Adj Close']







ML = Machine_Learning(Training,'label','SVM',0.6,data_type = 'Time_Series')

print Prediction_Model.predict(outsample_data[0][2:5])


