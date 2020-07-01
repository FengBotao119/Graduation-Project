"""Simple predictor using random forest
"""

import pandas as pd
import numpy as np
import math
#----------------------------------------------
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from common.metrics import ccc_score
#----------------------------------------------

#--------------rnn----------------------------
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
#--------------rnn---------------------------

from common.sql_handler import SqlHandler
import config
from global_values import *
from common.log_handler import get_logger
logger = get_logger()

def pre_data(df):
    df[np.isnan(df)] = 0.0
    df[np.isinf(df)] = 0.0
    return df

def netModel(shape1,shape2,dropout,reg):
    model = Sequential()
    model.add(keras.layers.Masking(mask_value=0,input_shape=(shape1,shape2)))
    model.add(keras.layers.GRU(64,dropout=dropout,kernel_regularizer=l2(reg))) 
    model.add(keras.layers.Dense(64,kernel_regularizer=l2(reg)))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.BatchNormalization(axis=-1))
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.Dense(1,kernel_regularizer=l2(reg)))
    RMSprop = keras.optimizers.RMSprop(lr=0.01,rho=0.8,decay=0.0)
    model.compile(optimizer=RMSprop,loss='mse',metrics=['mse'])   
    return model 

class RnnPredictor():
    def __init__(self, train, dev, test,features=None):
        """
        Input:
            train and dev are ndarray-like data
            features are the freature name in tran and dev
        """
        self.train_X ,self.train_Y = train
        self.dev_X,self.dev_Y = dev
        self.test_X,self.test_Y = test
        self.feature = features
        shape = self.train_X.shape[0]
        order = np.arange(0,shape)
        np.random.shuffle(order)
        self.train_X = self.train_X[order]
        self.train_Y = self.train_Y[order]
        self.train_X,self.train_Y,self.dev_X,self.dev_Y,self.test_X,self.test_Y = \
            pre_data(self.train_X),pre_data(self.train_Y)/25,pre_data(self.dev_X),pre_data(self.dev_Y)/25,pre_data(self.test_X),pre_data(self.test_Y)/25
                
    def train(self):
        path = 'E:/rnn_models/models/'
        self.model = Sequential()
        self.model.add(keras.layers.Masking(mask_value=0,input_shape=(self.train_X.shape[1],self.train_X.shape[-1])))
        self.model.add(keras.layers.GRU(64,dropout=0.5,kernel_regularizer=l2(5)))     
        self.model.add(keras.layers.Dense(64,kernel_regularizer=l2(5)))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.BatchNormalization(axis=-1))
        self.model.add(keras.layers.Activation('tanh'))
        self.model.add(keras.layers.Dense(1,kernel_regularizer=l2(5)))
        RMSprop = keras.optimizers.RMSprop(lr=0.01,rho=0.9,decay=0.0)
        self.model.compile(optimizer=RMSprop,loss='mse',metrics=['mse'])
        self.model.fit(self.train_X,self.train_Y,batch_size = 50,epochs=5,verbose =1,validation_data=[self.dev_X,self.dev_Y])
        self.model.save(path+self.feature+'.h5')
        
    def predict(self, X):
        y_pre = self.model.predict(X,batch_size=15)
        return y_pre

    def eval(self):
        #mae = mean_absolute_error(y_pre,self.dev_Y)
        #ccc = ccc_score(y_pre,self.dev_Y)
        #return {'MAE': mae, 'RMSE': rmse, 'CCC':ccc}
        train_y_pre = self.predict(self.train_X)
        train_y_pre.shape = self.train_Y.shape
        train_rmse = np.sqrt(mean_squared_error(train_y_pre,self.train_Y))*25

        dev_y_pre = self.predict(self.dev_X)
        dev_y_pre.shape =self.dev_Y.shape 
        dev_rmse = np.sqrt(mean_squared_error(dev_y_pre,self.dev_Y))*25

        test_y_pre = self.predict(self.test_X)
        test_y_pre.shape =self.test_Y.shape 
        test_rmse = np.sqrt(mean_squared_error(test_y_pre,self.test_Y))*25

        return {'train:':train_rmse,'dev:':dev_rmse,'test:':test_rmse,'dev_pre':dev_y_pre*25,'train_pre:':train_y_pre*25}


class MultiModalRnn():
    def __init__(self, data, features):
        """
        data and features is a dictionary that conatines data we need.
        """
        pass
