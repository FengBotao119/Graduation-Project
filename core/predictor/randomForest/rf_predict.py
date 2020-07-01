"""Simple predictor using random forest
"""

import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn import metrics
from sklearn.externals import joblib

from core.predictor.predictor import Predictor#
from common.sql_handler import SqlHandler
from common.metrics import ccc_score
import config
from global_values import *
from common.log_handler import get_logger
logger = get_logger()

def pre_data(df):
    df[np.isnan(df)] = 0.0
    df[np.isinf(df)] = 0.0
    return df

class RfPredictor(Predictor):
    def __init__(self, train, dev, test, features=None):
        """
        Input:
            train and dev are ndarray-like data
            features are the freature name in tran and dev
        """
        self.train_set = pre_data(train)
        self.dev_set = pre_data(dev)
        self.test_set = pre_data(test)
        self.feature_list = features

    def train(self):
        self.rf = RandomForestRegressor() #并行处理 
    
        train_X = self.train_set.iloc[:, 13:].values  
        train_y = self.train_set['PHQ8_Score'].values.ravel()  #将一行改为一列

        dev_X = self.dev_set.iloc[:, 13:].values
        dev_y = self.dev_set['PHQ8_Score'].values.ravel()

        X = np.vstack([train_X,dev_X])
        y = np.hstack([train_y,dev_y])

        val_fold = np.zeros(X.shape[0])   
        val_fold[:train_X.shape[0]] = -1 # 将训练集对应的index设为-1，表示永远不划分到验证集中
        ps = PredefinedSplit(test_fold=val_fold)
        paras = {'n_estimators':[10,20,30,50,100],\
                'max_features':[3,4,5],\
                'max_depth':np.arange(3,10),\
                'min_samples_split':np.arange(5,40,5),\
                'criterion':['mse'],\
                'n_jobs':[-1]}
        self.grid = GridSearchCV(estimator = self.rf,param_grid = paras,cv=ps,iid =False,n_jobs=-1)
        self.grid.fit(X, y)

        n_estimator = self.grid.best_estimator_.get_params()['n_estimators']
        n_feature = self.grid.best_estimator_.get_params()['max_features']
        depth = self.grid.best_estimator_.get_params()['max_depth']
        split = self.grid.best_estimator_.get_params()['min_samples_split']
        logger.info(f'n_estimator: {n_estimator}, n_feature: {n_feature}, max_depth: {depth}, min_samples_split: {split}')
        self.rf = RandomForestRegressor(n_jobs=-1,n_estimators=n_estimator,max_features = n_feature,criterion='mse',\
            max_depth=depth,min_samples_split=split)
        self.rf.fit(train_X, train_y)
        if X.shape[0]==304:
            joblib.dump(self.rf, "E:/rnn_models/models/rf_final_text.m")
        elif X.shape[0]==164:
            joblib.dump(self.rf, "E:/rnn_models/models/m_rf_final_text.m")
        else:
            joblib.dump(self.rf, "E:/rnn_models/models/f_rf_final_text.m")
        

    def predict(self, X):
        y = self.rf.predict(X)
        return y

    def eval(self):
        X = self.train_set.iloc[:, 13:].values
        y = self.train_set['PHQ8_Score'].values
        y_pred = self.predict(X)
        mae = metrics.mean_absolute_error(y,y_pred)
        rmse = math.sqrt(metrics.mean_squared_error(y,y_pred))
        ccc = ccc_score(y, y_pred)
        train_r = [rmse]

        X = self.dev_set.iloc[:, 13:].values
        y = self.dev_set['PHQ8_Score'].values
        y_pred = self.predict(X)
        mae = metrics.mean_absolute_error(y,y_pred)
        rmse = math.sqrt(metrics.mean_squared_error(y,y_pred))
        ccc = ccc_score(y, y_pred)
        dev_r = [rmse]

        X = self.test_set.iloc[:, 5:].values
        y = self.test_set['PHQ_Score'].values
        y_pred = self.predict(X)
        mae = metrics.mean_absolute_error(y,y_pred)
        rmse = math.sqrt(metrics.mean_squared_error(y,y_pred))
        ccc = ccc_score(y, y_pred)
        test_r = [rmse]

        return {'train':train_r,'dev':dev_r,'test':test_r}

        # fea_importance = self.rf.feature_importances_
        # fea_imp_dct = {fea:val for fea, val in zip(self.feature_list, fea_importance)}
        # top = sorted(fea_imp_dct, key=lambda x: fea_imp_dct[x], reverse=True)[:5]
        # top_fea = {fea: fea_imp_dct[fea] for fea in top}
        # return {'MAE': mae, 'RMSE': rmse, 'CCC':ccc,'features_importance':top_fea}

class MultiModalRandomForest(Predictor):
    def __init__(self, data, features):
        """
        data and features is a dictionary that conatines data we need.
        """
        self.audio_train,self.audio_dev,self.audio_fea = pre_data(data['audio_train']),pre_data(data['audio_dev']),features['audio']
        self.vedio_train,self.vedio_dev,self.vedio_fea = pre_data(data['vedio_train']),pre_data(data['vedio_dev']),features['vedio']
        self.text_train,self.text_dev, self.text_fea = pre_data(data['text_train']),pre_data(data['text_dev']),features['text']

    def train(self):
        X_audio = self.audio_train.iloc[:, 13:].values  #
        X_vedio = self.vedio_train.iloc[:, 13:].values
        X_text = self.text_train.iloc[:, 13:].values
        y = self.audio_train['PHQ8_Score'].values.ravel()  #三个module的y都是一样的

        model = RandomForestRegressor()
        paras = {'n_estimators':[10,20,30,50,100],\
           'criterion':['mse'],\
                'n_jobs':[-1]}
    
        self.rf_audio = GridSearchCV(estimator = model,param_grid = paras,cv=5,iid =False,n_jobs=-1)
        self.rf_audio.fit(X_audio, y)

        self.rf_vedio = GridSearchCV(estimator = model,param_grid = paras,cv=5,iid =False,n_jobs=-1)
        self.rf_vedio.fit(X_vedio, y)

        self.rf_text = GridSearchCV(estimator = model,param_grid = paras,cv=5,iid =False,n_jobs=-1)
        self.rf_text.fit(X_text, y)

    def predict(self,X,module):
        if module == 'audio':
            return self.rf_audio.predict(X)
        elif module =='vedio':
            return self.rf_vedio.predict(X)
        else:
            return self.rf_text.predict(X)
        
    def eval(self):
        self.train()

        X_audio = self.audio_dev.iloc[:,13:].values
        X_vedio = self.vedio_dev.iloc[:,13:].values
        X_text = self.text_dev.iloc[:,13:].values
        y = self.audio_dev['PHQ8_Score'].values.ravel()#三个module的y都是一样的
    

        y_audio_pred = self.predict(X_audio,'audio')
        y_vedio_pred = self.predict(X_vedio,'vedio')
        y_text_pred = self.predict(X_text,'text')

        y_pred = (y_audio_pred+y_vedio_pred+y_text_pred)/3

        mae = metrics.mean_absolute_error(y,y_pred)
        rmse = math.sqrt(metrics.mean_squared_error(y,y_pred))

        ccc = ccc_score(y, y_pred)

        return {'MAE': mae, 'RMSE': rmse, 'CCC':ccc}