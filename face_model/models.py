import pandas as pd
import numpy as np
import joblib
import os
import torch
from model_config import MODELS
from abc import ABC,abstractmethod
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

class ModelBase:
    @abstractmethod
    def gridsearch(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def save(self):
        pass
    

class SVM(ModelBase):
    def __init__(self):
        self.model_name = 'svm'
        self._model,self.param_grid = MODELS[self.model_name]

    def gridsearch(self,X,Y,n_folds):
        self._model = GridSearchCV(estimator=self._model,param_grid=self.param_grid,cv=n_folds,\
                                   scoring = "roc_auc", refit="roc_auc",n_jobs=-1)
        self._model.fit(X,Y)

    def predict(self,X):
        #输出为每一类的概率
        #按照每类降序排序
        return self._model.predict_proba(X)[:,1]

    def evaluate(self,X,Y):
        pre = self.predict(X)
        return roc_auc_score(Y,pre)

    def save(self,out_dir,file_name):
        joblib.dump(self._model,os.path.join(out_dir,file_name))

    @property
    def get_best_param(self):
        return self._model.best_params_
    
    @property
    def get_best_score(self):
        return self._model.best_score_


class Xgboost(ModelBase):
    def __init__(self):
        pass

    def gridsearch(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass


class Randomforest(ModelBase):
    def __init__(self):
        self.model_name = 'randomforest'
        self._model,self.param_grid = MODELS[self.model_name]

    def gridsearch(self,X,Y,n_folds):
        self._model = GridSearchCV(estimator=self._model,param_grid=self.param_grid,cv=n_folds,\
                                   scoring = "roc_auc", refit="roc_auc",n_jobs=-1)
        self._model.fit(X,Y)

    def predict(self,X):
        return self._model.predict_proba(X)[:,1]

    def evaluate(self,X,Y):
        pre = self.predict(X)
        return roc_auc_score(Y,pre)

    def save(self,out_dir,file_name):
        joblib.dump(self._model,os.path.join(out_dir,file_name))

    @property
    def get_best_param(self):
        return self._model.best_params_
    
    @property
    def get_best_score(self):
        return self._model.best_score_


class NN(ModelBase):
    def __init__(self):
        self.model_name = 'nn'
        self._model,self.param_grid = MODELS[self.model_name]
        self.devide = "cuda" torch.cuda.is_available() else "cpu"
        #self._model.to(self.devide)

    def gridsearch(self):
        
        

    def train(self,x,y):
        

    def predict(self):
        pass

    def evaluate(self):
        pass