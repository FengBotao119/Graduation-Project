import pandas as pd
import numpy as np
import joblib
import os
from model_config import MODELS
from abc import ABC,abstractmethod
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

class ModelBase:
    @abstractmethod
    def gridsearch(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def save(self):
        pass
    
    @abstractmethod
    def get_best_param(self):
        pass

class Model(ModelBase):
    _models = MODELS

    def __init__(self,model_name):
        self._model,self.param_grid = self._models[model_name]

    def gridsearch(self,X,Y,n_folds):
        self._model = GridSearchCV(estimator=self._model,param_grid=self.param_grid,cv=n_folds,\
                                   scoring =("accuracy","recall","precision","f1") ,refit="f1",n_jobs=-1)
        self._model.fit(X,Y)

    def evaluate(self,X,Y):
        pre = self._model.predict(X)
        accuracy = accuracy_score(Y,pre)
        recall   = recall_score(Y,pre)
        precision= precision_score(Y,pre)
        f1       = f1_score(Y,pre)
        return accuracy,recall,precision,f1

    def save(self,out_dir,file_name):
        joblib.dump(self._model,os.path.join(out_dir,file_name))

    @property
    def get_best_param(self):
        return self._model.best_params_
    
    @property
    def get_best_score(self):
        return self._model.best_score_

