from models import Model
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
from model_config import *

"""
read data->split data->train->test->save

"""

def Train(model,X,Y,n_folds,out_dir,file_name):
    model.gridsearch(X,Y,n_folds)
    model.save(out_dir,file_name)
    return model,model.get_best_param,model.get_best_score

def Evaluate(model,X,Y):
    return model.evaluate(X,Y)

def predict():
    pass


if __name__=='__main__':
    datasets = [] 
    for label in EXPRESSION_LABEL.values():
        data = pd.read_csv('./face_model/data/'+label+"_ovr_data.csv" )
        X, Y = data.drop(['label'],axis=1), data['label']
        data_train,data_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,stratify=Y)

        x, y = data_train[FEATURE_COLUMNS], data_train[label]
        x_test, y_test = data_test[FEATURE_COLUMNS], data_test[label]

        best_result = float('-inf')
        for name in MODEL_NAMES:
            model = Model(name)
            model.gridsearch(x,y,5)
            accuracy,recall,precision,f1 = model.evaluate(x_test,y_test)
            if f1>best_result:
                best_model_name = name
                best_model = model
                best_accuracy, best_recall, best_precision, best_f1 = accuracy,recall,precision,f1
        print("class {} has been trained, f1: {:.4f}, recall: {:.4f}, precision: {:.4f}".format(\
                            label,best_model.get_best_score,best_recall,best_precision))
        best_model.save(out_dir='./face_model/results/',file_name = label+"_"+best_model_name+".pkl")

