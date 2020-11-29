from models import SVM,Randomforest
from sklearn.metrics import accuracy_score, recall_score, precision_score
import pandas as pd 
import numpy as np
from model_config import *
import joblib
import os


def Test(X,Y,models):
    pres = []
    for model in models:
        pres.append(model.predict(X))
    pre = np.array(pres).argmax(axis=0)
    recall = recall_score(Y,pre,average=None)
    precision = precision_score(Y,pre,average=None)
    accuracy  = accuracy_score(Y,pre)
    return recall,precision,accuracy


if __name__=='__main__':
    #---------------TRAIN STAGE-----------------------------------------
    datasets = [] 
    best_models = {label:None for label in EXPRESSION_LABEL.values()}
    for label in EXPRESSION_LABEL.values():
        data = pd.read_csv('./face_model/data/'+label+"_ovr_data.csv" )
        models = [SVM(),Randomforest()]
        x, y = data[FEATURE_COLUMNS],data[label]
        best_result = float('-inf')
        for model in models:
            model.gridsearch(x,y,5)
            auc = model.get_best_score
            param = model.get_best_param
            if auc>best_result:
                best_model_name = model.model_name
                best_model = model
                best_result = auc
                best_param = param
            print("class {} model {} param {} auc {:.4f}".format(label,model.model_name,param,auc))   
        best_models[label] = best_model  
        #print("class {} has been trained via {}, auc: {:.4f}".format(\
        #                    label, best_model_name, best_result))
        best_model.save(out_dir='./face_model/results/',file_name = label+"_"+best_model_name+".pkl")
    
    #----------------TEST STAGE------------------------------------------
    test_data = pd.read_csv('./face_model/data/ovr_test_data.csv')
    recall, precison, accuracy = Test(test_data[FEATURE_COLUMNS],test_data['label'],best_models.values())
    print("recall",{EXPRESSION_LABEL[str(i)]:round(recall[i],3) for i in range(7)})
    print("precision",{EXPRESSION_LABEL[str(i)]:round(precison[i],3) for i in range(7)})
    print("accuracy",accuracy)
