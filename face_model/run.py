from models import SVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
import pandas as pd 
import numpy as np
from model_config import *

"""
呆解决问题：二分类需要调整阈值
"""

def Test(X,Y,models):
    pres = []
    for model in models:
        pres.append(model.predict(X))
    pre = np.array(pres).argmax(axis=1)
    recall = recall_score(Y,pre,average=None)
    precision = precision_score(Y,pre,average=None)
    accuracy  = accuracy_score(Y,pre)
    return pre,recall,precision,accuracy

def Train():
    pass

if __name__=='__main__':
    datasets = [] 
    #models = [SVM()]
    for label in EXPRESSION_LABEL.values():
        data = pd.read_csv('./face_model/data/'+label+"_ovr_data.csv" )
        X, Y = data.drop(['label'],axis=1), data['label']
        data_train,data_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,stratify=Y)

        x, y = data_train[FEATURE_COLUMNS], data_train[label]
        x_test, y_test = data_test[FEATURE_COLUMNS], data_test[label]

        best_result = float('-inf')
        #for model in models:
        model = SVM()
        model.gridsearch(x,y,5)
        auc = model.evaluate(x_test,y_test)
        if auc>best_result:
            best_model_name = model.model_name
            best_model = model
            best_result = auc
                    
        print("class {} has been trained via {}, auc: {:.4f}".format(\
                            label, best_model_name, best_result))
        best_model.save(out_dir='./face_model/results/',file_name = label+"_"+best_model_name+".pkl")

