import numpy as np
from face_model.model_config import *

def P_R(pre,y_true):
    pre = pre.argmax(dim=1).detach().cpu().numpy()
    y_true = y_true.int().numpy()
    one_hot_pre = np.zeros([pre.shape[0],7])
    one_hot_y   = np.zeros([pre.shape[0],7])
    for i in range(pre.shape[0]):
        one_hot_y[i][y_true[i]] = 1
        one_hot_pre[i][pre[i]]  = 1
    result = {value:None for value in EXPRESSION_LABEL.values()}
    
    for i in range(7):
        TP = sum(np.logical_and(one_hot_pre[:,i]==1,one_hot_y[:,i]==1))
        FP = sum(np.logical_and(one_hot_pre[:,i]==1,one_hot_y[:,i]==0))
        FN = sum(np.logical_and(one_hot_pre[:,i]==0,one_hot_y[:,i]==1))
        result[EXPRESSION_LABEL[str(i)]] = np.array([TP,FP,FN])
    return result

