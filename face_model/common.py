import numpy as np
import pandas as pd
from face_model.model_config import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
import random

def P_R(pre,y_true):
    pre = pre.argmax(dim=1).detach().cpu().numpy()
    y_true = y_true.int().cpu().numpy()
    one_hot_pre = np.zeros([pre.shape[0],7])
    one_hot_y   = np.zeros([pre.shape[0],7])
    for i in range(pre.shape[0]):
        one_hot_y[i][y_true[i]] = 1
        one_hot_pre[i][pre[i]]  = 1
    result = {value:None for value in EXPRESSION_LABEL.values()}
    
    for i in range(8):
        TP = sum(np.logical_and(one_hot_pre[:,i]==1,one_hot_y[:,i]==1))
        FP = sum(np.logical_and(one_hot_pre[:,i]==1,one_hot_y[:,i]==0))
        FN = sum(np.logical_and(one_hot_pre[:,i]==0,one_hot_y[:,i]==1))
        result[EXPRESSION_LABEL[str(i)]] = np.array([TP,FP,FN])
    return result

def cm(pre,y_true,topK):
    _,top = pre.detach().sort()
    top = top.cpu().numpy()
    top = top[:,-topK:]
    y_true = y_true.int().cpu().numpy() 
    pre = [y_true[i] if y_true[i] in top[i,:] else top[i,-1] for i in range(top.shape[0])]
    cm = confusion_matrix(y_true,pre)
    columns = [label for label in EXPRESSION_LABEL.values()]
    cm = pd.DataFrame(cm,columns=columns,index=columns)
    acc = round(sum([cm.iloc[i,i] for i in range(7)])/cm.values.sum(),3)
    recall = [str(round(cm.iloc[i,i]/cm.iloc[i,:].sum(),3)) for i in range(7)]
    precision = pd.DataFrame([[str(round(cm.iloc[i,i]/cm.iloc[:,i].sum(),3)) for i in range(7)]],columns=cm.columns,index=['precision'])
    cm['recall'] = recall
    cm = cm.append(precision)
    cm.iloc[-1,-1] = acc
    return cm

def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)       

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def prepare_date(train_df,test_df,mode):
    if mode == "plant_nn":
        pass
    elif mode == 'Deep_Wide_nn':
        
    else:
        print("模型正在开发中......")