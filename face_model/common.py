import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from face_model.model_config import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import  StandardScaler
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
    
    for i in range(7):
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
    if classname.find('Linear')!= -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0) 
    if classname.find('Embedding')!=-1:
        m.weight.data.normal_(0,0.01)  

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def prepare_data(train_df,test_df,mode,confidence_threshold,\
                train_batch_size,valid_batch_size):
    train_df = train_df[train_df[" confidence"]>confidence_threshold].copy()
    test_df  = test_df[test_df[" confidence"]>confidence_threshold].copy()
    continuous_cols = FEATURE_COLUMNS[:18]
    all_cols = FEATURE_COLUMNS
    column_idx = {col:index for index,col in enumerate(all_cols)}
    results = {}
    if mode == "Deep_Wide_nn":
        embeddings_input = [(col,2,7) for col in FEATURE_COLUMNS[18:]]
        categary_cols   = FEATURE_COLUMNS[18:]
        wide_cols = []
        for i in range(len(categary_cols)):
            for j in range(i+1,len(categary_cols)):
                new_col = categary_cols[i][1:]+"_"+categary_cols[j][1:]
                train_df[new_col] = train_df[categary_cols[i]]*train_df[categary_cols[j]]
                test_df[new_col] = test_df[categary_cols[i]]*test_df[categary_cols[j]]
                wide_cols.append(new_col)
        all_cols = FEATURE_COLUMNS+wide_cols
        column_idx = {col:index for index,col in enumerate(all_cols)}
        results['embeddings_input'] = embeddings_input
        results['column_idx'] = column_idx
        results['continuous_cols'] = continuous_cols
        results['categary_cols'] = categary_cols
        results['wide_cols'] = wide_cols

    labels   = train_df['label']
    train_df = train_df.drop(['label'],axis=1)

    X_train,X_valid, y_train, y_valid = train_test_split(train_df,labels,test_size=0.3,stratify = labels)
    alpha = (sum(y_train.value_counts())/y_train.value_counts())/sum(sum(y_train.value_counts())/y_train.value_counts())
    alpha = alpha.sort_index().values

    train_data = pd.concat([X_train,y_train],axis=1)
    valid_data = pd.concat([X_valid,y_valid],axis=1)

    X_train = train_data[all_cols].values
    X_valid = valid_data[all_cols].values
    X_test  = test_df[all_cols].values
    
    scaler = StandardScaler()
    temp_idx = [column_idx[col] for col in continuous_cols]
    scaler.fit(X_train[:,temp_idx])
    X_train[:,temp_idx] = scaler.transform(X_train[:,temp_idx])
    X_valid[:,temp_idx] = scaler.transform(X_valid[:,temp_idx])
    X_test[:,temp_idx]  = scaler.transform(X_test[:,temp_idx])
    
    X_train,Y_train = torch.Tensor(X_train), torch.Tensor(train_data['label'].values)
    X_valid,Y_valid = torch.Tensor(X_valid), torch.Tensor(valid_data['label'].values)
    X_test,Y_test   = torch.Tensor(X_test), torch.Tensor(test_df['label'].values)
    
    train_set = TensorDataset(X_train,Y_train)
    valid_set = TensorDataset(X_valid,Y_valid)
    train_loader = DataLoader(train_set,batch_size=train_batch_size,shuffle=True)
    valid_loader = DataLoader(valid_set,batch_size=valid_batch_size,shuffle=True)

    results['X_train'] = X_train
    results['Y_train'] = Y_train
    results['X_valid'] = X_valid
    results['Y_valid'] = Y_valid
    results['X_test'] = X_test
    results['Y_test'] = Y_test

    results['train_loader'] = train_loader
    results['valid_loader'] = valid_loader
    results['alpha']        = alpha

    return results