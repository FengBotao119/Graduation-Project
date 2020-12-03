from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NN(nn.Module):
    def __init__(self,feature_size,hidden_size,output_size,dropout):
        super().__init__()
        self.fc = nn.Linear(feature_size,hidden_size[0])
        self.h1 = nn.Linear(hidden_size[0],hidden_size[1])
        self.h2 = nn.Linear(hidden_size[1],hidden_size[2])

        self.output = nn.Linear(hidden_size[2],output_size)
        #self.dropout = nn.Dropout(dropout)
        
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])
        self.bn3 = nn.BatchNorm1d(hidden_size[2])

    
    def forward(self,x):
        #o1 = F.relu(self.fc(x))
        #o2 = F.relu(self.dropout(self.h1(o1)))
        #o3 = F.relu(self.h2(o2))
        o1 = torch.relu(self.bn1(self.fc(x)))
        o2 =  torch.relu(self.bn2(self.h1(o1)))
        o3 =  torch.relu(self.bn3(self.h2(o2)))

        output = self.output(o3)
        return output


class ResidualBlock(nn.Module):
    def __init__(self,paras):
        super().__init__()
        input_size,hidden_size,output_size,dropout = paras 
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        o1 = self.dropout(torch.relu(self.fc1(x)))
        o2 = self.fc2(o1)
        return x+o2
    

class ResidualNN(nn.Module):
    def __init__(self,feature_size,blocks,outout_size):
        super().__init__()
        self.fc1 = nn.Linear(feature_size,blocks[0][0])
        self.fc2 = nn.Linear(blocks[-1][-2],outout_size)
        self.sequencial = nn.Sequential()
        for i in range(len(blocks)):
            self.sequencial.add_module("block"+str(i),ResidualBlock(blocks[i]))
            
    
    def forward(self,x):
        o1 = torch.relu(self.fc1(x))
        o2 = torch.relu(self.sequencial(o1))
        o3 = self.fc2(o2)
        return o3


class FocalLoss_MultiLabel(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super().__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = torch.tensor(alpha)
        self.class_num = class_num
        self.size_average = size_average
        self.gamma = gamma
        
    def forward(self,inputs,targets):
        class_mask = torch.zeros_like(inputs)
        ids = targets.view(-1,1)
        class_mask.scatter_(1,ids,1)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        
        alpha = self.alpha[ids.data.view(-1)]
        P = torch.softmax(inputs,dim=1)
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss       

def P_R(pre,y_true):
    pre = pre.argmax(dim=1).detach().numpy()
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

MODELS = {"svm":(SVC(probability=True,random_state=123),{'kernel':['linear','poly','rbf','sigmoid'], 'C':[0.001, 0.1, 10]}),\
          "randomforest":(RandomForestClassifier(random_state=123),{'n_estimators':[50,100,150,200,250,300],'criterion':['gini','entropy'],'max_depth':[5,10,15,20,30]}),\
          "nn":(NN,{'hidden_size':[(32,64,128),(64,64,128),(32,32,64)],'dropout':[0.5,0.7,0.3]})}

MODEL_NAMES = ["svm",'randomforest','nn']

EXPRESSION_LABEL = {"0":"angry",
                    "1":"disgust",
                    "2":"fear",
                    "3":"happy",
                    "4":"sad",
                    "5":"surprise",
                    "6":"neutral"}

FEATURE_COLUMNS = ['confidence', ' AU01_r', ' AU02_r',
        ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r',
        ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r',
        ' AU25_r', ' AU26_r', ' AU45_r', ' AU01_c', ' AU02_c', ' AU04_c',
        ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c',
        ' AU14_c', ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c',
        ' AU26_c', ' AU28_c', ' AU45_c',]

#FEATURE_COLUMNS = ['confidence', ' AU01_r', ' AU02_r',
#        ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r',
#        ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r',
#        ' AU25_r', ' AU26_r', ' AU45_r']