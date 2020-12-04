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