import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NN(nn.Module):
    def __init__(self,feature_size,hidden_size,output_size,dropout):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size  = hidden_size 
        self.dropout      = dropout

        self.hidden_0 = nn.Linear(feature_size,hidden_size[0])
        if self.dropout:
            self.h0_dropout = nn.Dropout(self.dropout[0])
        for i,_ in enumerate(hidden_size[1:],1):
            setattr(self,"hidden_"+str(i),nn.Linear(hidden_size[i-1],hidden_size[i]))
            if self.dropout:
                setattr(self,"h"+str(i)+'_dropout',nn.Dropout(self.dropout[i]))
        for i,_ in enumerate(hidden_size):
            setattr(self,'bn'+str(i),nn.BatchNorm1d(hidden_size[i]))
        self.output = nn.Linear(hidden_size[-1],output_size)

    def forward(self,x):
        for i in range(len(self.hidden_size)):
            x = getattr(self,"hidden_"+str(i))(x)
            x = torch.relu(getattr(self,"bn"+str(i))(x))
            if self.dropout:
                x = getattr(self,"h"+str(i)+"_dropout")(x)
        output = self.output(x)
        return output


class Deep_Wide_NN(nn.Module):
    def __init__(self,wide_dim,embeddings_input,deep_column_idx,continuous_cols,hidden_layers,dropout,n_class):
        super().__init__()
        self.wide_dim = wide_dim
        self.deep_column_idx = deep_column_idx
        self.embeddings_input = embeddings_input
        self.continuous_cols = continuous_cols
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.n_class = n_class

        for col,val,dim in self.embeddings_input:
            setattr(self,"emb_"+col,nn.Embedding(val,dim,padding_idx=0))
        
        input_emb_dim = sum([emb[2] for emb in self.embeddings_input])
        self.linear_1 = nn.Linear(input_emb_dim+len(continuous_cols), self.hidden_layers[0])
        if self.dropout:
            self.linear_1_drop = nn.Dropout(self.dropout[0])
        for i,_ in enumerate(self.hidden_layers[1:],1):
            setattr(self, 'linear_'+str(i+1), nn.Linear( self.hidden_layers[i-1], self.hidden_layers[i] ))
            if self.dropout:
                setattr(self, 'linear_'+str(i+1)+'_drop', nn.Dropout(self.dropout[i]))

        self.output = nn.Linear(self.hidden_layers[-1]+self.wide_dim, self.n_class)

    def forward(self,X_w,X_d):
        emb = [getattr(self, 'emb_'+col)(X_d[:,self.deep_column_idx[col]].long())
               for col,_,_ in self.embeddings_input]
        if self.continuous_cols:
            cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
            cont = [X_d[:, cont_idx].float()]
            deep_inp = torch.cat(emb+cont, 1)
        else:
            deep_inp = torch.cat(emb,1)

        x_deep = F.relu(self.linear_1(deep_inp))
        if self.dropout:
            x_deep = self.linear_1_drop(x_deep)
        for i in range(1,len(self.hidden_layers)):
            x_deep = F.relu( getattr(self, 'linear_'+str(i+1))(x_deep) )
            if self.dropout:
                x_deep = getattr(self, 'linear_'+str(i+1)+'_drop')(x_deep)
        if X_w:
            wide_deep_input = torch.cat([x_deep, X_w.float()], 1)
        else:
            wide_deep_input = x_deep
        out = self.output(wide_deep_input)
        
        return out


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


class DeepandWideNN(nn.Module):
    pass


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