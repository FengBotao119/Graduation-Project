import sys
sys.path.append('/Users/andrew/Desktop/Script/Graduation-Project')
from face_model.model_config import *
from face_model.models import *
from face_model.common import *
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.tensorboard import SummaryWriter  
from sklearn.model_selection import train_test_split 
import numpy as np
import pandas as pd 


data = pd.read_csv('./face_model/data/train_RAF_data.csv')
labels = data.label 
data = data.drop(['label'],axis=1)
X_train,X_valid, y_train, y_valid = train_test_split(data,labels,test_size=0.3,stratify = labels)
alpha = (sum(y_train.value_counts())/y_train.value_counts())/sum(sum(y_train.value_counts())/y_train.value_counts())
alpha = alpha.sort_index().values
train_data = pd.concat([X_train,y_train],axis=1)
valid_data = pd.concat([X_valid,y_valid],axis=1)


X_train,y_train = torch.Tensor(train_data[FEATURE_COLUMNS].values),torch.Tensor(train_data['label'].values)
X_valid,y_valid   = torch.Tensor(valid_data[FEATURE_COLUMNS].values),torch.Tensor(valid_data['label'].values)
train_set = TensorDataset(X_train,y_train)
valid_set  = TensorDataset(X_valid,y_valid)
train_loader = DataLoader(train_set,batch_size=5096,shuffle=True)
valid_loader  = DataLoader(valid_set,batch_size=5096,shuffle=True)

writer = SummaryWriter('./face_model/results/log')
model1 = NN(36,(16,32,16),7,0.5)
#model2 = ResidualNN(18,[(16,32,16,0.4),(16,64,16,0.4),(16,32,16,0.4)],7)
device = "cuda" if torch.cuda.is_available() else "cpu"
#criterion = nn.NLddLLoss()
criterion = FocalLoss_MultiLabel(7,alpha,2)
optimizer = optim.Adam(model1.parameters(),lr=0.1)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)
model1.to(device)
#model2.to(device)

train_results = {label:np.array([0,0,0]) for label in EXPRESSION_LABEL.values()}
valid_results = {label:np.array([0,0,0]) for label in EXPRESSION_LABEL.values()}
valid_losses = []
for epoch in range(EPOCH):
    train_loss = 0
    for index,(x,y) in enumerate(train_loader):
        model1.train()
        x = x.to(device)
        y = y.to(device)
        pre = model1(x)
        if epoch==EPOCH-1:
            result = P_R(pre,y)
            train_results = {label:train_results[label]+result[label] for label in EXPRESSION_LABEL.values()}
        loss = criterion(pre,y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    #每次实验记得更改名字
    writer.add_scalar("focal_bn_plain_loss/train", train_loss , global_step=epoch, walltime=None)
    print("epoch {} training loss {:.4f}".format(epoch,train_loss),end=' ')
    valid_loss=0
    for x_valid,y_valid in valid_loader:
        model1.eval()
        x_valid = x_valid.to(device)
        y_valid = y_valid.to(device)
        pre = model1(x_valid)
        if epoch==EPOCH-1:
            result = P_R(pre,y_valid)
            valid_results = {label:valid_results[label]+result[label] for label in EXPRESSION_LABEL.values()}
        loss = criterion(pre,y_valid.long())
        valid_loss+=loss.item()
    #每次实验记得更改名字
    writer.add_scalar("focal_bn_plain_loss/valid", valid_loss , global_step=epoch, walltime=None)
    print("valid loss {:.4f}".format(valid_loss))
    if epoch==EPOCH-1:
        train_results_precison = {label:round(value[0]/(value[1]+value[0]),3) for label,value in train_results.items()}
        train_results_recall   = {label:round(value[0]/(value[2]+value[0]),3) for label,value in train_results.items()}
        
        valid_results_precison = {label:round(value[0]/(value[1]+value[0]),3) for label,value in valid_results.items()}
        valid_results_recall   = {label:round(value[0]/(value[2]+value[0]),3) for label,value in valid_results.items()}
    
    if valid_losses and valid_loss>min(valid_losses):
        scheduler.step()
    valid_losses.append(valid_loss)

print("*"*30)
print("train precision:")
print(train_results_precison)
print("train recall:")
print(train_results_recall)
print("*"*30)
print("valid precision:")
print(valid_results_precison)
print("valid recall:")
print(valid_results_recall)