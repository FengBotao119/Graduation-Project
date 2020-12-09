import sys
import time
import os
sys.path.append(os.getcwd())
from face_model.model_config import *
from face_model.models import *
from face_model.common import *
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.tensorboard import SummaryWriter  
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch',type=int,default=50)
parser.add_argument('--confidence_threshold',type=float,default=0.8)
parser.add_argument('--lr',type=float,default=0.1)
parser.add_argument('--lr_step',type=float,default=0.9)
parser.add_argument('--weight_decay',type=float,default=0.)
parser.add_argument('--gamma',type=int,default=1)
parser.add_argument('--topk',type=int,default=2)
parser.add_argument('--train_batch_size',type=int,default=2048)
parser.add_argument('--valid_batch_size',type=int,default=1024)
args = parser.parse_args()


data = pd.read_csv("./face_model/data/train_RAF_data_sub_happy.csv")
data = data[data[" confidence"]>args.confidence_threshold]
#data.label = [7 if data.loc[i," confidence"]<CONFIDENCE_THRESHOLD else data.loc[i,"label"] for i in range(data.shape[0])]

labels = data.label 
data = data.drop(['label'],axis=1)
X_train,X_valid, y_train, y_valid = train_test_split(data,labels,test_size=0.3,stratify = labels)
alpha = (sum(y_train.value_counts())/y_train.value_counts())/sum(sum(y_train.value_counts())/y_train.value_counts())
#alpha = sum(y_train.value_counts())/y_train.value_counts()
#alpha = None
alpha = alpha.sort_index().values
train_data = pd.concat([X_train,y_train],axis=1)
valid_data = pd.concat([X_valid,y_valid],axis=1)


X_train,Y_train = torch.Tensor(train_data[FEATURE_COLUMNS].values),torch.Tensor(train_data['label'].values)
X_valid,Y_valid   = torch.Tensor(valid_data[FEATURE_COLUMNS].values),torch.Tensor(valid_data['label'].values)
train_set = TensorDataset(X_train,Y_train)
valid_set  = TensorDataset(X_valid,Y_valid)
train_loader = DataLoader(train_set,batch_size=args.train_batch_size,shuffle=True)
valid_loader  = DataLoader(valid_set,batch_size=args.valid_batch_size,shuffle=True)

writer = SummaryWriter('./face_model/results/log')
model1 = NN(36,(32,128,256,512,256,128,32),7,None)
#model1 = ResidualNN(35,[(16,32,16,0.4),(16,32,16,0.4),(16,64,16,0.4),(16,32,16,0.4)],8)
device = "cuda" if torch.cuda.is_available() else "cpu"
#criterion = nn.CrossEntropyLoss()
criterion = FocalLoss_MultiLabel(7,alpha,args.gamma)
optimizer = optim.Adam(model1.parameters(),lr=args.lr,weight_decay = args.weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_step)
model1.to(device)
model1.apply(weights_init_normal)
#model2.to(device)

#train_results = {label:np.array([0,0,0]) for label in EXPRESSION_LABEL.values()}
#valid_results = {label:np.array([0,0,0]) for label in EXPRESSION_LABEL.values()}
valid_losses = []
model_name = str(args.lr)+"_"+str(args.lr_step)+"_"+str(args.gamma)+"_"+str(args.weight_decay)+"_"+"best_model"
for epoch in range(args.epoch):
    train_loss = 0
    for index,(x,y) in enumerate(train_loader):
        model1.train()
        x = x.to(device)
        y = y.to(device)
        pre = model1(x)
        #if epoch==EPOCH-1:
        #    result = P_R(pre,y)
        #    train_results = {label:train_results[label]+result[label] for label in EXPRESSION_LABEL.values()}
        loss = criterion(pre,y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    #每次实验记得更改名字
    writer.add_scalar(model_name+"_loss/train", train_loss , global_step=epoch, walltime=None)
    print("epoch {} training loss {:.4f}".format(epoch,train_loss),end=' ')
    valid_loss=0
    for x_valid,y_valid in valid_loader:
        model1.eval()
        x_valid = x_valid.to(device)
        y_valid = y_valid.to(device)
        pre = model1(x_valid)
        #if epoch==EPOCH-1:
        #    result = P_R(pre,y_valid)
        #    valid_results = {label:valid_results[label]+result[label] for label in EXPRESSION_LABEL.values()}
        loss = criterion(pre,y_valid.long())
        valid_loss+=loss.item()
    #每次实验记得更改名字
    writer.add_scalar(model_name+"_loss/valid", valid_loss , global_step=epoch, walltime=None)
    print("valid loss {:.4f}".format(valid_loss))
    #if epoch==EPOCH-1:
    #    train_results_precison = {label:round(value[0]/(value[1]+value[0]),3) for label,value in train_results.items()}
    #    train_results_recall   = {label:round(value[0]/(value[2]+value[0]),3) for label,value in train_results.items()}
        
    #    valid_results_precison = {label:round(value[0]/(value[1]+value[0]),3) for label,value in valid_results.items()}
    #   valid_results_recall   = {label:round(value[0]/(value[2]+value[0]),3) for label,value in valid_results.items()}
    
    if valid_losses and valid_loss>min(valid_losses):
        scheduler.step()
    if valid_losses and valid_loss<min(valid_losses):
        localtime = time.localtime(time.time())
        #model_name = "_".join([str(localtime.tm_mon),str(localtime.tm_mday),str(localtime.tm_hour),str(localtime.tm_min)])
        torch.save(model1.state_dict(),"./face_model/results/"+model_name+".pkl")
    valid_losses.append(valid_loss)

model = NN(36,(32,128,256,512,256,128,32),7,0.2)
model.load_state_dict(torch.load('./face_model/results/'+model_name+'.pkl'))
model.eval()

train_y_pre = model(X_train)
train_cm = cm(train_y_pre,Y_train,args.topk)

valid_y_pre = model(X_valid)
valid_cm = cm(valid_y_pre,Y_valid,args.topk)

print("*"*100)
print(train_cm)
print("*"*100)
print(valid_cm)
print(optimizer)

#print("*"*60)
#print("train precision:")
#print(train_results_precison)
#print("train recall:")
#print(train_results_recall)
#print("*"*60)
#print("valid precision:")
#print(valid_results_precison)
#print("valid recall:")
#print(valid_results_recall)

#mP = sum([0 if label=="Negtive" else valid_results_precison[label] for label in EXPRESSION_LABEL.values()])/7
#mA = sum([0 if label=="Negtive" else valid_results_recall[label] for label in EXPRESSION_LABEL.values()])/7

#print("Average precision",round(mP,4))
#print("Average recall",round(mA,4))


