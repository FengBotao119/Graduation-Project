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

setup_seed(123)

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
parser.add_argument('--mode',type=str,default='nn')
parser.add_argument('--loss',type=str,default='CrossEntropyLoss')

args = parser.parse_args()

train_df = pd.read_csv("./face_model/data/train_RAF_data.csv")
test_df  = pd.read_csv("./face_model/data/test_RAF_data.csv")
results = prepare_data(train_df,test_df,args.mode,args.confidence_threshold,\
                        args.train_batch_size,args.valid_batch_size)
train_loader = results['train_loader']
valid_loader = results['valid_loader']
alpha = results['alpha']
hidden_layers    = [64,128,32]
dropout = None
n_class = 7

if args.mode == 'Deep_Wide_nn':
    embeddings_input = results['embeddings_input']
    column_idx  = results['column_idx']
    continuous_cols  = results['continuous_cols']
    categary_cols  = results['categary_cols']
    wide_cols  = results['wide_cols']
    model1 = Deep_Wide_NN(embeddings_input,column_idx,continuous_cols,categary_cols,\
                        wide_cols,hidden_layers,dropout,n_class)
    model_final = Deep_Wide_NN(embeddings_input,column_idx,continuous_cols,categary_cols,\
                        wide_cols,hidden_layers,dropout,n_class)

elif args.mode == 'nn':
    model1 = NN(36,hidden_layers,n_class,dropout)
    model_final = NN(36,hidden_layers,n_class,dropout)

else:
    print("该模型正在开发中......")

if args.loss == 'FocalLoss':
    criterion = FocalLoss_MultiLabel(7,alpha,args.gamma)

elif args.loss == 'CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss()

else:
    print("请正确输入损失函数类型.....")

writer = SummaryWriter('./face_model/results/log')
device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = optim.Adam(model1.parameters(),lr=args.lr,weight_decay = args.weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_step)
model1.to(device)
model1.apply(weights_init_normal)

valid_losses = []
model_name = args.mode+"_"+args.loss+"_"+str(args.train_batch_size)+"_"+str(args.confidence_threshold)+"_"+\
            str(args.lr)+"_"+str(args.lr_step)+"_"+str(args.gamma)+"_"+str(args.weight_decay)+"_"+"best_model"
for epoch in range(args.epoch):
    train_loss = 0
    if epoch==args.epoch//2:
        optimizer = optim.Adam(model1.parameters(),lr=args.lr,weight_decay = args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_step)
    for index,(x,y) in enumerate(train_loader):
        model1.train()
        x = x.to(device)
        y = y.to(device)
        pre = model1(x)
        loss = criterion(pre,y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    writer.add_scalar(model_name+"_loss/train", train_loss , global_step=epoch, walltime=None)
    print("epoch {} training loss {:.4f}".format(epoch,train_loss),end=' ')
    valid_loss=0
    for x_valid,y_valid in valid_loader:
        model1.eval()
        x_valid = x_valid.to(device)
        y_valid = y_valid.to(device)
        pre = model1(x_valid)
        loss = criterion(pre,y_valid.long())
        valid_loss+=loss.item()
    writer.add_scalar(model_name+"_loss/valid", valid_loss , global_step=epoch, walltime=None)
    print("valid loss {:.4f}".format(valid_loss),end=" ")
    print("learning rate is {:.7f}".format(optimizer.state_dict()['param_groups'][0]['lr']))
    if valid_losses and valid_loss>min(valid_losses):
        scheduler.step()
    if valid_losses and valid_loss<min(valid_losses):
        localtime = time.localtime(time.time())
        torch.save(model1.state_dict(),"./face_model/results/"+model_name+".pkl")
    valid_losses.append(valid_loss)

model_final.load_state_dict(torch.load('./face_model/results/'+model_name+'.pkl'))
model_final.eval()

train_y_pre = model_final(results['X_train'])
train_cm = cm(train_y_pre,results['Y_train'],args.topk)

valid_y_pre = model_final(results['X_valid'])
valid_cm = cm(valid_y_pre,results['Y_valid'],args.topk)

test_y_pre = model_final(results['X_test'])
test_cm = cm(test_y_pre,results['Y_test'],args.topk)

print("*"*50+"TRAIN"+"*"*50)
print(train_cm)
print("*"*50+"VALID"+"*"*50)
print(valid_cm)
print("*"*50+"TEST"+"*"*50)
print(test_cm)


