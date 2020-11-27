"""
One VS One
One VS Rest
Error-Correcting Output-Codes

input: training set:

OVO output: class1_2_trainset, class1_3_trainset...
OVR output: class1_trainset, class2_trainset, class3_trainset...

"""
import pandas as pd 
from model_config import *
import argparse

def strategy(data,mode):
    classes = data['label'].unique()
    if mode=='ovr':
        for class_ in classes:
            data[EXPRESSION_LABEL[str(class_)]] = [1 if value==class_ else 0 for value in data['label']]
    else:
        pass
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',help="mode to split data",type=str,choices=['ovr','ovo'])
    args = parser.parse_args()

    data = pd.read_csv('./data/data.csv')
    data = strategy(data,args.mode)
    data.to_csv('./data/'+ args.mode +'_data.csv')

