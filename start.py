import sys
from global_values import *
#import librosa #暂时未用
from core.feature_extraction.data_to_db import data_set 
from core.feature_extraction import extract #
from core.predictor import training
#import tensorflow as tf
#读取数据
import config    
config.init()
#记录运行过程
from common import log_handler
logger = log_handler.get_logger()
import argparse

def cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',help = 'feature extraction or training',type=str,choices=['extraction','train'])
    parser.add_argument('--feature',help ='choose which feature to extract',type=str)
    parser.add_argument('--feature_tables',help='choose the feature to train model using table name',nargs='+')
    parser.add_argument('--model',help='choose which predictor to train',type=str)
    parser.add_argument('--gender',help='whether considering gender',type=str,choices=['yes','no'])

    args=parser.parse_args()

    if args.mode == 'extraction':
        fea_ext = extract.FeatureExtration(feature=args.feature)
        fea_ext.gen_fea()

    elif args.mode == 'train':
        if args.feature is not None:
            fea = args.feature
            model = args.model
            if args.gender == 'yes':
                logger.info(f'You are training using model {model} via feature {fea} and considering gender!')
                #training.Train(model_name=model, feature_name=fea, gender=True).start()
                training.Train(model_name=model, feature_name=fea, gender=True).run()
            else:
                logger.info(f'You are training using model {model} via feature {fea}')
                train= training.Train(model_name=model, feature_name=fea)
                #train.start() #有问题 待解决
                train.run()

        else:
            fea_tables = args.feature_tables
            model = args.model
            if args.gender =='yes':
                logger.info(f'[Training] You are using model {model} via feature tables {fea_tables} and considering gender!')
                train = training.Train(model_name=model,feature_tables=fea_tables,gender=True)
                #train.start()
                train.run()
            else:
                logger.info(f'[Training] You are using model {model} via feature tables {fea_tables}')
                train = training.Train(model_name=model,feature_tables=fea_tables)
                #train.start()
                train.run()

    else:
        logger.info('training model not finished yet!')
        
if __name__=='__main__':
    cmd()




