import itertools#笛卡尔积
from concurrent.futures import ThreadPoolExecutor, as_completed#???
import pandas as pd
import numpy as np
import os
from common.sql_handler import SqlHandler
from common.log_handler import get_logger

from core.feature_extraction.final_model.lld import Audio_features,Video_features,Text_features
from core.feature_extraction.final_model.deepSpectrum import genImg,genDeepFea,VggFc7,alexnetFc7
from core.feature_extraction.final_model.bags_of_words import bow

import torch
from torchvision import models
from tensorflow import keras
from global_values import *
import config

vgg19 = models.vgg.vgg19(pretrained=False)
vgg19.load_state_dict(torch.load(config.pretrained_model_dir+'vgg19.pth'))
vggfc =VggFc7(vgg19)

alexnet = models.alexnet(pretrained=False)
alexnet.load_state_dict(torch.load(config.pretrained_model_dir+'alexnet.pth'))
alexnetfc = alexnetFc7(alexnet)

logger = get_logger()
sqlhandler = SqlHandler()
#提取某个文件夹下的数据 eg：300_P
def gen_sigle_fea(fold):
    # audio_fea = Audio_features()
    # video_fea = Video_features()
    # #text_fea = Text_features()
    path = f"{config.data_dir}/{fold}P/{fold}{SUFFIX['covarep']}"
    # covarep =  audio_fea.covarep_fea(path)
    bow(path,feature_name = 'covarep')
    path = f'E:/database/COVAREP_BOW/{fold}covarep_bow.csv'
    covarep_bow = np.loadtxt(path,delimiter=';')
    covarep_bow = covarep_bow[:,1:]
    os.system('rm '+path)

    path = f"{config.data_dir}/{fold}P/{fold}{SUFFIX['formant']}"
    # formant = audio_fea.formant_fea(path)
    bow(path,feature_name = 'formant')
    path = f'E:/database/FORMANT_BOW/{fold}formant_bow.csv'
    formant_bow = np.loadtxt(path,delimiter=';')
    formant_bow = formant_bow[:,1:]
    os.system('rm '+path)

    # path = f"{config.data_dir}/{fold}P/{fold}{SUFFIX['au']}"
    # faus = video_fea.fau_fea(path)
    # bow(path,feature_name = 'faus')
    # path = f'E:/database/FAUs_BOW/{fold}faus_bow.csv'
    # faus_bow = np.loadtxt(path,delimiter=';')
    # faus_bow = faus_bow[:,1:]
    # os.system('rm '+ path)

    # path1 = f"{config.data_dir}/{fold}P/{fold}{SUFFIX['gaze']}"
    # path2 = f"{config.data_dir}/{fold}P/{fold}{SUFFIX['pose']}"
    # gaze_pose = video_fea.gaze_pose_fea(path1,path2)
    # bow(path1,path2,feature_name='gaze_pose')
    # path = f'E:/database/GAZE_POSE_BOW/{fold}gaze_pose_bow.csv'
    # gaze_pose_bow = np.loadtxt(path,delimiter=';')
    # gaze_pose_bow = gaze_pose_bow[:,1:]
    # os.system('rm '+path)

    #text特征在exp3中已经提取入库(tbl_exp3_text_fea)，可以直接从数据库中提取，这里不重复操作

    #genImg(fold)
    # deep_spectrum_vgg = genDeepFea(vggfc,fold)
    # deep_spectrum_alexnet = genDeepFea(alexnetfc,fold)

    logger.info(f'{fold} features have been extracted!..')
    #return fold[:-1],covarep,formant,faus,gaze_pose,deep_spectrum_vgg,deep_spectrum_alexnet,faus_bow,gaze_pose_bow
    return fold[:-1],covarep_bow,formant_bow
   
def calMaxLen(path,IDs,fileName,maxLen):
    X = []
    for ID in IDs:
        temp  = np.load(path+str(ID)+fileName)
        X.append(temp[:,1:])
    X = keras.preprocessing.sequence.pad_sequences(X,maxlen = maxLen,padding='post',dtype = 'float')
    return X

def store_fea(path,fileName,feature):
    #区分男女
    store_path = 'E:/rnn_models/data/'
    path = path
    fileName = fileName
    train = sqlhandler.get_df(config.tbl_train_ros_set)
    dev = sqlhandler.get_df(config.tbl_develop_set)
    test = sqlhandler.get_df(config.tbl_test_set)

    train['Participant_ID']=train['Participant_ID'].astype(int)
    dev['Participant_ID']=dev['Participant_ID'].astype(int)
    test['Participant_ID']=test['Participant_ID'].astype(int)

#----------not consider gender-----------------
    maxLen = max([np.load(path+file).shape[0] for file in os.listdir(path)])
    train_ID = train.iloc[:,0].values
    train_Y  = train.iloc[:,-1].values

    dev_ID = dev.iloc[:,0].values
    dev_Y = dev.iloc[:,-1].values

    test_ID = test.iloc[:,0].values
    test_Y = test.iloc[:,2].values

    train_X = calMaxLen(path,train_ID,fileName,maxLen)
    dev_X = calMaxLen(path,dev_ID,fileName,maxLen)
    test_X = calMaxLen(path,test_ID,fileName,maxLen)

    np.save(store_path+feature+'/train_X.npy',train_X)
    np.save(store_path+feature+'/train_Y.npy',train_Y)
    np.save(store_path+feature+'/dev_X.npy',dev_X)
    np.save(store_path+feature+'/dev_Y.npy',dev_Y)
    np.save(store_path+feature+'/test_X.npy',test_X)
    np.save(store_path+feature+'/test_Y.npy',test_Y)  
    logger.info(f'{feature} feature have been stored!..')
#------------------------------------------------------------

#--------------consider gender--------------------------------
#取得maxlen=???
    m_train_ID = train['Participant_ID'].values[train['Gender']==1]
    m_dev_ID = dev['Participant_ID'].values[dev['Gender']==1]
    m_test_ID = test['Participant_ID'].values[test['Gender']==1]

    m_train_Y = train['PHQ8_Score'].values[train['Gender']==1]
    m_dev_Y = dev['PHQ8_Score'].values[dev['Gender']==1]
    m_test_Y = test['PHQ_Score'].values[test['Gender']==1]

    f_train_ID = train['Participant_ID'].values[train['Gender']==0]
    f_dev_ID = dev['Participant_ID'].values[dev['Gender']==0]
    f_test_ID = test['Participant_ID'].values[test['Gender']==0]

    f_train_Y = train['PHQ8_Score'].values[train['Gender']==0]
    f_dev_Y = dev['PHQ8_Score'].values[dev['Gender']==0]
    f_test_Y = test['PHQ_Score'].values[test['Gender']==0]  

    m_maxLen = max([np.load(path+file).shape[0] for file in os.listdir(path) \
                   if int(file[:3]) in m_train_ID.tolist()+m_dev_ID.tolist()+m_test_ID.tolist() ])
    f_maxLen = max([np.load(path+file).shape[0] for file in os.listdir(path) \
                   if int(file[:3]) in f_train_ID.tolist()+f_dev_ID.tolist()+f_test_ID.tolist() ])

    m_train_X = calMaxLen(path,m_train_ID,fileName,m_maxLen)
    m_dev_X = calMaxLen(path,m_dev_ID,fileName,m_maxLen)
    m_test_X = calMaxLen(path,m_test_ID,fileName,m_maxLen)

    f_train_X = calMaxLen(path,f_train_ID,fileName,f_maxLen)
    f_dev_X = calMaxLen(path,f_dev_ID,fileName,f_maxLen)
    f_test_X = calMaxLen(path,f_test_ID,fileName,f_maxLen)  

    np.save(store_path+feature+'/m_train_X.npy',m_train_X)
    np.save(store_path+feature+'/m_train_Y.npy',m_train_Y)
    np.save(store_path+feature+'/m_dev_X.npy',m_dev_X)
    np.save(store_path+feature+'/m_dev_Y.npy',m_dev_Y)
    np.save(store_path+feature+'/m_test_X.npy',m_test_X)
    np.save(store_path+feature+'/m_test_Y.npy',m_test_Y)

    np.save(store_path+feature+'/f_train_X.npy',f_train_X)
    np.save(store_path+feature+'/f_train_Y.npy',f_train_Y)
    np.save(store_path+feature+'/f_dev_X.npy',f_dev_X)
    np.save(store_path+feature+'/f_dev_Y.npy',f_dev_Y)
    np.save(store_path+feature+'/f_test_X.npy',f_test_X)
    np.save(store_path+feature+'/f_test_Y.npy',f_test_Y)  

    logger.info(f'{feature} feature have been stored considering gender!..')

#-------------------------------------------------------------

def gen_fea():
    covarep_path_name = 'E:/database/COVAREP/'
    formant_path_name = 'E:/database/FORMANT/'
    faus_path_name = 'E:/database/FAUs/'
    gaze_pose_path_name = 'E:/database/GAZE-POSE/'
    ds_vgg_path_name = 'E:/database/DEEPSPECTRUM_VGG/'
    ds_alexnet_path_name = 'E:/database/DEEPSPECTRUM_ALEXNET/'
    faus_bow_path_name = 'E:/database/FAUs_BOW/'
    gaze_pose_bow_path_name = 'E:/database/GAZE_POSE_BOW/'
    covarep_bow_path_name = 'E:/database/COVAREP_BOW/'
    formant_bow_path_name = 'E:/database/FORMANT_BOW/'


    with ThreadPoolExecutor(max_workers=4) as executor: #并行启动任务
        task = [executor.submit(gen_sigle_fea, fold) for fold in PREFIX]      
        for future in as_completed(task):
            try: 
                # fold,covarep,formant,faus,gaze_pose,\
                #     deep_spectrum_vgg,deep_spectrum_alexnet,\
                #       faus_bow,gaze_pose_bow= future.result()
                # np.save(covarep_path_name+fold+'_covarep.npy',covarep)
                # np.save(formant_path_name+fold+'_formant.npy',formant)
                # np.save(faus_path_name+fold+"_faus.npy",faus)
                # np.save(gaze_pose_path_name+fold+"_gaze_pose.npy",gaze_pose)
                #np.save(faus_bow_path_name+fold+'_faus_bow.npy',faus_bow)
                #np.save(gaze_pose_bow_path_name+fold+'_gaze_pose_bow.npy',gaze_pose_bow)
                fold,covarep_bow,formant_bow = future.result()
                np.save(covarep_bow_path_name+fold+'_covarep_bow.npy',covarep_bow)
                np.save(formant_bow_path_name+fold+'_formant_bow.npy',formant_bow)
            except:
                print('there exist some errors')
                continue
    logger.info('all features have been extracted!..')
    
    # store_fea(covarep_path_name,'_covarep.npy','covarep')
    # store_fea(formant_path_name,'_formant.npy','formant')
    # store_fea(faus_path_name,'_faus.npy','faus')
    # store_fea(gaze_pose_path_name,'_gaze_pose.npy','gaze_pose')
    # store_fea(ds_vgg_path_name,'_ds_vgg.npy','ds_vgg')
    # store_fea(ds_alexnet_path_name,'_ds_alexnet.npy','ds_alexnet')
    #store_fea(faus_bow_path_name,'_faus_bow.npy','faus_bow')
    #store_fea(gaze_pose_bow_path_name,'_gaze_pose_bow.npy','gaze_pose_bow')
    store_fea(covarep_bow_path_name,'_covarep_bow.npy','covarep_bow')
    store_fea(formant_bow_path_name,'_formant_bow.npy','formant_bow')
    logger.info('all features have been stored!..')
  