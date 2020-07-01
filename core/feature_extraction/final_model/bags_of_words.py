"""
use this tool(https://github.com/openXBOW/openXBOW) to extract more features about COVAREP and FAUs
AVEC 2019Workshop and Challenge: State-of-Mind, Detecting Depression with AI, and Cross-Cultural Affect Recognition

randomly choose 50% samples to create codebook
block size 4s
hop size 1s
codebook size 100
standardization
do log for COVAREP 

FAUS -> randomly choose 50% samples to create codebook -> extract all files' features -> log
GAZE_POSE ->standardised ->randomly choose 50% samples to create codebook -> extract all files' features -> log

attention: gaze_pose的地方写错了 需要重新计算和提取特征 covarep 考虑把时间结点给加上 训练一下
"""
from global_values import *
import config
import pandas as pd 
import numpy as np 
import os 
from common.sql_handler import SqlHandler
from common.write_csv import save_features

def createCodebook(feature_name):
    """
    COVAREP, FORMANT, FAUs ,***p.CSV->stack dataframe -> create codebook ->extract all files' features
    java -jar openXBOW.jar -i examples/example2/llds.csPv  -o bow.csv -a 1 -log -c kmeans++ -size 100 -B codebook -writeName -writeTimeStamp    

    """
    sqlhandler = SqlHandler()
    train  = sqlhandler.get_df(config.tbl_training_set)
    dev = sqlhandler.get_df(config.tbl_develop_set)
    trainID = train['Participant_ID'].values
    devID = dev['Participant_ID'].values 
    trainDev = np.hstack([trainID,devID])
    folds = np.random.choice(trainDev,20,replace=False) # for video 50, for audio 20 

    window_size = 4
    hop_size = 1
    openxbow = 'java -jar E:/openXBOW/openXBOW.jar '
    openxbow_options = '-writeName -writeTimeStamp -t ' + str(window_size) +' ' + str(hop_size)
    codebook_out = 'E:/openXBOW/codebooks/'
    openxbow_options_codebook = f'-size 100 -a 1 -log -B {codebook_out}{feature_name}_codebook '

    if feature_name == 'faus':
        for fold in folds:
            path = config.data_dir + str(fold) + '_P/' + str(fold) + '_' + SUFFIX['au']
            feature = np.loadtxt(path,delimiter=',',skiprows=1)
            success = feature[:,3]==1
            feature = feature[success,1:18]
            feature = np.delete(feature,[1,2],axis=1)
            save_features(codebook_out+'fausTrainDevRandom.csv', feature, append=True, instname=str(fold))
        os.system(openxbow+ f'-standardizeInput -i {codebook_out}fausTrainDevRandom.csv '+openxbow_options_codebook+\
            openxbow_options+ ' -c kmeans++'+ f' -o {codebook_out}temp.csv')
    elif feature_name == 'gaze_pose':
        for fold in folds:
            path1 = config.data_dir + str(fold) + '_P/' + str(fold) + '_' + SUFFIX['gaze']
            path2 = config.data_dir + str(fold) + '_P/' + str(fold) + '_' + SUFFIX['pose']
            gaze_data = pd.read_csv(path1)
            pose_data = pd.read_csv(path2)
            if fold in [367, 396, 432]:
                temp = np.all(pose_data.values != ' -1.#IND',axis=1) #['367_', '396_', '432_'] 缺失 存在异常值
                data = pd.merge(gaze_data,pose_data)#key = frame timestamps confidence success
                data = data[temp]
                data.iloc[:,-6:] = data.iloc[:,-6:].applymap(lambda x:float(x[1:]))
            else:
                data = pd.merge(gaze_data,pose_data)
            success = data[' success']==1
            data = data.values[:,1:]
            data = np.delete(data,[1,2],axis=1)
            data = data[success]
            save_features(codebook_out+'gazePoseTrainDevRandom.csv', data, append=True, instname=str(fold))
        os.system(openxbow+ f'-standardizeInput -i {codebook_out}gazeposeTrainDevRandom.csv '+openxbow_options_codebook+\
            openxbow_options+ ' -c kmeans++'+ f' -o {codebook_out}temp.csv')

    elif feature_name == 'covarep':
        for fold in folds:
            path = config.data_dir + str(fold) + '_P/' + str(fold) + '_' + SUFFIX['covarep']
            data = np.loadtxt(path,delimiter=',')
            timestamp = np.arange(0,data.shape[0]).reshape(data.shape[0],1)
            timestamp = timestamp/100
            data = np.hstack([timestamp,data])
            data = data[data[:,2]==1]
            data = np.delete(data,2,axis=1)
            data[np.isnan(data)]=0
            data[np.isinf(data)]=0  
            save_features(codebook_out+'covarepTrainDevRandom.csv', data, append=True, instname=str(fold))
        os.system(openxbow+ f'-standardizeInput -i {codebook_out}covarepTrainDevRandom.csv '+openxbow_options_codebook+\
            openxbow_options+ ' -c kmeans++'+ f' -o {codebook_out}temp.csv')
    else:
        for fold in folds:
            path = config.data_dir + str(fold) + '_P/' + str(fold) + '_' + SUFFIX['formant']
            data = np.loadtxt(path,delimiter=',')
            timestamp = np.arange(0,data.shape[0]).reshape(data.shape[0],1)
            timestamp = timestamp/100
            data = np.hstack([timestamp,data])
            data[np.isnan(data)]=0
            data[np.isinf(data)]=0  
            save_features(codebook_out+'formantTrainDevRandom.csv', data, append=True, instname=str(fold))
        os.system(openxbow+ f'-standardizeInput -i {codebook_out}formantTrainDevRandom.csv '+openxbow_options_codebook+\
            openxbow_options+ ' -c kmeans++'+ f' -o {codebook_out}temp.csv')

def bow(*path,feature_name):
    #java -jar openXBOW.jar -i examples/example2/llds.csv -l examples/example2/labels.csv -t 5.0 0.5 -o bow.arff -a 1 -b codebook
    openxbow = 'java -jar E:/openXBOW/openXBOW.jar '
    openxbow_options =' -writeTimeStamp -t 4 1 '
    openxbow_options_codebook = '-b E:/openXBOW/codebooks/'+feature_name+'_codebook'
    file_out = 'E:/database/'

    #数据处理
    if feature_name == 'faus':
        path = path[0]
        feature = np.loadtxt(path,delimiter=',',skiprows=1)
        success = feature[:,3]==1
        feature = feature[success,1:18]
        feature = np.delete(feature,[1,2],axis=1)
        save_features('E:/database/FAUs_BOW/'+path[-16:-13]+'_'+feature_name+'.csv',feature,instname=path[-16:-13])
        file_in = 'E:/database/FAUs_BOW/'+path[-16:-13]+'_'+feature_name+'.csv'
        file_out = 'E:/database/FAUs_BOW/'+path[-16:-13]+'_'+feature_name+'_bow'+'.csv'
        operation = openxbow + '-i ' +file_in + openxbow_options + openxbow_options_codebook + ' -o ' +file_out
        os.system(operation)
        os.system('rm '+file_in)
        
    elif feature_name == 'gaze_pose':
        gaze_path , pose_path = path
        gaze_data = pd.read_csv(gaze_path)
        pose_data = pd.read_csv(pose_path)

        if gaze_path[8:11] in ['367', '396', '432']:
            temp = np.all(pose_data.values != ' -1.#IND',axis=1) #['367_', '396_', '432_'] 缺失 存在异常值
            data = pd.merge(gaze_data,pose_data)#key = frame timestamps confidence success
            data = data[temp]
            data.iloc[:,-6:] = data.iloc[:,-6:].applymap(lambda x:float(x[1:]))
        else:
            data = pd.merge(gaze_data,pose_data)
        
        success = data[' success']==1
        data = data.values[:,1:]
        data = np.delete(data,[1,2],axis=1)
        data = data[success]
        save_features('E:/database/GAZE_POSE_BOW/'+gaze_path[-17:-14]+'_'+feature_name+'.csv',data,instname=gaze_path[-17:-14])
        file_in = 'E:/database/GAZE_POSE_BOW/'+gaze_path[-17:-14]+'_'+feature_name+'.csv'
        file_out = 'E:/database/GAZE_POSE_BOW/'+gaze_path[-17:-14]+'_'+feature_name+'_bow'+'.csv'
        operation = openxbow  + '-i ' +file_in + openxbow_options + openxbow_options_codebook + ' -o ' +file_out
        os.system(operation)
        os.system('rm '+file_in)      

    elif feature_name == 'covarep':
        path = path[0]
        data = np.loadtxt(path,delimiter=',')
        timestamp = np.arange(0,data.shape[0]).reshape(data.shape[0],1)
        timestamp = timestamp/100
        data = np.hstack([timestamp,data])
        data = data[data[:,2]==1]
        data = np.delete(data,2,axis=1)
        data[np.isnan(data)]=0
        data[np.isinf(data)]=0          

        save_features('E:/database/COVAREP_BOW/'+path[-15:-12]+'_'+feature_name+'.csv',data,instname=path[-15:-12])
        file_in = 'E:/database/COVAREP_BOW/'+path[-15:-12]+'_'+feature_name+'.csv'
        file_out = 'E:/database/COVAREP_BOW/'+path[-15:-12]+'_'+feature_name+'_bow'+'.csv'
        operation = openxbow + '-i ' +file_in + openxbow_options + openxbow_options_codebook + ' -o ' +file_out
        os.system(operation)
        os.system('rm '+file_in)

    else:
        path = path[0]
        data = np.loadtxt(path,delimiter=',')
        timestamp = np.arange(0,data.shape[0]).reshape(data.shape[0],1)
        timestamp = timestamp/100
        data = np.hstack([timestamp,data])
        data[np.isnan(data)]=0
        data[np.isinf(data)]=0          

        save_features('E:/database/FORMANT_BOW/'+path[-15:-12]+'_'+feature_name+'.csv',data,instname=path[-15:-12])
        file_in = 'E:/database/FORMANT_BOW/'+path[-15:-12]+'_'+feature_name+'.csv'
        file_out = 'E:/database/FORMANT_BOW/'+path[-15:-12]+'_'+feature_name+'_bow'+'.csv'
        operation = openxbow + '-i ' +file_in + openxbow_options + openxbow_options_codebook + ' -o ' +file_out
        os.system(operation)
        os.system('rm '+file_in)       



