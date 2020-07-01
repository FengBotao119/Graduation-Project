"""
merge two dataframe according to ID to generate training and dev set;
"""

import pandas as pd
import numpy as np
from common.sql_handler import SqlHandler
import config
from global_values import *
from common.log_handler import get_logger
logger = get_logger()

#融合PHQ8csv与特征的csv
def merge_df_by_id(df1, df2): 
    if df1 is None or df2 is None:
        return None
    return pd.merge(df1, df2, left_on='Participant_ID',
                                              right_on='ID',how='left') #因为过采样的原因
#合并所有特征，如果一个list有多个特征
def merge_dfs_by_id(dfs):  #合并多个数据集 但是好像没有必要 因为大部分论文都是单独训练 在第二轮综合考虑多个模态的结果再预测
    if not dfs:
        logger.info('dfs is empty, which means one of the modality is None') #后半句话什么意思？
        return None
    if len(dfs) == 1:
        return dfs[0]
    else:
        merged_df = dfs[0]
        for df in dfs[1:]:
            # the common column is ID
            merged_df = pd.merge(merged_df, df)
        return merged_df

#取单个特征
def get_data_by_id(feature_table, gender=False):
    sql_handler = SqlHandler()
    feature = sql_handler.get_df(feature_table)
    feature['ID'] = feature['ID'].apply(pd.to_numeric) #转为数值
    train = sql_handler.get_df(config.tbl_train_ros_set) 
    dev = sql_handler.get_df(config.tbl_develop_set)
    test = sql_handler.get_df(config.tbl_test_set)

    if not gender:
        train_set = merge_df_by_id(train, feature) #合并特征csv和PHQcsv 调用了之前的函数 merge_df_by_id()
        dev_set = merge_df_by_id(dev, feature) #合并特征csv和PHQcsv
        test_set = merge_df_by_id(test,feature)
        return train_set, dev_set,test_set
    else:  #考虑性别因素
        train_male = train[train['Gender'] == 1]
        train_female = train[train['Gender'] == 0]
        dev_male = dev[dev['Gender'] == 1]
        dev_female = dev[dev['Gender'] == 0]
        test_male = test[test['Gender']==1]
        test_female = test[test['Gender']==0]

        train_male = merge_df_by_id(train_male, feature)
        train_female = merge_df_by_id(train_female, feature)
        dev_male = merge_df_by_id(dev_male, feature)
        dev_female = merge_df_by_id(dev_female, feature)
        test_male = merge_df_by_id(test_male, feature)
        test_female = merge_df_by_id(test_female, feature)
        return train_male, dev_male,test_male, train_female, dev_female,test_female

#取多个特征
def get_data_multi_modality(tables, gender=False):
    """gather data from different tables in every modality
        and generate train set and dev dev set of them.
    """
    sql_handler = SqlHandler()
    audio_df, video_df, text_df = [], [], []
    for tb in tables:
        if tb in AUDIO_TABLE: audio_df.append(sql_handler.get_df(tb))  #从数据库提取特征存入list
        elif tb in VIDEO_TABLE: video_df.append(sql_handler.get_df(tb))
        elif tb in TEXT_TABLE: text_df.append(sql_handler.get_df(tb))
        else: pass

    audio_merge_df = merge_dfs_by_id(audio_df)
    video_merge_df = merge_dfs_by_id(video_df)
    text_merge_df = merge_dfs_by_id(text_df)
    
    train = sql_handler.get_df(config.tbl_train_ros_set)
    dev = sql_handler.get_df(config.tbl_develop_set)
        
    if not gender: #不考虑性别 
        data_dct = {
            'audio_train': merge_df_by_id(train, audio_merge_df),
            'audio_dev': merge_df_by_id(dev, audio_merge_df),
            'vedio_train': merge_df_by_id(train, video_merge_df),
            'vedio_dev': merge_df_by_id(dev, video_merge_df),
            'text_train': merge_df_by_id(train, text_merge_df),
            'text_dev': merge_df_by_id(dev, text_merge_df)
        }
    else:#考虑性别
        train_male = train[train['Gender'] == 1]
        train_female = train[train['Gender'] == 0]
        dev_male = dev[dev['Gender'] == 1]
        dev_female = dev[dev['Gender'] == 0]

        data_dct = {
            'male': {
                'audio_train': merge_df_by_id(train_male, audio_merge_df),
                'audio_dev': merge_df_by_id(dev_male, audio_merge_df),
                'vedio_train': merge_df_by_id(train_male, video_merge_df),
                'vedio_dev': merge_df_by_id(dev_male, video_merge_df),
                'text_train': merge_df_by_id(train_male, text_merge_df),
                'text_dev': merge_df_by_id(dev_male, text_merge_df)
            },
            'female':{
                'audio_train': merge_df_by_id(train_female, audio_merge_df),
                'audio_dev': merge_df_by_id(dev_female, audio_merge_df),
                'vedio_train': merge_df_by_id(train_female, video_merge_df),
                'vedio_dev': merge_df_by_id(dev_female, video_merge_df),
                'text_train': merge_df_by_id(train_female, text_merge_df),
                'text_dev': merge_df_by_id(dev_female, text_merge_df)
            }
        }
     

    return data_dct
        
def get_npdata_by_id(path,gender):
    data_table =['train_X.npy','train_Y.npy','dev_X.npy','dev_Y.npy','test_X.npy','test_Y.npy']

    if not gender:
        return (np.load(path+data) for data in data_table)
    else:
        m_train_X,m_train_Y,m_dev_X,m_dev_Y,m_test_X,m_test_Y = (np.load(path+'m_'+data) for data in data_table)
        f_train_X,f_train_Y,f_dev_X,f_dev_Y,f_test_X,f_test_Y = (np.load(path+'f_'+data) for data in data_table)
        return m_train_X,m_train_Y,m_dev_X,m_dev_Y,m_test_X,m_test_Y,f_train_X,f_train_Y,f_dev_X,f_dev_Y,f_test_X,f_test_Y
        


