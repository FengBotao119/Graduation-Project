import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import mean_absolute_error,mean_squared_error
from common.metrics import ccc_score

def get_single_result(feature):
    gender = ''
    if 'm_' in feature or 'f_' in feature:
        gender = feature[:2]
        feature = feature[2:]
    path = 'E:/rnn_models/data/'
    path_model = 'E:/rnn_models/models/'
    cols = ['rmse','mae','ccc']
    indexs = ['train','dev','test']
    train_X = np.load(path+feature+'/'+gender+'train_X.npy')
    train_Y = np.load(path+feature+'/'+gender+'train_Y.npy')
    dev_X = np.load(path+feature+'/'+gender+'dev_X.npy')
    dev_Y = np.load(path+feature+'/'+gender+'dev_Y.npy')
    test_X = np.load(path+feature+'/'+gender+'test_X.npy')
    test_Y = np.load(path+feature+'/'+gender+'test_Y.npy')
    model = keras.models.load_model(path_model+gender+'final_'+feature+'.h5')

    train_pre = model.predict(train_X)
    rmse = np.sqrt(mean_squared_error(train_Y,train_pre))
    mae = mean_absolute_error(train_Y,train_pre)
    ccc = ccc_score(train_Y,train_pre.reshape(train_Y.shape))
    train_r = [rmse,mae,ccc]

    dev_pre = model.predict(dev_X)
    rmse = np.sqrt(mean_squared_error(dev_Y,dev_pre))
    mae = mean_absolute_error(dev_Y,dev_pre)
    ccc = ccc_score(dev_Y,dev_pre.reshape(dev_Y.shape))
    dev_r = [rmse,mae,ccc]

    test_pre = model.predict(test_X)
    rmse = np.sqrt(mean_squared_error(test_Y,test_pre))
    mae = mean_absolute_error(test_Y,test_pre)
    ccc = ccc_score(test_Y,test_pre.reshape(test_Y.shape))
    test_r = [rmse,mae,ccc]       

    result= [train_r,dev_r,test_r]
    result= pd.DataFrame(result,index = indexs,columns = cols)
    return result
 

def getResults(feature,gender):
    if not gender:
        return get_single_result(feature)
    else:
        return get_single_result('m_'+feature),get_single_result('f_'+feature)

def get_single_output(feature,path):
    gender = ''
    if 'm_' in feature or 'f_' in feature:
        gender = feature[:2]
        feature = feature[2:]
    path_model = 'E:/rnn_models/models/'
    train_X = np.load(path+feature+'/'+gender+'train_X.npy')
    dev_X = np.load(path+feature+'/'+gender+'dev_X.npy')
    test_X = np.load(path+feature+'/'+gender+'test_X.npy')
    model = keras.models.load_model(path_model+gender+'final_'+feature+'.h5')

    train_pre = model.predict(train_X)*25
    dev_pre = model.predict(dev_X)*25
    test_pre = model.predict(test_X)*25

    return train_pre,dev_pre,test_pre

def get_all_outputs(gender):
    features = ['covarep','formant','covarep_bow','formant_bow','gaze_pose',\
               'faus','gaze_pose_bow','faus_bow','ds_vgg','ds_alexnet']
    features = [gender+feature for feature in features]
    path = 'E:/rnn_models/data/'

    train = np.load(path+'formant/'+gender+'train_Y.npy')
    dev = np.load(path+'formant/'+gender+'dev_Y.npy')
    test = np.load(path+'formant/'+gender+'test_Y.npy')

    for feature in features:
        train_pre,dev_pre,test_pre = get_single_output(feature,path)
        if 'covarep' in feature[-7:]:
            train.shape = train_pre.shape
            dev.shape = dev_pre.shape
            test.shape = test_pre.shape
        try:
            train = np.hstack((train,train_pre))
            dev = np.hstack((dev,dev_pre))
            test = np.hstack((test,test_pre))
        except:
            print(dev.shape,dev_pre.shape)

    train = pd.DataFrame(train,columns = ['score']+features)
    dev = pd.DataFrame(dev,columns = ['score']+features)
    test = pd.DataFrame(test,columns = ['score']+features)
    train.to_csv(f'e:/data/{gender}pre_train_scores.csv')
    dev.to_csv(f'e:/data/{gender}pre_dev_scores.csv')
    test.to_csv(f'e:/data/{gender}pre_test_scores.csv')


def getOutputs(gender):
    if not gender:
        return get_all_outputs('')
    else:
        return get_all_outputs('m_'),get_all_outputs('f_')