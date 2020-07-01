from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np 
from global_values import *
import config

train = pd.read_csv(config.data_dir+TRAIN_SET_NAME)

label = train[['PHQ8_Score']]
train = train.drop(['PHQ8_Score'],axis=1)

train_X= train.values
train_X_cols = train.columns.tolist()
train_Y = label.values
train_Y_col = label.columns.tolist()

ROS = RandomOverSampler(random_state =1234)

new_X,new_Y = ROS.fit_resample(train_X,train_Y)

cols = train_X_cols+train_Y_col
new_train = np.concatenate((new_X,new_Y.reshape(-1,1)),axis=1 )

data = pd.DataFrame(new_train,columns = cols)

data.to_csv(config.data_dir+'train_ROS.csv',index=0)
