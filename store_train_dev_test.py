import pandas as pd
from common import sql_handler
from global_values import *
import config
config.init()

sqlHandler = sql_handler.SqlHandler()

trainData = pd.read_csv(config.data_dir+TRAIN_SET_NAME)
sqlHandler.df_to_db(trainData,config.tbl_training_set)


devData = pd.read_csv(config.data_dir+DEL_SET_NAME)
sqlHandler.df_to_db(devData,config.tbl_develop_set)


testData = pd.read_csv(config.data_dir+TEST_SET_NAME)
sqlHandler.df_to_db(testData,config.tbl_test_set)

trainROSData = pd.read_csv(config.data_dir+TRAIN_ROS_NAME)
sqlHandler.df_to_db(trainROSData,config.tbl_train_ros_set)


