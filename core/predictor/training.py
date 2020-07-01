"""
Gather training request from user and dispatch them
"""
from multiprocessing import Process #多线程处理
from concurrent.futures import ThreadPoolExecutor, wait #多线程处理

from global_values import *
from common.df_handler import get_data_by_id, get_data_multi_modality,get_npdata_by_id
from common.sql_handler import SqlHandler
from common.log_handler import get_logger
import config
import numpy as np
import pandas as pd
config.init()
logger = get_logger()


class Train(Process):
    def __init__(self, model_name=None,
                 feature_name=None,
                 gender=False,
                 feature_tables=None):
        """Train model Controller, dispatch the training tasks;
        Input:
            model_name: certain model depend on papers
            feature_name: support for a group of absolute features
            feature_tables: support for different feature table, which make it
                            is possible for us to combine different modality
                            features freely. But note that the train controller
                            is not responsible for processing the feature table,
                            it should be completed by a certain model.
            gender: if the model should consider the gender difference

        Output:
            Result and realted information will be printed by each estimator in logs'
        """
        super().__init__()#调用父类PROCESS
        self.model_name =model_name
        self.feature_name = feature_name
        self.feature_tables = feature_tables
        self.gender = gender
        self.sql_handler = SqlHandler()
        self._set_feature()#?
        

    def _set_feature(self):
        if self.feature_name is not None:
            # you r using feature from a ceratin way! 
            if self.feature_name == FEATURE_EXP_2:
                # if choose exp2 the data will be in pandas's dataframe by defaut
                self.data = get_data_by_id(config.tbl_exp2_audio_fea, self.gender)
                self.feature_list = self.sql_handler.get_cloumns_from_table(config.tbl_exp2_audio_fea)
                self.feature_list.remove('ID')
            elif self.feature_name == FEATURE_EXP_1:
                self.data = get_data_by_id(config.tbl_exp1_fea,self.gender)
                self.feature_list = self.sql_handler.get_cloumns_from_table(config.tbl_exp1_fea)
                self.feature_list.remove('ID')
#---------------------baseline----------------------------------              
            elif self.feature_name == FEATURE_EXP_3_VEDIO:  #
                self.data = get_data_by_id(config.tbl_exp3_vedio_fea,self.gender)
                self.feature_list = self.sql_handler.get_cloumns_from_table(config.tbl_exp3_vedio_fea)
                self.feature_list.remove('ID')

            elif self.feature_name == FEATURE_EXP_3_TEXT:  #
                self.data = get_data_by_id(config.tbl_exp3_text_fea,self.gender)
                self.feature_list = self.sql_handler.get_cloumns_from_table(config.tbl_exp3_text_fea)
                self.feature_list.remove('ID')

            elif self.feature_name == FEATURE_EXP_3_AUDIO:  #
                self.data = get_data_by_id(config.tbl_exp3_audio_fea,self.gender)
                self.feature_list = self.sql_handler.get_cloumns_from_table(config.tbl_exp3_audio_fea)
                self.feature_list.remove('ID')

            elif self.feature_name == FEATURE_EXP_3_HOGPCA:  #
                self.data = get_data_by_id(config.tbl_exp3_hog_fea,self.gender)
                self.feature_list = self.sql_handler.get_cloumns_from_table(config.tbl_exp3_hog_fea)
                self.feature_list.remove('ID')
#-----------------baseline---------------------------------------------

#-----------------finalmodel------------------------------------
            elif self.feature_name == FEATURE_FINAL_COVAREP:
                path = 'E:/rnn_models/data/covarep/'
                self.data = get_npdata_by_id(path,self.gender)

            elif self.feature_name == FEATURE_FINAL_FORMANT:
                path = 'E:/rnn_models/data/formant/'
                self.data = get_npdata_by_id(path,self.gender)
         
            elif self.feature_name == FEATURE_FINAL_FAUs:
                path = 'E:/rnn_models/data/faus/'
                self.data = get_npdata_by_id(path,self.gender)

            elif self.feature_name == FEATURE_FINAL_GAZE_POSE:
                path = 'E:/rnn_models/data/gaze_pose/'
                self.data = get_npdata_by_id(path,self.gender)

            elif self.feature_name == FEATURE_FINAL_TEXT:
                self.data = get_data_by_id(config.tbl_exp3_text_fea,self.gender)
                self.feature_list = self.sql_handler.get_cloumns_from_table(config.tbl_exp3_text_fea)
                self.feature_list.remove('ID')
            
            elif self.feature_name == FEATURE_FINAL_VGG:
                path = 'E:/rnn_models/data/ds_vgg/'
                self.data = get_npdata_by_id(path,self.gender)
            
            elif self.feature_name == FEATURE_FINAL_ALEXNET:
                path = 'E:/rnn_models/data/ds_alexnet/'
                self.data = get_npdata_by_id(path,self.gender) 

            elif self.feature_name == FEATURE_FINAL_GAZE_POSE_BOW:
                path = 'E:/rnn_models/data/gaze_pose_bow/'
                self.data = get_npdata_by_id(path,self.gender) 

            elif self.feature_name == FEATURE_FINAL_FAUs_BOW:
                path = 'E:/rnn_models/data/faus_bow/'
                self.data = get_npdata_by_id(path,self.gender) 
            
            elif self.feature_name == FEATURE_FINAL_COVAREP_BOW:
                path = 'E:/rnn_models/data/covarep_bow/'
                self.data = get_npdata_by_id(path,self.gender) 
            
            elif self.feature_name == FEATURE_FINAL_FORMANT_BOW:
                path = 'E:/rnn_models/data/formant_bow/'
                self.data = get_npdata_by_id(path,self.gender) 
            
            elif self.feature_name == FEATURE_FINAL_FUSION:
                path = 'E:/data/'
                if self.gender:
                    m_dev = pd.read_csv(path+'m_pre_dev_scores.csv')
                    m_dev = m_dev.values
                    m_dev_label = m_dev[:,0].reshape(m_dev.shape[0],1)
                    m_dev_features = m_dev[:,1:]

                    m_test = pd.read_csv(path+'m_pre_test_scores.csv')
                    m_test = m_test.values
                    m_test_label = m_test[:,0].reshape(m_test.shape[0],1)
                    m_test_features = m_test[:,1:]

                    f_dev = pd.read_csv(path+'f_pre_dev_scores.csv')
                    f_dev = f_dev.values
                    f_dev_label = f_dev[:,0].reshape(f_dev.shape[0],1)
                    f_dev_features = f_dev[:,1:]

                    f_test = pd.read_csv(path+'f_pre_test_scores.csv')
                    f_test = f_test.values
                    f_test_label = f_test[:,0].reshape(f_test.shape[0],1)
                    f_test_features = f_test[:,1:]

                    self.data =  m_dev_features,m_dev_label,m_test_features,m_test_label,\
                                f_dev_features,f_dev_label,f_test_features,f_test_label 
                else:
                    dev = pd.read_csv(path+'pre_dev_scores.csv')
                    dev = dev.values
                    dev_label = dev[:,0].reshape(dev.shape[0],1)
                    dev_features = dev[:,1:]

                    test = pd.read_csv(path+'pre_test_scores.csv')
                    test = test.values
                    test_label = test[:,0].reshape(test.shape[0],1)
                    test_features = test[:,1:]

                    self.data = dev_features,dev_label,test_features,test_label
                    

#------------------finalmodel-----------------------------------------------
            else:
                print('not finished yet')
        elif self.feature_tables is not None:
            # Now you r using a multi-modality model!
            #audio特征暂时只计算一个
            #feature_tables 需要与数据库中的表名吻合  不然会报错
            self.data = get_data_multi_modality(self.feature_tables, self.gender)
            self.audio_fea, self.vedio_fea, self.text_fea = \
                        self.sql_handler.get_cloumns_from_table(self.feature_tables)
            self.audio_fea.remove('ID')
            self.vedio_fea.remove('ID')
            self.text_fea.remove('ID')
            self.feature_list = {'audio':self.audio_fea,'vedio':self.vedio_fea,'text':self.text_fea}
        else:
            print('You must choose a set of features to train!!!')

    def _train_eval(self, train,dev,test,model,feature):
        model = model(train, dev,test,features=feature)# rf 和rnn不一样 差了一个test参数 记得改
        model.train()
        return  model.eval()

    def run(self):
        if self.model_name == MODEL_RF:
            from core.predictor.randomForest.rf_predict import RfPredictor
            if self.feature_name is not None:
                if not self.gender:
                    train, dev,test= self.data
                    #运行的时候 需要改一下_train_eval函数 加入一个test参数
                    score = self._train_eval(train, dev, test,RfPredictor,self.feature_list)
                    logger.info(f'Evalutaion Scores {self.model_name} with {self.feature_name}: {score}')
                else:
                    train_m, dev_m,test_m, train_f, dev_f,test_f= self.data
                    score = self._train_eval(train_m, dev_m,test_m, RfPredictor,self.feature_list)
                    logger.info(f'Evalutaion Scores Male {self.model_name} with {self.feature_name}: {score}')

                    score = self._train_eval(train_f, dev_f,test_f,RfPredictor,self.feature_list)
                    logger.info(f'Evalutaion Scores Female {self.model_name} with {self.feature_name}: {score}')
            else:
                from core.predictor.randomForest.rf_predict import MultiModalRandomForest
                if not self.gender:
                    # multi_modality
                    mmrf = MultiModalRandomForest(self.data,self.feature_list)
                    score = mmrf.eval()
                    logger.info(f'Evalutaion Scores {self.model_name} with {self.feature_tables}: {score}')
                else:
                    data_male = self.data['male']
                    mmrf = MultiModalRandomForest(data_male,self.feature_list)
                    score = mmrf.eval()
                    logger.info(f'Evalutaion Scores Male {self.model_name} with {self.feature_tables}: {score}')
                    
                    data_female = self.data['female']
                    mmrf = MultiModalRandomForest(data_female,self.feature_list)
                    score = mmrf.eval()
                    logger.info(f'Evalutaion Scores Female {self.model_name} with {self.feature_tables}: {score}')
        
        elif self.model_name == MODEL_RNN:
            if self.feature_name is not None:
                from core.predictor.rnn.RNN import RnnPredictor
                if self.gender:
                    m_train_X,m_train_Y,m_dev_X,m_dev_Y,m_test_X,m_test_Y,f_train_X,f_train_Y,f_dev_X,f_dev_Y,f_test_X,f_test_Y = self.data
                    #m_train,m_dev,m_test = (m_train_X,m_train_Y),(m_dev_X,m_dev_Y),(m_test_X,m_test_Y)
                    f_train,f_dev,f_test = (f_train_X,f_train_Y),(f_dev_X,f_dev_Y),(f_test_X,f_test_Y)

                    #score = self._train_eval(m_train,m_dev,m_test,RnnPredictor,'m_'+self.feature_name)
                    #logger.info(f'Evalutaion Scores male {self.model_name} with {self.feature_name}: {score}')
                    
                    score = self._train_eval(f_train,f_dev,f_test,RnnPredictor,'f_'+self.feature_name)
                    logger.info(f'Evalutaion Scores female {self.model_name} with {self.feature_name}: {score}')

                else:
                    train_X,train_Y,dev_X,dev_Y,test_X,test_Y = self.data
                    train,dev,test = (train_X,train_Y),(dev_X,dev_Y),(test_X,test_Y)
                    score = self._train_eval(train,dev,test,RnnPredictor,self.feature_name)
                    logger.info(f'Evalutaion Scores {self.model_name} with {self.feature_name}: {score}')
        elif self.model_name == MODEL_LINEAR:
            from sklearn import linear_model
            from sklearn.metrics import mean_squared_error
            reg = linear_model.Ridge(alpha=10)
            if self.gender:
                pass
            else:
                dev_features,dev_label,test_features,test_label = self.data 
                reg.fit(dev_features,dev_label)
                dev_pre = reg.predict(dev_features)
                test_pre = reg.predict(test_features)

                dev_rmse = np.sqrt(mean_squared_error(dev_label,dev_pre))
                test_rmse = np.sqrt(mean_squared_error(test_label,test_pre))
                logger.info(f"dev_rmse: {dev_rmse}; test_rmse: {test_rmse}")
        else:
            print('not finish yet!')