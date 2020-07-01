import configparser
import ast
import common.log_handler as log_handler

BASE_DIR = 'E:/'

conf_file = BASE_DIR + 'config/config.ini.model'
config = configparser.ConfigParser()


trainable = None
data_dir = None
pretrained_model_dir = None

logger = None

# table's name here
feature_exp01 = None

# database info here
db_path = None
tbl_training_set = None
tbl_develop_set = None
tbl_test_set = None
tbl_exp1_fea = None
tbl_exp1_head_fea = None
tbl_exp1_face_fea = None

tbl_exp2_audio_fea = None

tbl_exp3_audio_fea = None
tbl_exp3_vedio_fea = None
tbl_exp3_text_fea = None
tbl_exp3_hog_fea = None

tbl_train_ros_set = None
tbl_dev_ros_set = None


def init():
    global trainable
    global data_dir
    global pretrained_model_dir
    global logger
    global feature_exp01
    global db_path, tbl_develop_set, tbl_training_set, tbl_test_set, tbl_exp2_audio_fea
    global tbl_exp1_face_fea, tbl_exp1_head_fea,tbl_exp1_fea
    global tbl_exp3_audio_fea,tbl_exp3_vedio_fea,tbl_exp3_text_fea,tbl_exp3_hog_fea
    global tbl_train_ros_set
    logger = log_handler.get_logger()

    config.read(conf_file)
    data_dir = config.get('data', 'data_dir')
    pretrained_model_dir = config.get('pretrained_models','pretrained_models_dir')

    trainable = ast.literal_eval(config.get('training', 'trainable'))

    feature_exp01 = config.get('feature_table', 'tbl_exp01')
    db_path = config.get('database', 'db_path')
    tbl_training_set = config.get('database', 'tbl_training_set')
    tbl_develop_set = config.get('database', 'tbl_develop_set')
    tbl_test_set = config.get('database', 'tbl_test_set')
    tbl_exp2_audio_fea = config.get('database', 'tbl_exp2_audio_fea')
    tbl_exp1_head_fea = config.get('database', 'tbl_exp1_head_fea')
    tbl_exp1_face_fea = config.get('database', 'tbl_exp1_face_fea')
    tbl_exp1_fea = config.get('database', 'tbl_exp1_fea')
    
    tbl_exp3_audio_fea = config.get('database', 'tbl_exp3_audio_fea')
    tbl_exp3_vedio_fea = config.get('database', 'tbl_exp3_vedio_fea')
    tbl_exp3_text_fea = config.get('database', 'tbl_exp3_text_fea')
    tbl_exp3_hog_fea = config.get('database', 'tbl_exp3_hog_fea')

    tbl_train_ros_set = config.get('database','tbl_train_ros_set')
    logger.info('Init config!..')


if __name__ == '__main__':
    init()
    
    