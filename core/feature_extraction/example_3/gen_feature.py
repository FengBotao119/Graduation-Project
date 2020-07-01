import itertools#笛卡尔积
from concurrent.futures import ThreadPoolExecutor, as_completed#???
import pandas as pd
import numpy as np
from common.sql_handler import SqlHandler
from common.log_handler import get_logger

from core.feature_extraction.example_3.audioFea import audio_fea
from core.feature_extraction.example_3.vedioFea import vedio_fea
from core.feature_extraction.example_3.textFea import text_fea
from core.feature_extraction.example_3.hogPca import hog_pca

from global_values import *
import config

logger = get_logger()

audio_fea = audio_fea()
text_fea = text_fea()
vedio_fea = vedio_fea()

#提取某个文件夹下的数据 eg：300_P
def gen_sigle_fea(fold):
    index = np.array([[int(fold[:-1])]]) #已经转为整数了

    path = f"{config.data_dir}/{fold}P/{fold}{SUFFIX['covarep']}"
    covarep =  pd.read_csv(path, header=None)
    covarep.columns = COVAREP_COLUMNS

    path = f"{config.data_dir}/{fold}P/{fold}{SUFFIX['formant']}"
    formant = pd.read_csv(path, header=None)
    formant.columns = FORMANT_COLUMNS

    path = f"{config.data_dir}/{fold}P/{fold}{SUFFIX['face_2d']}"
    face_2d = pd.read_csv(path)

    path = f"{config.data_dir}/{fold}P/{fold}{SUFFIX['text']}"
    text = pd.read_csv(path,delimiter='\t')

    audio_covarep_fea = audio_fea.audioFeas(covarep) # return np.ndarray
    audio_formant_fea = audio_fea.audioFeas(formant)
    audioFea = np.concatenate((index,audio_covarep_fea,audio_formant_fea),axis=1) #73+5
    logger.info(f'{fold}audio features have been extracted!..')
    textFea = np.concatenate((index,text_fea.textFeas(text)),axis=1) #9
    logger.info(f'{fold}text features have been extracted!..')
    vedioFea = np.concatenate((index,vedio_fea.diff_cur_prev(face_2d)),axis=1) #44*2
    logger.info(f'{fold}vedio features have been extracted!..')

    logger.info(f'{fold}P has been extrated audio, vedio and text features in exp3!..')
    return audioFea,textFea,vedioFea



def gen_fea():
    sql_handler = SqlHandler()
    audio_feas,text_feas,vedio_feas = gen_sigle_fea(PREFIX[0])
    #读取hog特征 应该在模型训练的地方做
    #分三个表来提取数据

    with ThreadPoolExecutor(max_workers=30) as executor: #并行启动任务
        task = [executor.submit(gen_sigle_fea, fold) for fold in PREFIX[1:]]
        for future in as_completed(task):
            try: 
                audio_value,text_value,vedio_value = future.result() #每一个文件下所有数据的特征 eg：300_P
                audio_feas = np.concatenate((audio_feas,audio_value))
                vedio_feas = np.concatenate((vedio_feas,vedio_value))
                text_feas = np.concatenate((text_feas,text_value))
            except:
                continue
                
    
    COVAREP_COLUMNS.remove('VUV')
    audio_fea_name = ['ID']
    text_fea_name = ['ID']
    vedio_fea_name = ['ID']
    
    audio_fea_name.extend(COVAREP_COLUMNS+FORMANT_COLUMNS)
    text_fea_name.extend(TEXT_COLUMNS)
    vedio_fea_name.extend(STABLE_POINTS)
 
    assert len(audio_feas[0]) == len(audio_fea_name) and len(text_feas[0]) == len(text_fea_name) \
        and len(vedio_feas[0]) == len(vedio_fea_name)
    audio_df = pd.DataFrame(audio_feas, columns=audio_fea_name)
    vedio_df = pd.DataFrame(vedio_feas, columns=vedio_fea_name)
    text_df = pd.DataFrame(text_feas, columns=text_fea_name)

    hog_pca()
    
    sql_handler.execute(f'drop table if exists {config.tbl_exp3_audio_fea};') #因为每次选择特征不一样，所以入库之前需要删除原来的表
    sql_handler.df_to_db(audio_df, config.tbl_exp3_audio_fea)
    logger.info('audio feature exp3 has been stored!')

    sql_handler.execute(f'drop table if exists {config.tbl_exp3_vedio_fea};') #因为每次选择特征不一样，所以入库之前需要删除原来的表
    sql_handler.df_to_db(vedio_df, config.tbl_exp3_vedio_fea)
    logger.info('vedio feature exp3 has been stored!')
    
    sql_handler.execute(f'drop table if exists {config.tbl_exp3_text_fea};') #因为每次选择特征不一样，所以入库之前需要删除原来的表
    sql_handler.df_to_db(text_df, config.tbl_exp3_text_fea)
    logger.info('text feature exp3 has been stored!')