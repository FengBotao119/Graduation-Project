"""
Use the features from paper
Extract some statistical features of them;

Audio: COVAREP FORMANT
Text: DEPRESSION WORDS
Vedio:

"""
import itertools#???
from concurrent.futures import ThreadPoolExecutor, as_completed#???
import pandas as pd
from common.sql_handler import SqlHandler
from common.stat_features import StatsFea
from core.feature_extraction.example_1.text_fea import textFea
from common.log_handler import get_logger
from global_values import *
import config

logger = get_logger()
stats_fea = StatsFea()

readDepressionWord=open(config.data_dir+'depression_words.txt')
depressionWords = readDepressionWord.read()
depressionWords= [word.strip().lower() for word in depressionWords.split(',')]
readDepressionWord.close()
text_fea = textFea(depressionWords)

#提取某个文件夹下的数据 eg：300_P
def gen_sigle_fea(fold):
    fea_item = list()
    fea_item.append(fold[:-1])
    path = f"{config.data_dir}/{fold}P/{fold}{SUFFIX['covarep']}"
    covarep =  pd.read_csv(path, header=None)
    covarep.columns = COVAREP_COLUMNS

    path = f"{config.data_dir}/{fold}P/{fold}{SUFFIX['formant']}"
    formant = pd.read_csv(path, header=None)
    formant.columns = FORMANT_COLUMNS

    path = f"{config.data_dir}/{fold}P/{fold}{SUFFIX['text']}"
    text = pd.read_csv(path,delimiter='\t')

    #以上代码在读取FORMANT 和 COVAREP数据 和 TEXT数据
    
    covarep = covarep[covarep['VUV'] == 1]  #读取有效数据
    for fea in COVAREP_COLUMNS:
        if fea is 'VUV':
            continue
        else:
            fea_item += stats_fea.gen_fea(covarep[fea].values) #计算统计特征

    for fea in FORMANT_COLUMNS:
        fea_item += stats_fea.gen_fea(formant[fea].values)

    fea_item += text_fea.textFeas(text)
    logger.info(f'{fold} has been extrated audio and text feature in exp1!..')
    return fea_item

def gen_fea():
    sql_handler = SqlHandler()

    audio_text_value = list()
    with ThreadPoolExecutor(max_workers=30) as executor: #并行启动任务
        task = [executor.submit(gen_sigle_fea, fold) for fold in PREFIX]
        for future in as_completed(task):
            try:
                fea_item = future.result() #每一个文件下所有数据的特征 eg：300_P
                audio_text_value.append(fea_item)
            except:
                continue
                
    COVAREP_COLUMNS.remove('VUV')
    audio_fea = list()
    audio_fea.append('ID')
    COVAREP_COLUMNS.extend(FORMANT_COLUMNS)
    for a_fea, s_fea in itertools.product(COVAREP_COLUMNS, stats_fea.columns): #笛卡尔积 相当于嵌套for循环
        audio_fea.append(a_fea + '_' + s_fea)
    audio_text_fea = audio_fea+TEXT_COLUMNS

    assert len(audio_text_value[0]) == len(audio_text_fea)

    audio_text_df = pd.DataFrame(audio_text_value, columns=audio_text_fea)

    sql_handler.execute(f'drop table if exists {config.tbl_exp1_fea};') #因为每次选择特征不一样，所以入库之前需要删除原来的表
    sql_handler.df_to_db(audio_text_df, config.tbl_exp1_fea)
    logger.info('audio feature exp1 has been stored!')