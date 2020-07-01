import config
from global_values import *
from sklearn.decomposition import PCA
import pandas as pd 
from common.sql_handler import SqlHandler
from common.log_handler import get_logger

logger = get_logger()

def hog_pca():
    sql_handler = SqlHandler()
    pca = PCA(n_components=0.999)
    hog = pd.read_csv(config.data_dir+FACE_HOG)
    hog_pca_values = pca.fit_transform(hog)
    hog_pca_names = ['hog_pca_'+str(i) for i in range(184)]
    hog_pca = pd.DataFrame(hog_pca_values,columns = hog_pca_names)
    id = [float(id[:-1]) for id in PREFIX]
    col_name = hog_pca.columns.tolist()
    col_name.insert(0,'ID')
    hog_pca= hog_pca.reindex(columns = col_name,fill_value = 1)
    hog_pca['ID'] = id

    sql_handler.execute(f'drop table if exists {config.tbl_exp3_hog_fea};') #因为每次选择特征不一样，所以入库之前需要删除原来的表
    sql_handler.df_to_db(hog_pca, config.tbl_exp3_hog_fea)
    logger.info('hog feature exp3 has been stored!')