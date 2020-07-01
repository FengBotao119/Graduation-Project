import sqlite3
import traceback
from sqlalchemy import create_engine, MetaData

import pandas as pd
from common.log_handler import get_logger
import config
from global_values import *
config.init()

logger = get_logger()
#有问题 有些函数没有commit
class SqlHandler:
    def __init__(self):
        self.conn = sqlite3.connect(config.db_path)
        self.engine = create_engine(f'sqlite:///{config.db_path}')
    
    def execute(self, sql):
        try:
            cur = self.conn.cursor()
            cur.execute(sql)
            #??
            res = None
            if 'returning' in sql:
                res = cur.fetchone()
            #??
            self.conn.commit()
            cur.close()
            return res
        except:
            traceback.print_exc()  #报错信息更详细

    def query(self, sql):
        try:
            cur = self.conn.cursor()
            cur.execute(sql)
            res = cur.fetchall()
            cur.close()
            if not res:
                res = []
            return res
        except:
            traceback.print_exc()
            return #需要删掉吗

    def disconnect(self):
        """Always remember to invoke disconnect
        """
        self.conn.close()
        return
    def df_to_db(self, data_frame, table):
        try:
            self.execute(f'drop table if exists {table};') #删除某个表
        except:   #没用 感觉可以删了
            pass
        data_frame.to_sql(table, self.engine, index=False)  #将df存入数据库
        logger.info('stored into ' + table)

    def get_df(self, table):
        return pd.read_sql(table, self.engine)  #从数据库中读取某个table

    def get_cloumns_from_table(self, table):
        if type(table) == str:
            df = pd.read_sql(f'select * from {table} limit 1', self.engine) #只输出第一句 limit n 表示前n条
            return list(df.columns.values)
        else:
            audio_fea, video_fea, text_fea = [], [], []
            #暂时不考虑 一个module下存在多个特征
            for tb in table:
                df = pd.read_sql(f'select * from {tb} limit 1', self.engine)
                cols = list(df.columns.values)
                if tb in AUDIO_TABLE:
                    audio_fea += cols
                elif tb in VIDEO_TABLE:
                    video_fea += cols
                elif tb in TEXT_TABLE:
                    text_fea += cols
                else:
                    raise ValueError
            return audio_fea,video_fea,text_fea

if __name__ == "__main__":
    sqlhandler = SqlHandler()
    

