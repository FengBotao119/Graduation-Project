import pandas as pd
import numpy as np 

class audio_fea():   
     #数据预处理 ？？__init__
    def __init__(self):
        pass

    def audioFeas(self,df):
        if 'VUV' in df.columns:
            df = df[df.VUV==1]
            df.drop('VUV',axis=1,inplace=True)     

        cols = len(df.columns)
        return np.mean(df.values,axis=0).reshape(1,cols)


if __name__=='__main__':
    sol = audio_fea()
    data1 = pd.read_csv('E:/data/492_P/492_COVAREP.csv')
    data2 = pd.read_csv('E:/data/492_P/492_FORMANT.csv')
    COVAREP_COLUMNS = ['F0', 'VUV', 'NAQ', 'QOQ', 'H1H2', 'HRF', 'PSP',\
         'MDQ', 'peakSlope','Rd', 'Rd_conf']
    for i in range(25): COVAREP_COLUMNS.append('MCEP_' + str(i))
    for i in range(25): COVAREP_COLUMNS.append('HMPDM_' + str(i))
    for i in range(13): COVAREP_COLUMNS.append('HMPDD_' + str(i))
    data1.columns = COVAREP_COLUMNS
    data1 = sol.audioFeas(data1)

    data2 = sol.audioFeas(data2)
    print(np.concatenate((data1,data2,data1),axis=1).shape)

