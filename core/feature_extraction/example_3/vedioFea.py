import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA
from global_values import *

class vedio_fea():
    def __init__(self):
        pass

    def diff_cur_prev(self,df):
        df.columns = [col.strip() for col in df.columns.values]
        df = df[df.success==1]
        #current one substract the previous one
        stablePoints = df.loc[:,STABLE_POINTS]
        diff_cur_prev = -2*(stablePoints.rolling(window=2).mean()-stablePoints)
        diff_cur_prev = np.nanmean(diff_cur_prev.values,axis=0).reshape(1,88)

        return diff_cur_prev
    
    #stable points mean shape medium shape from train dev and test


    #calculate L2 distance and angles
    def angles_l2(self,df):
        df.columns = [col.strip() for col in df.columns.values]
        df = df[df.success==1]
        def calAngle(A,B):
            if A[0]==B[0]:return np.pi/2
            else:
                dx = A[0]-B[0]
                dy = A[1]-B[1]
            return np.arctan(dy/dx)
        #choose area which includes eyes, eyebows and mouth
        rows =  df.shape[0] #数据长度
        mouth_fea= df.loc[:,MOUTH]
        left_fea = df.loc[:,LEFTEYES]
        right_fea = df.loc[:,RIGHTS]
        feas = {'mouth':mouth_fea,'left_eye':left_fea,'right_eye':right_fea}

        for fea_name in feas.keys():
            fea = feas[fea_name]
            index = [col[1:] for col in fea.columns if col.startswith('x')]
            nrows = len(index)
            fea_col = [fea_name+index[i]+'_'+index[j] for i in range(nrows) for j in range(i+1,nrows)]
            #fea_name
            fea_angles = [fea_name+'_angles' for fea in fea_col]
            fea_L2 = [fea_name+'_L2' for fea in fea_col]
            fea_col = fea_angles+fea_L2
            feas_df = pd.DataFrame()
            fea_df = pd.DataFrame(columns=fea_col)

            for row in range(rows):
                tempDf = fea.iloc[[row]]
                x = tempDf.stack()[:len(index)].values
                y = tempDf.stack()[len(index):].values
                Temp = pd.DataFrame({'x':x,'y':y},index=index)
                
                angles = []
                L2 = []

                for i in range(nrows):
                    for j in range(i+1,nrows):
                        angles.append(calAngle(Temp.iloc[i],Temp.iloc[j]))
                        L2.append(np.linalg.norm(Temp.iloc[i]-Temp.iloc[j]))

                fea = pd.DataFrame(data=[angles+L2],columns=fea_col)
                fea_df = pd.concat([fea_df,fea],axis=0)
            feas_df = pd.concat([feas_df,fea_df],axis=1)
        angles_l2_fea = np.nanmean(feas_df.values,axis=0).reshape(1,600)    
        return angles_l2_fea


        

        


