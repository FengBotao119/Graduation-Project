from global_values import *
from common.slide_windows import slideWindows
from afinn import Afinn
import numpy as np 
import pandas as pd
class Audio_features():
    def __init__(self):
        self.slide_size = 4
        self.hop_size =1
        self.fps = 100
    def formant_fea(self,path):
        data = np.loadtxt(path,delimiter=',')
        data[np.isnan(data)]=0
        data[np.isinf(data)]=0  
        return slideWindows(self.slide_size,self.hop_size,self.fps,data)

    def covarep_fea(self,path):
        data = np.loadtxt(path,delimiter=',')
        #VUV = 1
        data = data[data[:,1]==1]
        #romove VUV i forget!!!
        data = np.delete(data,1,axis=1)
        data[np.isnan(data)]=0
        data[np.isinf(data)]=0  
        return slideWindows(self.slide_size,self.hop_size,self.fps,data)

class Video_features():
    def __init__(self):
        self.slide_size = 4
        self.hop_size = 1
        self.fps = 30
    def fau_fea(self,path):
        #不用处理 数据已经被处理过了
        data = pd.read_csv(path)
        success = data[' success']==1
        data = data.iloc[:,4:].values
        data = data[success]
        return slideWindows(self.slide_size,self.hop_size,self.fps,data)
    def gaze_pose_fea(self,*path):
        gaze_path , pose_path = path
        gaze_data = pd.read_csv(gaze_path)
        pose_data = pd.read_csv(pose_path)

        if gaze_path[8:11] in ['367', '396', '432']:
            temp = np.all(pose_data.values != ' -1.#IND',axis=1) #['367_', '396_', '432_'] 缺失 存在异常值
            data = pd.merge(gaze_data,pose_data)#key = frame timestamps confidence success
            data = data[temp]
            data.iloc[:,-6:] = data.iloc[:,-6:].applymap(lambda x:float(x[1:]))
        else:
            data = pd.merge(gaze_data,pose_data)
        
        success = data[' success']==1
        data = data.values[:,4:]
        data = data[success]
        #data[:,-6:] = data[:,-6:].astype(int)
        return slideWindows(self.slide_size,self.hop_size,self.fps,data)
        

class Text_features():
    def __init__(self):
        self.depressionWords = depressionWords
        self.afinn = Afinn()
        
    def textFeas(self,df):
        features = list()
        df=df[df.speaker=='Participant']
        #存在异常数据 ['381_', '402_', '409_', '453_', '476_']:nan
        df = df[np.logical_not(pd.isnull(df.value))]
        
        duration = sum(df.stop_time-df.start_time)

        sentenceNumber=df.shape[0]
        features.append(sentenceNumber/duration) #sentences/duration
         
        sentences=[sentence for sentence in df.value.values ]
        words=[word.lower() for sentence in sentences for word in sentence.split()]
        wordsNumber = len(words)
        features.append(wordsNumber/duration) #words/duration

        laughter=[ 1 for sentence in sentences if 'laughter' in sentence ]
        laughterDivWords = np.sum(laughter)/wordsNumber
        features.append(laughterDivWords) #laughters

        depressionWordNumber=sum([1 for word in words if word in self.depressionWords])
        depDivWord = depressionWordNumber/wordsNumber
        features.append(depDivWord) #depression

        #stats
        wordsAfinn = [self.afinn.score(word) for word in words]
        features.append(np.mean(wordsAfinn))
        features.append(np.median(wordsAfinn))
        features.append(np.max(wordsAfinn))
        features.append(np.min(wordsAfinn))
        features.append(np.std(wordsAfinn))
        features = np.array([features])
        return features



