import numpy as np
import pandas as pd
from afinn import Afinn
from global_values import *

afinn = Afinn()

class text_fea():
    def __init__(self):
        self.depressionWords = depressionWords
        
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
        wordsAfinn = [afinn.score(word) for word in words]
        features.append(np.mean(wordsAfinn))
        features.append(np.median(wordsAfinn))
        features.append(np.max(wordsAfinn))
        features.append(np.min(wordsAfinn))
        features.append(np.std(wordsAfinn))
        features = np.array([features])
        return features


if __name__ == "__main__":
    readDepressionWord=open('E:/data/depression_words.txt')
    depressionWords = readDepressionWord.read()
    depressionWords= [word.strip().lower() for word in depressionWords.split(',')]
    readDepressionWord.close()
    #print(depressionWords[:5])
    textstates = text_fea(depressionWords)
    df = pd.read_csv('E:/data/492_P/492_TRANSCRIPT.csv',delimiter='\t')
    #print(sum(df.start_time-df.stop_time))
    TEXT_COLUMNS=['sentencesNum','wordsNum','laughters/words','depressions/words',\
    'afinnMean','afinnMedian','afinnMin','afinnMax','afinnStd']
    df1 = textstates.textFeas(df)
    df2 = textstates.textFeas(df)
    print(pd.DataFrame(np.concatenate((df1,df2))))



