import numpy as np
import pandas as pd
from afinn import Afinn

afinn = Afinn()

class textFea():
    def __init__(self,depressionWords):
        self.depressionWords = depressionWords
        
    def textFeas(self,df):
        features = list()
        df=df[df.speaker=='Participant']

        sentenceNumber=df.shape[0]
        features.append(sentenceNumber) #sentences
         
        sentences=[sentence for sentence in df.value.values ]
        words=[word.lower() for sentence in sentences for word in sentence.split()]
        wordsNumber = len(words)
        features.append(wordsNumber) #words

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
        return features


if __name__ == "__main__":
    readDepressionWord=open('D:/data/depression_words.txt')
    depressionWords = readDepressionWord.read()
    depressionWords= [word.strip().lower() for word in depressionWords.split(',')]
    readDepressionWord.close()
    #print(depressionWords[:5])
    textstates = textFea(depressionWords)
    df = pd.read_csv('D:/data/492_P/492_TRANSCRIPT.csv',delimiter='\t')
   # print(df.columns)
    print(textstates.textFeas(df))
    



