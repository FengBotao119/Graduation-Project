import torch
from torch import optim

class Train():
    def __init__(self,feature,model):
        self.feature = feature
        self.model = model

    def train(self):
        if self.model=='Embedding_Average':
            from core.model.EmbeddingAverage import EmbedAvg
            
        else:
            pass


