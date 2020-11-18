import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

class EmbedAvg(nn.Module):
    def __init__(self,voc_size,emb_size,output_size,dropout):
        super(EmbedAvg,self).__init__()
        self.emb = nn.Embedding(voc_size,emb_size,padding_idx=0)
        self.fc = nn.Linear(emb_size,output_size)
        self.drop = nn.Dropout(dropout)
        
    def forward(self,x):
        emb = self.emb(x) # batch_size*sequence_size*embedding_size
        embAvg = F.avg_pool2d(emb,(emb.shape[1],1)).squeeze()
        output = torch.sigmoid(self.fc(self.drop(embAvg)))
        return output
