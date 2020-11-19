import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,PackedSequence

class WordAttention(nn.Module):
    def __init__(self,voc_size,emb_size,rnn_size,rnn_layers,bidirection,att_size,dropout):
        super().__init__()
        self.embedding = nn.Embedding(voc_size,emb_size)
        self.gru = nn.GRU(emb_size,rnn_size,rnn_layers,batch_first=True,bidirectional=bidirection)
        self.att = nn.Linear(2*rnn_size,att_size)
        self.word_context_vector = nn.Linear(att_size,1)
        self.dropout = nn.Dropout(dropout)

    def forward(self,sentences,words_per_sentence):
        words_per_sentence, sent_sort_ind = words_per_sentence.sort(dim=0, descending=True)
        sentences = sentences[sent_sort_ind] #batch_size*padding_size
        sentences = self.embedding(sentences) #batch_size*padding_size*emb_size
        #压缩数据
        pack = pack_padded_sequence(sentences,lengths=words_per_sentence,batch_first=True) #n_words*emb_size
        pack = self.gru(pack) #n_words*(2*rnn_size) pack[0]:压缩后的数据和对应的batch_size
        att  = F.tanh(self.att(pack[0][0])) #n_words*att_size
        weight = torch.exp(self.word_context_vector(att)).squeeze() #n_words
        #解压数据
        sentences,_ = pad_packed_sequence(pack[0],batch_first=True) #batch_size*max_len*(2*rnn_size)
        weight,_    = pad_packed_sequence(PackedSequence(weight,pack[0][1]),batch_first=True).unsqueeze(2)#batch_size*max_len*1
        weight = weight/torch.sum(weight,dim=1,keepdim=True) #batch_size*max_len*(2*rnn_size)
        sentences = torch.sum(sentences*weight,dim=1)#batch_size*(2*rnn_size)
        
        _, sent_unsort_ind = sent_sort_ind.sort(dim=0, descending=False)  # (n_sentences)
        sentences = sentences[sent_unsort_ind]  # (n_sentences, 2 * word_rnn_size)
        weight = weight[sent_unsort_ind]  # (n_sentences, max_sent_len_in_batch)

        return sentences,weight





















