import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

class WordAttention(nn.Module):
    def __init__(self, vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size, dropout):
        """
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param word_att_size: size of word-level attention layer
        :param dropout: dropout
        """
        super(WordAttention, self).__init__()

        # Embeddings (look-up) layer input->embedding
        self.embeddings = nn.Embedding(vocab_size, emb_size)

        # Bidirectional word-level RNN embedding->GRU
        self.word_rnn = nn.GRU(emb_size, word_rnn_size, num_layers=word_rnn_layers, bidirectional=True,
                               dropout=dropout, batch_first=True)

        # Word-level attention network hidden->attention
        self.word_attention = nn.Linear(2 * word_rnn_size, word_att_size)

       # Word context vector to take dot-product with attentino->weight
        self.word_context_vector = nn.Linear(word_att_size, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, sentences, words_per_sentence):
        """
        Forward propagation.
        :param sentences: encoded sentence-level data, a tensor of dimension (n_sentences, word_pad_len, emb_size)
        :param words_per_sentence: sentence lengths, a tensor of dimension (n_sentences)
        :return: sentence embeddings, attention weights of words
        """
        # Sort sentences by decreasing sentence lengths (SORTING #2)
        words_per_sentence, sent_sort_ind = words_per_sentence.sort(dim=0, descending=True)
        sentences = sentences[sent_sort_ind]  # (n_sentences, word_pad_len, emb_size)

        # Get word embeddings, apply dropout
        sentences = self.dropout(self.embeddings(sentences))  # (n_sentences, word_pad_len, emb_size)

        # Re-arrange as words by removing pad-words (SENTENCES -> WORDS)
        pack = pack_padded_sequence(sentences,
                                         lengths=words_per_sentence,
                                         batch_first=True)
        #(n_words, emb_size), bw is the effective batch size at each word-timestep

        (words, _, _, _), _ = self.word_rnn(pack)  # (n_words, 2 * word_rnn_size), (max(sent_lens))

        # Find attention vectors by applying the attention linear layer
        att_w = self.word_attention(words)  # (n_words, att_size)
        att_w = F.tanh(att_w)  # (n_words, att_size)

        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(1)  # (n_words)

        max_value = att_w.max()  
        att_w = torch.exp(att_w - max_value)  # (n_words)

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        att_w, _ = pad_packed_sequence(PackedSequence(att_w, bw), batch_first=True)
        # (n_sentences, max_sent_len_in_batch)

        # Calculate softmax values
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)

        # (n_sentences, max_sent_len_in_batch)

        # Similarly re-arrange word-level RNN outputs as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentences, _ = pad_packed_sequence(PackedSequence(words, bw), batch_first=True)
        # (n_sentences, max_sent_len_in_batch, 2 * word_rnn_size)

        # Find sentence embeddings
        sentences = sentences * word_alphas.unsqueeze(2)  # (n_sentences, max_sent_len_in_batch, 2 * word_rnn_size)
        sentences = sentences.sum(dim=1)  # (n_sentences, 2 * word_rnn_size)

        # Unsort sentences into the original order (INVERSE OF SORTING #2)
        _, sent_unsort_ind = sent_sort_ind.sort(dim=0, descending=False)  # (n_sentences)
        sentences = sentences[sent_unsort_ind]  # (n_sentences, 2 * word_rnn_size)
        word_alphas = word_alphas[sent_unsort_ind]  # (n_sentences, max_sent_len_in_batch)

        return sentences, word_alphas


if __name__ == "__main__":
    model = WordAttention(vocab_size=10,emb_size=12,word_rnn_size=2,word_rnn_layers=2,word_att_size=1,dropout=0.2)
    print(model)