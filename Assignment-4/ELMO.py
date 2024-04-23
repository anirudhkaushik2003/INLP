import torch
import numpy as np
import torch.nn as nn
import torchtext
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import pandas as pd
import os
from tqdm import tqdm

from dataset import MakeData, PretrainDataset

class ELMO(nn.Module):
    def __init__(self,hidden_size, embedding_size, embedding_matrix, vocab):
        super(ELMO, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab = vocab

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), freeze=True, padding_idx=self.vocab['<PAD>'])
        self.lsftm_f1 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lsftm_f2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm_b1 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm_b2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        

    def forward(self, xf, xb):
        x = self.embedding(x)

        outf1, _ = self.lsftm_f1(xf) # next word prediction
        outb1, _ = self.lstm_b1(xb) # previous word prediction

        outb1 = torch.flip(outb1, [1]) # flip along seq_len dimension

        net_out1 = torch.cat((outf1, outb1), 2) # cat along hidden_size dimension

        outf2, _ = self.lsftm_f2(outf1) # next word prediction
        outb2, _ = self.lstm_b2(outb1) # previous word prediction

        outb2 = torch.flip(outb2, [1])

        net_out2 = torch.cat((outf2, outb2), 2) # cat along hidden_size dimension

        return net_out1, net_out2 # concatenated output of forward and backward lstm for layer 1 and 2

class NewsClassifierElmo(nn.Module):
    def __init__(self, hidden_size, embedding_size, embedding_matrix, vocab, num_classes, pretrained_path=None):
        super(NewsClassifierElmo, self).__init__()
        self.pretrained_path = pretrained_path

        self.elmo = ELMO(hidden_size, embedding_size, embedding_matrix, vocab)
        if self.pretrained_path != None:
            self.elmo.load_state_dict(torch.load(self.pretrained_path))

        self.lamdas = nn.Parameter(torch.randn(2, requires_grad=True))

        self.lstm = nn.LSTM(hidden_size*2, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, xf, xb):
        net_out1, net_out2 = self.elmo(xf, xb)

        lamdas_softmax = nn.Softmax(dim=0)(self.lamdas)
        combined_out = lamdas_softmax[0]*net_out1 + lamdas_softmax[1]*net_out2

        _, (hidden, _) = self.lstm(combined_out) 
        # hidden shape is (num_layers*num_directions, batch_size, hidden_size)
        hidden = hidden.permute(1, 0, 2).reshape(hidden.size(1), -1) # reshape to (batch_size, hidden_size*2)

        output = self.fc(hidden)

        return output

    


if __name__ == "__main__":
    md = MakeData('data', '/ssd_scratch/cvit/anirudhkaushik/glove6b300dtxt/glove.6B.300d.txt')
    train_data, test_data, val_data = md.load_data()
    md.build_vocab(train_data)

    pretrain_train_dataset = PretrainDataset(train_data, md.vocab)
    pretrain_val_dataset = PretrainDataset(val_data, md.vocab)




    