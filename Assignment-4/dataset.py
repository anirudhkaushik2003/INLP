from torch.utils.data import Dataset, DataLoader
import pandas as pd
from nltk.tokenize import word_tokenize
import torch
import torchtext
from tqdm import tqdm
import numpy as np
import os


class MakeData():
    def __init__(self, data_path, glove_path):
        self.data_path = data_path
        self.glove_path = glove_path
        self.special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']


    def load_data(self):
        train_data = pd.read_csv(f'{self.data_path}/train.csv')
        test_data = pd.read_csv(f'{self.data_path}/test.csv')


        val_data = train_data[20000:20000+len(test_data)]
        train_data = train_data[:20000]
        
        print(f"Shape of train data: {train_data.shape}")
        print(f"Shape of test data: {test_data.shape}")
        print(f"Shape of val data: {val_data.shape}")

        return train_data, test_data, val_data
    
    def build_vocab(self, data):
        word_list = []
        len_vocab = 4

        for i in tqdm(range(len(data)), desc="Building Vocabulary..."):
            for word in word_tokenize(data['Description'][i]):
                word_list.append(word.lower())

        self.vocab = torchtext.vocab.build_vocab_from_iterator(word_list, specials=self.special_tokens)
        self.vocab.set_default_index(self.vocab['<UNK>']) # set default index to <UNK> token -> index to return when oov word is encountered


class PretrainDataset(Dataset):
    def __init__(self, data, vocab, ):
        self.data = data
        self.vocab = vocab
        self.max_len = 0
        self.special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        self.forward_data = []
        self.backward_data = []
        self.forward_targets = []
        self.backward_targets = []

    def build(self):

        for i in tqdm(range(len(self.data)), desc="Building Pretrain Dataset..."):
            words = word_tokenize(self.data['Description'][i])
            
            word_list = []
            index_list = []

            for word in words:
                word_list.append(word.lower())
                index_list.append(self.vocab[word.lower()])

            self.forward_data += list(index_list[:i] for i in range(1, len(index_list)))
            self.forward_targets += index_list[1:]

            index_list = list(reversed(index_list))
            self.backward_data += list(index_list[:i] for i in range(1, len(index_list)))
            self.backward_targets += index_list[1:]

            if (len(index_list)-1) > self.max_len:
                self.max_len = (len(index_list)-1)



        # pad the sequences
        self.forward_data = [i + [self.vocab['<PAD>']]*(self.max_len-len(i)) for i in self.forward_data]
        self.backward_data = [i + [self.vocab['<PAD>']]*(self.max_len-len(i)) for i in self.backward_data]

        self.forward_data = torch.tensor(self.forward_data)
        self.backward_data = torch.tensor(self.backward_data)
        self.forward_targets = torch.tensor(self.forward_targets)
        self.backward_targets = torch.tensor(self.backward_targets)


    def __len__(self):
        return len(2*self.forward_data)
    
    def __getitem__(self, idx):
        return self.forward_data[idx], self.forward_targets[idx], self.backward_data[idx], self.backward_targets[idx]
    

def GloveEmbeddings(glove_path, vocab):
    def get_unk(v):
        return torch.mean(v, dim=0)
    
    glove = torchtext.vocab.GloVe(name='6B', dim=300)
    unk = get_unk(glove.vectors)

    embeddings = []
    for word in tqdm(vocab.get_itos(), desc="Building Embeddings..."):
        if word in glove.itos:
            embeddings.append(glove[word])
        else:
            embeddings.append(unk)

    return torch.stack(embeddings)
