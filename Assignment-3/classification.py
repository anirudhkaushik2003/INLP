import torch 
import pandas as pd
import numpy as np
import torch.nn as nn

from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize

START_TOKEN = '<START>'
END_TOKEN = '<END>'
UNK_TOKEN = '<UNK>'



class RNN_Classifier(nn.Module):
    def __init__(self, vocabulary_size: int, embedding, num_classes, embedding_size):
        super().__init__()

        self.embedding = embedding
        self.lstm = nn.LSTM(embedding_size, embedding_size, batch_first=True, bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(embedding_size*2, 64),
            nn.ReLU(),
            torch.nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x, _= self.lstm(x)
        # get the last hidden state
        x = x[:,-1,:]
        x = self.linear(x)
        return x

class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        text = torch.tensor(text).long()
        label = torch.tensor(label).long()
        return text, label
    
    def collate(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a list of datapoints, batch them together"""
        sentences = [i[0] for i in batch]
        labels = torch.tensor([i[1] for i in batch])
        padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0) 

        return padded_sentences, labels

def preprocess_data(data, words_to_index):
    words = [k for k in words_to_index.keys()]
    words = ["<PAD>"] + words
    words_to_index = {word: i for i, word in enumerate(words)} 
    for i, sentence in enumerate(data):
        words = [START_TOKEN] + word_tokenize(sentence) + [END_TOKEN]
        for j, word in enumerate(words):
            if word not in words_to_index:
                words[j] = UNK_TOKEN
        data[i] = [words_to_index[word] for word in words]
    return data


def test_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(testloader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f"Accuracy: {correct/total}")

def train_model(model, trainloader, testloader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss/100}")
                running_loss = 0.0
        test_model(model, testloader, device)