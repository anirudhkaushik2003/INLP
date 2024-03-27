import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
import pickle as pkl
import collections

#pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import one_hot

from sklearn.model_selection import train_test_split

from nltk.tokenize import word_tokenize


START_TOKEN = '<START>'
END_TOKEN = '<END>'
UNK_TOKEN = '<UNK>'



class SkipGram:
    def __init__(self, corpus, window_size=1,):
        self.corpus = corpus
        self.window_size = window_size
        self.vocab = None
        self.words_to_index = None
        self.co_oc_matrix = None
        self.threshold = 2
        self.k = 2

        self.build_vocab()
        
    def build_vocab(self,):
        unique_words = set()
        word_freq = collections.Counter()
        for text in tqdm(self.corpus, desc="Building unique words"):
            words = word_tokenize(text)
            for word in words:
                word_freq[word] += 1

        unique_words = set([word for word, freq in word_freq.items() if freq >= self.threshold])
        unique_words.add(UNK_TOKEN)
        unique_words.add(START_TOKEN)
        unique_words.add(END_TOKEN)

        self.vocab = sorted(list(unique_words))
        print(f"Vocab size: {len(self.vocab)}")
        self.words_to_index = {word: i for i, word in enumerate(self.vocab)}

    def build_co_oc_matrix(self):
        self.co_oc_matrix = np.zeros((len(self.vocab), len(self.vocab)))

        for text in tqdm(self.corpus, desc="Building co-occurrence matrix"):
            words = word_tokenize(text)
            words = [START_TOKEN] + words + [END_TOKEN]
            for i, word in enumerate(words):
                if word not in self.words_to_index:
                    word = UNK_TOKEN
                for j in range(max(0,i-self.window_size), min(i+self.window_size+1, len(words))):
                    if j < 0 or j >= len(words) or i == j:
                        continue
                    word2 = words[j]
                    if words[j] not in self.words_to_index:
                        word2 = UNK_TOKEN
                    self.co_oc_matrix[self.words_to_index[word], self.words_to_index[word2]] += 1
                    self.co_oc_matrix[self.words_to_index[word2], self.words_to_index[word]] += 1

        np.fill_diagonal(self.co_oc_matrix, 0)

    def build_train_pair(self):
        X = []
        y = []
        self.build_co_oc_matrix()


        for i, _ in tqdm(enumerate(self.vocab), desc="Building training pairs", total=len(self.vocab)):

            pos_context_indices = list(np.where(self.co_oc_matrix[i] > 0)[0])
            neg_context_indices = list(np.where(self.co_oc_matrix[i] == 0)[0])


            if i in pos_context_indices:
                pos_context_indices.remove(i)
            if i in neg_context_indices:
                neg_context_indices.remove(i)


            X.extend(list(zip([i]*len(pos_context_indices), pos_context_indices)))
            y.extend(list(zip([i]*min(self.k*len(pos_context_indices), len(neg_context_indices)), np.random.choice(neg_context_indices, min(self.k*len(pos_context_indices), len(neg_context_indices)), replace=False))))

        return X, y

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.w = nn.Embedding(vocab_size, embedding_dim)
        self.c = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, word, context):
        # given word, context_word: predict probability word occurs with context
        # return torch.sigmoid(torch.dot(self.w(word), self.c(context)))
        # return torch.sigmoid(torch.bmm(self.w(word).unsqueeze(1), self.c(context).unsqueeze(2)).squeeze())
        word, context = self.w(word), self.c(context)
        nr, nc = word.shape[0], word.shape[1]
        x = torch.bmm(word.view(nr, 1, nc), context.view(nr, nc, 1))
        return x.flatten()

class Word2VecDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (word, context), label = self.data[idx]
        word = torch.tensor(word).long()
        context = torch.tensor(context).long()
        label = torch.tensor(label).float()
        return (word, context), label

def test_model(model, device, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(testloader):
            word, context = data
            word, context, target = word.to(device), context.to(device), target.to(device)
            outputs = model(word, context) # returns probability of word occuring with context
            # probability should be 1 for similar pairs and 0 for dissimilar pairs
            # apply logit
            outputs = torch.sigmoid(outputs)
            predicted = torch.round(outputs)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Accuracy: {100*correct/total}")

def train_model(model, trainloader, testloader, optimizer, criterion, device, epochs=10):
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (data, target) in tqdm(enumerate(trainloader), desc=f"Epoch {epoch+1}", total=len(trainloader)):
            word, context = data            
            word, context, target = word.to(device), context.to(device), target.to(device)
            optimizer.zero_grad()

            outputs = model(word, context)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # if i % 1000 == 999:
            #     print(f"Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss/100}")
            #     running_loss = 0.0


        print(f"Epoch: {epoch+1}, Loss: {running_loss/len(trainloader)}")
        test_model(model,device, testloader)


if __name__ == "__main__":
    train_df = pd.read_csv("/ssd_scratch/cvit/anirudhkaushik/iNLP-A2/ANLP-2/train.csv")
    test_df = pd.read_csv("/ssd_scratch/cvit/anirudhkaushik/iNLP-A2/ANLP-2/test.csv")

    # get only the Desciption column
    train_df = train_df['Description']
    test_df = test_df['Description']


    train_df = train_df.to_list()[:20000]
    test_df = test_df.to_list()
    
    word_size = 7

    sg = SkipGram(train_df, window_size=word_size)

    X, y = sg.build_train_pair()

    train_data_x = []
    train_data_y = []

    for i in range(len(X)):
        train_data_x.append((X[i][0], X[i][1])) # similar
        train_data_y.append(1)
    for i in range(len(y)):
        train_data_x.append((y[i][0], y[i][1])) # dissimilar
        train_data_y.append(0)

    X_train, X_val, y_train, y_val = train_test_split(train_data_x, train_data_y, test_size=0.02, random_state=42)

        

    train_dataset = Word2VecDataset(list(zip(X_train, y_train)))
    test_dataset = Word2VecDataset(list(zip(X_val, y_val)))

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    print(f"Train size: {len(train_dataset)}\nTest size: {len(test_dataset)}")
    print(f"Train Batches: {len(train_loader)}\nTest Batches: {len(test_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Word2Vec(len(sg.vocab), 100)
    # model = torch.nn.DataParallel(model)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, test_loader, optimizer, criterion, device, epochs=10)
    test_model(model, device, test_loader)

    # save model
    torch.save(model.state_dict(), f"/ssd_scratch/cvit/anirudhkaushik/word2vec_{word_size}.pth")