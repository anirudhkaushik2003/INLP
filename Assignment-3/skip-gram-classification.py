import torch
import torch.nn as nn
import numpy 
import pandas as pd
from classification import RNN_Classifier, train_model, test_model, preprocess_data, NewsDataset
import numpy as np
from  skip_gram import Word2Vec, SkipGram

START_TOKEN = '<START>'
END_TOKEN = '<END>'
UNK_TOKEN = '<UNK>'

threshold = 3

train_df = pd.read_csv("/ssd_scratch/cvit/anirudhkaushik/iNLP-A2/ANLP-2/train.csv")
test_df = pd.read_csv("/ssd_scratch/cvit/anirudhkaushik/iNLP-A2/ANLP-2/test.csv")


train_df = train_df[:20000]
skipgram_data = train_df['Description'].to_list()

num_classes = np.unique(train_df['Class Index']).shape[0]

train_text = train_df['Description'].to_list()
test_text = test_df['Description'].to_list()

train_df['Class Index'] = train_df['Class Index'].apply(lambda x: x-1)
test_df['Class Index'] = test_df['Class Index'].apply(lambda x: x-1)


embedding_size = 100

skipgram = SkipGram(skipgram_data, window_size=10) 
word2vec = Word2Vec(len(skipgram.vocab), embedding_size)

# load word2vec
word2vec.load_state_dict(torch.load("/ssd_scratch/cvit/anirudhkaushik/word2vec.pth"))
embedding_matrix = nn.Embedding(len(skipgram.vocab)+1, embedding_size, _freeze=True)
torch.nn.init.constant_(embedding_matrix.weight[:,:], 0)
embedding_matrix.weight.data[1:,:].data.copy_(word2vec.w.weight.data)


# freeze the embedding matrix
embedding_matrix.weight.requires_grad = False


model = RNN_Classifier(len(skipgram.vocab), embedding_matrix, num_classes, embedding_size)

# prepare the data
train_text = preprocess_data(train_text, skipgram.words_to_index)
test_text = preprocess_data(test_text, skipgram.words_to_index)

train_data = list(zip(train_text, train_df['Class Index']))
test_data = list(zip(test_text, test_df['Class Index']))

train_dataset = NewsDataset(train_data)
test_dataset = NewsDataset(test_data)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=train_dataset.collate)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=test_dataset.collate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=30)
test_model(model, test_loader, device)

# save model
torch.save(model.state_dict(), "/ssd_scratch/cvit/anirudhkaushik/word2vec_rnn_classifier.pth")