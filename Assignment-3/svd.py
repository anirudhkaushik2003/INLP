import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
import pickle as pkl
import collections


train_df = pd.read_csv("/ssd_scratch/cvit/anirudhkaushik/iNLP-A2/ANLP-2/train.csv")
test_df = pd.read_csv("/ssd_scratch/cvit/anirudhkaushik/iNLP-A2/ANLP-2/test.csv")

# get only the Desciption column
train_df = train_df['Description']
test_df = test_df['Description']

print(f"Train shape: {train_df.shape}\nTest shape: {test_df.shape}")

START_TOKEN = '<START>'
END_TOKEN = '<END>'
UNK_TOKEN = '<UNK>'

# append start and end tokens
train_df = train_df.apply(lambda x: f"{START_TOKEN} {x} {END_TOKEN}")
test_df = test_df.apply(lambda x: f"{START_TOKEN} {x} {END_TOKEN}")

train_df = train_df.to_list()[:20000]
test_df = test_df.to_list()

tester = [
    "i like deep learning.",
    "i like nlp.",
    "i enjoy flying."
]

vocabulary = ["i", "like", "enjoy", "deep", "learning", "nlp", "flying", "."]


from nltk.tokenize import word_tokenize


# def build_co_occurrence_matrix(corpus):
#     #build unique words
#     unique_words=set()
#     for text in tqdm(corpus, desc="Building unique words"):
#         for word in word_tokenize(text):
#             unique_words.add(word)
  
#     unique_words = sorted(list(unique_words))
#     index_to_word = {i: word for i, word in enumerate(unique_words)}
#     word_to_index = {word: i for i, word in enumerate(unique_words)}
#     pairs = []
#     cooc_mat = np.zeros((len(unique_words), len(unique_words)))

#     # count all pairs that occur together in a sentence
#     for text in tqdm(corpus, desc="Building co-occurrence matrix"):
#         words = word_tokenize(text)
#         for i, word in enumerate(words):
#             for j in range(i+1, len(words)):
#                 # pairs.append((word_to_index[word], word_to_index[words[j]]))
#                 cooc_mat[word_to_index[word], word_to_index[words[j]]] += 1
#                 cooc_mat[word_to_index[words[j]], word_to_index[word]] += 1

#     # counts = collections.Counter(pairs)
#     # ind = np.array(list(counts.keys())).T
#     # cooc_mat[ind[0], ind[1]] = list(counts.values())
                
                

#     return index_to_word, cooc_mat



# index_to_word, coo_dict=build_co_occurrence_matrix(train_df)

# # save the co-occurrence matrix
# with open("/ssd_scratch/cvit/anirudhkaushik/coo_dict_2.pkl", "wb") as f:
#     pkl.dump(coo_dict, f)

# # save the index to word mapping
# with open("/ssd_scratch/cvit/anirudhkaushik/index_to_word_2.pkl", "wb") as f:
#     pkl.dump(index_to_word, f)


# load 
with open("/ssd_scratch/cvit/anirudhkaushik/coo_dict_2.pkl", "rb") as f:
    coo_dict = pkl.load(f)

print(coo_dict.shape)

# svd
U, s, Vh = np.linalg.svd()