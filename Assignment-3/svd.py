import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
import pickle as pkl
import collections
import time

train_df = pd.read_csv("/ssd_scratch/cvit/anirudhkaushik/iNLP-A2/ANLP-2/train.csv")
test_df = pd.read_csv("/ssd_scratch/cvit/anirudhkaushik/iNLP-A2/ANLP-2/test.csv")

# get only the Desciption column
train_df = train_df['Description']
test_df = test_df['Description']

threshold = 3

print(f"Train shape: {train_df.shape}\nTest shape: {test_df.shape}")

START_TOKEN = '<START>'
END_TOKEN = '<END>'
UNK_TOKEN = '<UNK>'

# append start and end tokens
train_df = train_df.apply(lambda x: f"{START_TOKEN} {x} {END_TOKEN}")
test_df = test_df.apply(lambda x: f"{START_TOKEN} {x} {END_TOKEN}")

train_df = train_df.to_list()[:20000]
test_df = test_df.to_list()

from nltk.tokenize import word_tokenize


def build_co_occurrence_matrix(corpus, window_size=5, threshold=2):
    #build unique words
    unique_words=set()
    
    # find all words that occur only once and remove them
    word_freq = collections.Counter()
    for text in tqdm(corpus, desc="building unique words"):
        words = word_tokenize(text)
        for word in words:
            word_freq[word] += 1

    unique_words = set([word for word, freq in word_freq.items() if freq >= threshold])
    unique_words.add(UNK_TOKEN)
    unique_words.add(START_TOKEN)
    unique_words.add(END_TOKEN)
    unique_words = sorted(list(unique_words))
    index_to_word = {i: word for i, word in enumerate(unique_words)}
    words_to_index = {word: i for i, word in enumerate(unique_words)}

    co_oc_matrix = np.zeros((len(words_to_index.keys()), len(words_to_index.keys())))

    for text in tqdm(corpus, desc="Building co-occurrence matrix"):
        words = word_tokenize(text)
        words = [START_TOKEN] + words + [END_TOKEN]
        for i, word in enumerate(words):
            if word not in words_to_index:
                word = UNK_TOKEN
            for j in range(max(0,i-window_size), min(i+window_size+1, len(words))):
                if j < 0 or j >= len(words) or i == j:
                    continue
                word2 = words[j]
                if words[j] not in words_to_index:
                    word2 = UNK_TOKEN
                co_oc_matrix[words_to_index[word], words_to_index[word2]] += 1
                co_oc_matrix[words_to_index[word2], words_to_index[word]] += 1

    # set diagonal to 0
    np.fill_diagonal(co_oc_matrix, 0)

    return index_to_word, co_oc_matrix



index_to_word, coo_dict=build_co_occurrence_matrix(train_df)

# save the co-occurrence matrix
with open("/ssd_scratch/cvit/anirudhkaushik/coo_dict_2.pkl", "wb") as f:
    pkl.dump(coo_dict, f)

# save the index to word mapping
with open("/ssd_scratch/cvit/anirudhkaushik/index_to_word_2.pkl", "wb") as f:
    pkl.dump(index_to_word, f)

# load 
with open("/ssd_scratch/cvit/anirudhkaushik/coo_dict_2.pkl", "rb") as f:
    coo_mat = pkl.load(f)

with open("/ssd_scratch/cvit/anirudhkaushik/index_to_word_2.pkl", "rb") as f:
    index_to_word = pkl.load(f)

print(coo_mat.shape)

# svd
start_time = time.time()
U, s, Vh = np.linalg.svd(coo_mat, full_matrices=False)
end_time = time.time()
print(f"Time taken for SVD: {end_time-start_time}s")
print(U.shape, s.shape, Vh.shape)
#print representation for word "this"
word_to_index = {word: i for i, word in enumerate(index_to_word.values())}

# save U matrix
U = U[:, :300] # reduce to 300 dimensions
with open("/ssd_scratch/cvit/anirudhkaushik/U_mat.pkl", "wb") as f:
    pkl.dump(U, f)