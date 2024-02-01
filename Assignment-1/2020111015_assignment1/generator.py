import pickle
import numpy as np


# display a pickle file

def display_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

f = display_pickle("europarl-corpusCentralDictTrain.pickle")
print(f[4])