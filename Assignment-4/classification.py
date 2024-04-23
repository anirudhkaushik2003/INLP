import torch
import numpy as np
import torch.nn as nn
from torchtext import vocab
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import pandas as pd
from ELMO import ELMO, NewsClassifierElmo