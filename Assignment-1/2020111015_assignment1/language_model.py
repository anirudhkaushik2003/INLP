from tokenizer import Tokenizer
import numpy as np

class N_Gram_Model:
    def __init__(self,):
        self.file = None
        self.tokenized = None
        self.n_grams = []
        self.freq_dict_n = {}
        self.freq_dict_n_minus_1 = {}
        self.train_sentences = []
        self.test_sentences = []

    def read_file(self, file_path):
        self.file = open(file_path, 'r')

    def tokenize(self):
        T = Tokenizer(self.file.read())
        tokenized = T.tokenize()
        self.tokenized = tokenized
    
    def sentence_to_ngrams(self, sentence, n):
        if(len(sentence)>0):
            sentence = ["<SOS>" for _ in range(n-2)] + sentence + ["<EOS>"]
            # make n gram
            for i in range(len(sentence)-n+1):
                self.n_grams.append(sentence[i:i+n]) 
                self.freq_dict_n[tuple(sentence[i:i+n])] = self.freq_dict_n.get(tuple(sentence[i:i+n]), 0) + 1 # convert to tuple since key is a list
                self.freq_dict_n_minus_1[tuple(sentence[i:i+n-1])] = self.freq_dict_n_minus_1.get(tuple(sentence[i:i+n-1]), 0) + 1

            # add the last n-1 grams
            self.freq_dict_n_minus_1[tuple(sentence[len(sentence)-n+1:])] = self.freq_dict_n_minus_1.get(tuple(sentence[len(sentence)-n+1:]), 0) + 1


    def make_n_grams(self, sentence_list):
        self.tokenize()
        for sentence in sentence_list:
            # add <SOS> and <EOS> tags
            self.sentence_to_ngrams(sentence, n)
            
    def calc_prob(self, sentence, n):
        # convert to n-gram
        T = Tokenizer(sentence)
        sentence = T.tokenize()[0]
        sentence = ["<SOS>" for _ in range(n-2)] + sentence + ["<EOS>"]
        # make n gram
        temp_n_grams = []
        temp_n_gram_minus_1 = []

        pr_sentence = 1
        for i in range(len(sentence)-n+1):
            n_gram = sentence[i:i+n]
            temp_n_grams.append(n_gram)
            n_gram_minus_1 = sentence[i:i+n-1]
            temp_n_gram_minus_1.append(n_gram_minus_1)
        
        n_gram_minus_1 = sentence[len(sentence)-n+1:]
        temp_n_gram_minus_1.append(n_gram_minus_1)
        
        # calculate P(w_n|w_1, w_2, ..., w_n-1)
        for i in range(len(temp_n_grams)):
            A = self.freq_dict_n.get(tuple(temp_n_grams[i]), 0)
            B = self.freq_dict_n_minus_1.get(tuple(temp_n_gram_minus_1[i]), 0)
            if B!=0:
                pr_sentence *= A/B
            else: 
                pr_sentence = 0
                break
        return pr_sentence
            

    def setup(self):
        test_indices = np.random.choice(len(self.tokenized), 1000, replace=False)
        self.train_sentences = []
        self.test_sentences = []
        for i in range(len(self.tokenized)):
            if i in test_indices:
                self.test_sentences.append(self.tokenized[i])
            else:
                self.train_sentences.append(self.tokenized[i])


    def train(self):
        self.make_n_grams(self.train_sentences)
    
    def save(self):
        pass

    def load(self):
        pass

    def perplexity(self, sentence):
        pass

    def generate(self):
        pass

    def evaluate(self):
        pass

    def good_turing(self):
        pass

    def interpolation(self):
        pass

model = N_Gram_Model()
model.read_file('Pride and Prejudice - Jane Austen.txt')
model.make_n_grams(3)
print(model.calc_prob("\"Oh! yes. Pray read on.\"", 3))