from tokenizer import Tokenizer

class N_Gram_Model:
    def __init__(self,):
        self.file = None
        self.tokenized = None
        self.n_grams = []

    def read_file(self, file_path):
        self.file = open(file_path, 'r')

    def tokenize(self):
        T = Tokenizer(self.file.read())
        tokenized = T.tokenize()
        self.tokenized = tokenized
    
    def make_n_grams(self, n):
        self.tokenize()
        for sentence in self.tokenized:
            # add <SOS> and <EOS> tags
            if(len(sentence)>0):
                sentence = ["<SOS>" for _ in range(n-2)] + sentence + ["<EOS>"]
                # make n gram
                for i in range(len(sentence)-n+1):
                    self.n_grams.append(sentence[i:i+n]) 
                    
                    

    def setup(self):
        pass

    def train(self):
        pass
    
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
model.make_n_grams(4)