from tokenizer import Tokenizer
import numpy as np
import pickle
from arguments import get_args
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import os

class N_Gram_Model:
    def __init__(
        self,
    ):
        self.file = None
        self.tokenized = None
        self.n_grams = []
        self.freq_dict_n = {}
        self.freq_dict_n_minus_1 = {}
        self.train_sentences = []
        self.test_sentences = []
        self.unigrams = {}
        self.unk_threshold = 4
        self.unkown_unigrams = 0
        self.n_gram_probs = {}
        self.args = get_args()
        self.version = 1
        self.n = 3
        self.freq_freq = {}
        self._p0 = 0
        self.smooth_counts = {}
        self._count_models = None
        self.totals = 0
        self.recur_freq_dict = {N:{} for N in range(1, self.n+1)}
        self.lambdas = [0 for _ in range(self.n)]
        self.cache_dict = {}


        self.parse_args()
        self.read_file(self.corpus_path)
        self.tokenize()

        self.setup()
        if os.path.exists("models/2020111015_LM" + str(self.version) + ".pkl"):
            self.load()
            print(f"Number of N-grams: {len(self.n_grams) ,len(self.freq_dict_n.keys())}")

        else:
            self.train()
            print(f"Number of N-grams: {len(self.n_grams) ,len(self.freq_dict_n.keys())}")

            if self.lm_type == "g":
                self.good_turing()
                self.n_tokens, self.n_seen = sum(self.unigrams.values())-self.unigrams["<OOV>"], sum(self.freq_dict_n.values())
            if self.lm_type == "i":
                self.interpolation()
            
            self.save()


    def parse_args(self):
        self.lm_type = self.args.lm_type
        self.corpus_path = self.args.corpus_path
        self.k = self.args.k
        if "Pride" in self.corpus_path:
            if self.lm_type == "g":
                self.version = 1
            else:
                self.version = 2
        else:
            if self.lm_type == "g":
                self.version = 3
            else:
                self.version = 4

    def read_file(self, file_path):
        self.file = open(file_path, "r", encoding="utf-8")

    def tokenize(self):
        T = Tokenizer(self.file.read())
        tokenized = T.tokenize()
        self.tokenized = tokenized

    def make_unigrams(self, sentence_list) -> list:
        """
        find all unigrams and unknown words (freq <= threshold) and replace them with <OOV>
        """
        for i in range(len(sentence_list)):
            sentence = sentence_list[i]
            sentence = ["<SOS>" for _ in range(max(1,self.n - 2))] + sentence + ["<EOS>"]
            for j in range(len(sentence)):

                self.unigrams[sentence[j]] = (
                    self.unigrams.get(sentence[j], 0) + 1
                )

        for i in range(len(sentence_list)):
            for j in range(len(sentence_list[i])):
                if self.unigrams[sentence_list[i][j]] <= self.unk_threshold:
                    sentence_list[i][j] = "<OOV>"
                    self.unkown_unigrams += 1

        self.unigrams["<OOV>"] = self.unkown_unigrams

        return sentence_list

    def sentence_to_ngrams(self, sentence, n):
        if len(sentence) > 0:
            sentence = ["<SOS>" for _ in range(max(1, n - 2))] + sentence + ["<EOS>"]
            # make n gram
            for i in range(len(sentence) - n + 1):
                self.n_grams.append(tuple(sentence[i : i + n]))
                self.freq_dict_n[tuple(sentence[i : i + n])] = (
                    self.freq_dict_n.get(tuple(sentence[i : i + n]), 0) + 1
                )  # convert to tuple since key is a list
                self.freq_dict_n_minus_1[tuple(sentence[i : i + n - 1])] = (
                    self.freq_dict_n_minus_1.get(tuple(sentence[i : i + n - 1]), 0) + 1
                )

            # add the last n-1 grams
            self.freq_dict_n_minus_1[tuple(sentence[len(sentence) - n + 1 :])] = (
                self.freq_dict_n_minus_1.get(
                    tuple(sentence[len(sentence) - n + 1 :]), 0
                )
                + 1
            )

            # remove duplicate n-grams

    def make_n_grams(self, sentence_list, n) -> None:
        sentence_list = self.make_unigrams(sentence_list)
        print("Generating n-grams...")
        for sentence in tqdm(sentence_list):
            # add <SOS> and <EOS> tags
            self.sentence_to_ngrams(sentence, n)
        self.n_grams = list(set(tuple(gram) for gram in self.n_grams))

    def recursive_n_gram_gen(self, sentence_list)-> None:
        '''
        call after make_n_grams
        '''
        self.recur_freq_dict[self.n] = self.freq_dict_n
        self.recur_freq_dict[1] = self.unigrams
        if self.n > 1:
            self.recur_freq_dict[self.n-1] = self.freq_dict_n_minus_1

        print("Generating 1 to n-grams...")
        for k in range(2, self.n-1):
            for sentence in tqdm(sentence_list):
                if len(sentence) > 0:
                    sentence = ["<SOS>" for _ in range(max(1, k - 2))] + sentence + ["<EOS>"]
                    # make n gram
                    for i in range(len(sentence) - k + 1):
                        self.recur_freq_dict[k][tuple(sentence[i : i + k])] = self.recur_freq_dict[k].get(tuple(sentence[i : i + k]), 0) + 1



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
        self.make_n_grams(self.train_sentences, self.n)
        for n_gram in self.n_grams:
            if len(n_gram) > 1:
                B = self.freq_dict_n_minus_1[tuple(n_gram[:-1])]
                if B > 0:
                    n_gram_prob = self.freq_dict_n[tuple(n_gram)] / B
                    self.n_gram_probs[
                        tuple(n_gram)
                    ] = n_gram_prob  # save the n gram probability

    def save(self):
        # save the probabilities and frequency in a file
        filename = "models/2020111015_LM" + str(self.version) + ".pkl"
        if self.version == 1 or self.version == 3: # good turing
            model_dict = {
                "n_gram_probs": self.n_gram_probs,
                "freq_dict_n": self.freq_dict_n,
                "freq_dict_n_minus_1": self.freq_dict_n_minus_1,
                "unigrams": self.unigrams,
                "freq_freq": self.freq_freq,
                "smooth_counts": self.smooth_counts,
                "totals": self.totals,
                "_p0": self._p0,
                "unkown_unigrams": self.unkown_unigrams,
                "count_models": self._count_models,
                "n_tokens": self.n_tokens,
                "n_seen": self.n_seen
            }

        elif self.version == 2 or self.version == 4: # linear interpolation
            model_dict = {
                "n_gram_probs": self.n_gram_probs,
                "freq_dict_n": self.freq_dict_n,
                "freq_dict_n_minus_1": self.freq_dict_n_minus_1,
                "unigrams": self.unigrams,
                "lambdas": self.lambdas,
                "recur_freq_dict": self.recur_freq_dict,
                "unkown_unigrams": self.unkown_unigrams
            }

        print(f"Saving model to {filename}")
        pickle.dump(model_dict, open(filename, "wb"))


            

    def load(self):
        # load the probabilities
        filename = "models/2020111015_LM" + str(self.version) + ".pkl"
        model_dict = pickle.load(open(filename, "rb"))
        if self.version == 1 or self.version == 3: # good turing
            self.n_gram_probs = model_dict["n_gram_probs"]
            self.freq_dict_n = model_dict["freq_dict_n"]
            self.freq_dict_n_minus_1 = model_dict["freq_dict_n_minus_1"]
            self.unigrams = model_dict["unigrams"]
            self.freq_freq = model_dict["freq_freq"]
            self.smooth_counts = model_dict["smooth_counts"]
            self.totals = model_dict["totals"]
            self._p0 = model_dict["_p0"]
            self.unkown_unigrams = model_dict["unkown_unigrams"]
            self._count_models = model_dict["count_models"]
            self.n_tokens = model_dict["n_tokens"]
            self.n_seen = model_dict["n_seen"]
            
        elif self.version == 2 or self.version == 4: # linear interpolation
            self.n_gram_probs = model_dict["n_gram_probs"]
            self.freq_dict_n = model_dict["freq_dict_n"]
            self.freq_dict_n_minus_1 = model_dict["freq_dict_n_minus_1"]
            self.unigrams = model_dict["unigrams"]
            self.lambdas = model_dict["lambdas"]
            self.recur_freq_dict = model_dict["recur_freq_dict"]
            self.unkown_unigrams = model_dict["unkown_unigrams"]



    def perplexity(self, sentence):
        for i in range(len(sentence)):
            if sentence[i] not in self.unigrams.keys():
                sentence[i] = "<OOV>"

        sentence = ["<SOS>" for _ in range(max(1,self.n - 2))] + sentence + ["<EOS>"]
        # make n gram
        temp_n_grams = []
        temp_n_gram_minus_1 = []

        log_pr_sentence = 0
        for i in range(len(sentence) - self.n + 1):
            n_gram = sentence[i : i + self.n]
            temp_n_grams.append(n_gram)
            n_gram_minus_1 = sentence[i : i + self.n - 1]
            temp_n_gram_minus_1.append(n_gram_minus_1)

        n_gram_minus_1 = sentence[len(sentence) - self.n + 1 :]
        temp_n_gram_minus_1.append(n_gram_minus_1)

        if self.lm_type == "g":
            for i in temp_n_grams:
                log_pr_ngram = self.gt_log_ngram_probs(i)
                log_pr_sentence += log_pr_ngram
        elif self.lm_type == "i":
            for i in temp_n_grams:
                log_pr_ngram = self.i_log_ngram_probs(i)
                log_pr_sentence += log_pr_ngram
        else: # unsmoothed
            # calculate P(w_n|w_1, w_2, ..., w_n-1)
            for i in range(len(temp_n_grams)):
                A = self.freq_dict_n.get(tuple(temp_n_grams[i]), 0)
                B = self.freq_dict_n_minus_1.get(tuple(temp_n_gram_minus_1[i]), 0)
                if B != 0:
                    log_pr_sentence += (np.log(A) - np.log(B))

        ### Check this boundary condition once
        # perplexity = -1*log_pr_sentence
        # perplexity = (1/len(sentence))*perplexity
        # perplexity = np.exp(perplexity)
                    
        perplexity = np.exp(-1 * log_pr_sentence / len(sentence))
        return perplexity

    def generate(self, k):
        words = input("input sentence: ")
        T = Tokenizer(words)
        words = T.tokenize()[0]
        words = ["<SOS>" for _ in range(max(1,self.n - 2))] + words 
        words = words[-(self.n-1):]
        outputs = {}
        for unigrams in self.unigrams.keys():
            if unigrams == "<SOS>" or unigrams == "<EOS>" or unigrams == "<OOV>":
                continue
            if self.lm_type == "g":
                prob = np.exp(self.gt_log_ngram_probs(words + [unigrams]))
                outputs[unigrams] = prob
            elif self.lm_type == "i":
                prob = np.exp(self.i_log_ngram_probs(words + [unigrams]))
                outputs[unigrams] = prob
        # sort as per probability
        outputs = {l: v for l, v in sorted(outputs.items(), key=lambda item: item[1], reverse=True)}
        
        print(f"Top {k} predictions: ")
        for i, (key, value) in enumerate(outputs.items()):
            print(f"{key}: {value}")
            if i == k-1:
                break

    def sentence_probability(self):
        words = input("input sentence: ")
        T = Tokenizer(words)
        words = T.tokenize()[0]

        for i  in range(len(words)):
            word = words[i]
            if word not in self.unigrams.keys():
                words[i] = "<OOV>"
        words = ["<SOS>" for _ in range(max(1,self.n - 2))] + words + ["<EOS>"]
        prob = 0
        if self.lm_type == "g":
            for i in range(len(words) - self.n + 1):
                prob += self.gt_log_ngram_probs(words[i : i + self.n])
            
            prob = np.exp(prob)
        elif self.lm_type == "i":
            for i in range(len(words) - self.n + 1):
                prob += self.i_log_ngram_probs(words[i : i + self.n])
            
            prob = np.exp(prob)

        print(f"Probability of sentence: {prob}")
            


    def evaluate(self):
        filename_train = (
            "perplexity/2020111015_LM" +str( self.version) + "_train-perplexity.txt"
        )
        filename_test = (
            "perplexity/2020111015_LM" +str( self.version) + "_test-perplexity.txt"
        )

        print("\n\n\n##############################################")
        print("Evaluating...")

        train_scores = []
        for sentence in tqdm(self.train_sentences):
            score = self.perplexity(sentence)
            train_scores.append(score)

        avg_train_perplexity = np.mean(train_scores)
        with open(filename_train, "w", encoding="utf-8") as f:
            f.write(str(avg_train_perplexity) + "\n")
            # write sentence \t perplexity
            for i in range(len(self.train_sentences)):
                f.write(
                    str(self.train_sentences[i]) + "\t" + str(train_scores[i]) + "\n"
                )

        test_scores = []
        for sentence in tqdm(self.test_sentences):
            score = self.perplexity(sentence)
            test_scores.append(score)

        avg_test_perplexity = np.mean(test_scores)
        with open(filename_test, "w", encoding="utf-8") as f:
            f.write(str(avg_test_perplexity) + "\n")
            # write sentence \t perplexity
            for i in range(len(self.test_sentences)):
                f.write(str(self.test_sentences[i]) + "\t" + str(test_scores[i]) + "\n")

    def good_turing(self):
        totals = 0
        self.calc_freq_freq()
        self._p0 = self.freq_freq[1] / sum(self.freq_dict_n_minus_1.values())
        
        self.fit_model()
        use_interp = False
        self.smooth_counts[0] = 1

        for count in sorted(set(self.freq_dict_n.values())):
            c1_lm = np.exp(self._count_models.predict(np.c_[np.log(count+1)])).item() ## for a given r predict Zr
            c0_lm = np.exp(self._count_models.predict(np.c_[np.log(count)])).item()
            count_interp = ((count + 1) * c1_lm )/ c0_lm ## interpolate between Zr+1 and Zr

            c1, c0 = self.freq_freq.get(count + 1, 0), self.freq_freq.get(count, 0)
            if use_interp or c1 == 0:
                use_interp = True
                self.smooth_counts[count] = count_interp
                totals += c0*self.smooth_counts[count]
                continue

            count_emp = ((count + 1) * c1) / c0 # smoothed count c*
            t = 1.96** np.sqrt((count + 1) ** 2 * (c1 / c0 ** 2) * (1 + c1 / c0))

            # if the difference between the empirical and interpolated
            # smoothed counts is greater than t, the empirical estimate
            # tends to be more accurate. otherwise, use interpolated

            if abs(count_emp - count_interp) > t:
                self.smooth_counts[count] = count_emp
                totals += c0*self.smooth_counts[count]
                continue
            
            use_interp = True
            self.smooth_counts[count] = count_interp
            totals += c0*self.smooth_counts[count]
            
        self._p0 = self.smooth_counts[1]/ self.smooth_counts[0]
        self.totals = totals

    def gt_log_ngram_probs(self, ngram):
        N = len(ngram)


        # approx. prob of an out-of-vocab ngram (i.e., a fraction of p0)
        
        bigram = list(ngram)[:-1]
        sigma_c_star = 0
        if tuple(bigram) in self.cache_dict.keys():
            sigma_c_star = self.cache_dict[tuple(bigram)]

        else: # calculate sigma_c_star
            for unigrams in self.unigrams.keys():
                        
                words = tuple(bigram + [unigrams])
                
                if words in self.freq_dict_n:
                    cp = self.freq_dict_n[words]
                    cp_1 = cp+1
                    cp1_lm = np.exp(self._count_models.predict(np.c_[np.log(cp_1+1)])).item()
                    cp0_lm = np.exp(self._count_models.predict(np.c_[np.log(cp_1)])).item()
                    cp_1 = ((cp_1 + 1) * cp1_lm )/ cp0_lm
                    cp_star = ((cp+1)*cp_1/self.smooth_counts[cp])
                    sigma_c_star += cp_star

                else:
                    sigma_c_star += self.smooth_counts[1]

            self.cache_dict[tuple(bigram)] = sigma_c_star # bigram sigma of w1w2 over all unigrams in vocab

        prob = np.log(self._p0 / sigma_c_star)
        # prob = np.log(self._p0)

        if tuple(ngram) in self.freq_dict_n:
            c = self.freq_dict_n[tuple(ngram)]
            c_1 = c+1
            c1_lm = np.exp(self._count_models.predict(np.c_[np.log(c_1+1)])).item()
            c0_lm = np.exp(self._count_models.predict(np.c_[np.log(c_1)])).item()

            c_1 = ((c_1 + 1) * c1_lm )/ c0_lm
            c_star = ((c+1)*c_1/self.smooth_counts[c])

            ### TOOO SLOW ###
            sigma_c_star = 0
            bigram = list(ngram)[:-1]
            if tuple(bigram) in self.cache_dict.keys():
                sigma_c_star = self.cache_dict[tuple(bigram)]
            else:
                for unigrams in self.unigrams.keys():
                    
                    words = tuple(bigram + [unigrams])
                    
                    if words in self.freq_dict_n:
                        cp = self.freq_dict_n[words]
                        cp_1 = cp+1
                        cp1_lm = np.exp(self._count_models.predict(np.c_[np.log(cp_1+1)])).item()
                        cp0_lm = np.exp(self._count_models.predict(np.c_[np.log(cp_1)])).item()
                        cp_1 = ((cp_1 + 1) * cp1_lm )/ cp0_lm
                        cp_star = ((cp+1)*cp_1/self.smooth_counts[cp])
                        sigma_c_star += cp_star

                    else:
                        sigma_c_star += self.smooth_counts[1]

                self.cache_dict[tuple(bigram)] = sigma_c_star

            prob = np.log(c_star) - np.log(sigma_c_star) 

            # prob = np.log(1-self._p0) + np.log(c_star) - np.log(self.totals) 

        return prob
    

    def calc_freq_freq(self):
        # sort the n-grams by frequency
        sorted_n_grams = sorted(self.freq_dict_n.items(), key=lambda x: x[1])
        for i in range(len(sorted_n_grams)):
            self.freq_freq[sorted_n_grams[i][1]] = self.freq_freq.get(
                sorted_n_grams[i][1], 0
            ) + 1
            

    def fit_model(self):
        # calculate zr and fit log zr log r plot
        X, Y = [], []
        sorted_counts = sorted(set(self.freq_dict_n.values()))  # r

        for ix, j in enumerate(sorted_counts):
            i = 0 if ix == 0 else sorted_counts[ix - 1]
            k = 2 * j - i if ix == len(sorted_counts) - 1 else sorted_counts[ix + 1]
            y = 2 * self.freq_freq[j] / (k - i)
            X.append(j)
            Y.append(y)

        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)
        # fit log-linear model: log(counts) ~ log(average_transform(counts))
        self._count_models = LinearRegression(fit_intercept=True)
        self._count_models.fit(np.log(X), np.log(Y))
        

    def interpolation(self):
        self.recursive_n_gram_gen(self.train_sentences)
        cases = []
        self.lambdas = [0 for _ in range(self.n)]
        for ngram in self.n_grams:
            cases = []
            ngram_k = ngram
            for k in range(0, len(ngram)):
                ngram_k = ngram[k:]

                if(len(ngram_k) == 1):
                    if ngram_k[0] not in self.recur_freq_dict[1]:
                        ca = 0
                        cases.append(ca)
                        continue
                    if ngram_k[0] == "<OOV>":
                        ca = ((self.unkown_unigrams-1)/ (sum(self.unigrams.values())-1))
                    else:
                        ca = (self.recur_freq_dict[1][ngram_k[0]]-1)/(sum(self.unigrams.values())-1)
                    cases.append(ca)
                    continue
                else:

                    if tuple(ngram_k) not in self.recur_freq_dict[len(ngram_k)]:
                        ca = 0
                        cases.append(ca)
                        continue

                    f = self.recur_freq_dict[len(ngram_k)][tuple(ngram_k)]

                    if tuple(ngram_k[:-1]) not in self.recur_freq_dict[len(ngram_k)-1]:
                        ca = 0
                        cases.append(ca)
                        continue

                    f_n_1 = self.recur_freq_dict[len(ngram_k)-1][tuple(ngram_k[:-1])]

                    if(f_n_1 ==0 or f == 0):
                        ca = 0
                        cases.append(ca)
                        continue

                    if(len(ngram_k) == 1):
                        ca = (f-1)/(sum(self.unigrams.values())-1)
                    else:
                        try:
                            ca = (f -1)/(f_n_1-1)
                        except:
                            ca = 0
                    cases.append(ca)
            # find index of max case
            max_case = max(cases)
            max_index = cases.index(max_case)
            self.lambdas[self.n - 1 - max_index] += self.freq_dict_n[tuple(ngram)]
        
        # normalize lambdas
        self.lambdas = [i/sum(self.lambdas) for i in self.lambdas]

    def i_log_ngram_probs(self, ngram):
        prob = 0
        for i in range(0, len(ngram)):
            if(i == 0):
                prob += self.lambdas[i]*self.recur_freq_dict[i+1][ngram[len(ngram)-i-1:][0]]/ sum(self.unigrams.values())
            else:
                try:
                    prob += self.lambdas[i]*(self.recur_freq_dict[i+1][tuple(ngram[len(ngram)-i-1:])]/self.recur_freq_dict[i][tuple(ngram[len(ngram)-i:])])
                except:
                    if i == 1:
                        try:
                            prob += self.lambdas[i]*(self.recur_freq_dict[i+1][tuple(ngram[len(ngram)-i-1:])]/self.recur_freq_dict[i][ngram[len(ngram)-i:][0]])
                        except:
                            pass
        return np.log(prob)

            



if __name__ == "__main__":
    model = N_Gram_Model()
    model.sentence_probability()