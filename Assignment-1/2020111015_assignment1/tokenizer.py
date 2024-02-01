import re


class Tokenizer:
    def __init__(self, text):
        self.text = text
        self.sentence_tokens = []
        self.word_tokens = []
        self.email_tokens = []
        self.url_tokens = []
        self.hashtag_tokens = []
        self.mention_tokens = []
        self.email_placeholder = '<EMAIL>'
        self.url_placeholder = '<URL>'
        self.hashtag_placeholder = '<HASHTAG>'
        self.mention_placeholder = '<MENTION>'
        self.punc_placeholder = '<PUNC>'

        self.word_pattern = r"""
        (?:[A-Z]\.)+            # abbreviations, e.g. U.S.A.
        | (?:\<EMAIL\>|\<URL\>|\<HASHTAG\>|\<MENTION\>)
        | (?:\w+\:\w+)
        | \$?\d+(?:\.\d+)?%?    # currency and percentages, $12.40, 50%
        | \w+(?:'[a-z])         # words with apostrophes
        | \.\.\.                # ellipsis
        |(?:[A-Z][a-z])\.     # honorifics
        |(?:[A-Z][a-z][a-z])\.     # honorifics
        | \w+                   # normal words
        | [\]\[\.\,\;\"\'\?\(\)\:\-\_\`\!]      # these are separate tokens; includes ], [
        """

        self.sentence_tokenizer = re.compile(
            r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z][a-z][a-z]\.)(?<=\.|\?|\!)\s"
        )  # abbreviations, honorifics, split by punc ., ?, !
        self.word_tokenizer = re.compile(self.word_pattern, re.VERBOSE | re.IGNORECASE)
        self.email_tokenizer = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}(?<![.,!?])\b')
        self.url_tokenizer = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+(?<![.,!?])')
        self.hashtag_tokenizer = re.compile(r'#\w+')
        self.mention_tokenizer = re.compile(r'@\w+')
        self.punc_tokenizer = re.compile(r'[\]\[\.\,\;\"\'\?\(\)\:\-\_\`\!]')


    def tokenize_sentence(self)->list:
        self.sentence_tokens = self.sentence_tokenizer.split(self.text)

        return self.sentence_tokens
    
    def tokenize_email(self)->list:
        '''Tokenize emails in text'''
        self.email_tokens = self.email_tokenizer.findall(self.text)

        return self.email_tokens
    
    def tokenize_url(self)->list:
        '''Tokenize urls in text'''
        self.url_tokens = self.url_tokenizer.findall(self.text)

        return self.url_tokens
    
    def tokenize_hashtag(self)->list:
        '''Tokenize hashtags in text'''
        self.hashtag_tokens = self.hashtag_tokenizer.findall(self.text)

        return self.hashtag_tokens
    
    def tokenize_mention(self)->list:
        '''Tokenize mentions in text'''
        self.mention_tokens = self.mention_tokenizer.findall(self.text)

        return self.mention_tokens
    
    def tokenize_punc(self)->list:
        '''Tokenize punctuation in text'''
        self.punc_tokens = self.punc_tokenizer.findall(self.text)

        return self.punc_tokens
    
    def replace_with_placeholder(self,):
        

        # replace tokens with placeholder in sentence tokens
        for i in range(len(self.sentence_tokens)):
            self.sentence_tokens[i] = self.email_tokenizer.sub(self.email_placeholder, self.sentence_tokens[i])
            self.sentence_tokens[i] = self.url_tokenizer.sub(self.url_placeholder, self.sentence_tokens[i])
            self.sentence_tokens[i] = self.hashtag_tokenizer.sub(self.hashtag_placeholder, self.sentence_tokens[i])
            self.sentence_tokens[i] = self.mention_tokenizer.sub(self.mention_placeholder, self.sentence_tokens[i])
            self.sentence_tokens[i] = self.punc_tokenizer.sub(self.punc_placeholder, self.sentence_tokens[i])

            

    def tokenize_words(self)->list:
        '''Tokenize words in text sentence by sentence'''
        for sentence in self.sentence_tokens:
            self.word_tokens.append(self.word_tokenizer.findall(sentence))

        return self.word_tokens
    
    def tokenize(self)->list:
        self.tokenize_sentence()
        self.replace_with_placeholder()
        return self.tokenize_words()

sentence = "Mrs. Smith bought cheapsite.com's o-k asshole for $12.40 1.5-million dollars, i.e. he paid a lot for it at 13:30pm. Did he mind? Adam Jones Jr. thinks he didn't. In any case, this isn't true... Well, with a probability of .9 it isn't. Hey @john, check out this amazing website: https://example.com! #omg #exciting."


if __name__ == "__main__":
    inp = input("your text: ")
    print(inp)
    T = Tokenizer(inp)
    tokenized = T.tokenize()
    print("tokenized text: ", tokenized)

