import re

# class Tokenizer:
#     def __init__(self, text):
#         self.text = text
#         self.tokens = []

#         self.sentence_tokenizer = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')


sentence = "Hello, Mr. Smith, how are you doing today? The weather is great, and city is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."

sentence_tokenizer1 = re.compile(r'(?<!\w\.\w.)')
tokenized = sentence_tokenizer1.split(sentence)