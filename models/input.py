import pandas as pd
import json

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
import re

import torchtext
from collections import Counter

# Define start and end tokens
SOS = 0
EOS = 1
pad = 2
unk = 3

stop_words = set(stopwords.words('english'))
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

class QuestionSentenceDataset:
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {'<SOS>': SOS, '<EOS>': EOS, '<pad>': pad, '<unk>': unk}
            self.idx2word = {0: "<SOS>", 1: "<EOS>", 2:"<pad>", 3:"<unk>"}
            self.word_count = 4
            self.counter = Counter()
        else:
            self.word2idx = word2idx
            self.idx2word = {idx: word for (word, idx) in word2idx.items()}
            self.word_count = len(word2idx)

    def add_to_bag(self, sentence, origin="question"):
        for word in word_tokenize(sentence):
            word = self.preprocess_word(word, origin)
            if origin=="question":
                self.counter.update([word])
            elif word != "" and word not in stop_words and word not in punctuations:
                self.counter.update([word])
                
    def make_vocab(self, max_size=50000):
        limit = max_size - self.word_count
        sorted_by_freq_tuples = sorted(self.counter.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        for item in sorted_by_freq_tuples:
            word = item[0]
            self.word2idx[word] = self.word_count
            self.idx2word[self.word_count] = word
            self.word_count += 1
    
    def preprocess_word(self, word, origin="question"):
        word = word.lower()
        if type == "sentence":
            word =  re.sub('\W+','', word)
            word = contractions.fix(word)
        return word

if __name__=="__main__":
    DATA_DIR = 'data/'
    df = pd.read_csv(DATA_DIR+"qg_train.csv")
    train = QuestionSentenceDataset()

    for i, row in df.iterrows():
        input_sent = row['sentence']
        output_qn = row['question']
        train.add_to_bag(input_sent, origin="sentence")
        train.add_to_bag(output_qn, origin="question")
    
    train.make_vocab(max_size=50000)
        
    with open(DATA_DIR+'vocab.json', 'w', encoding='utf-8') as f:
        json.dump(train.word2idx, f)