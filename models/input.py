import pandas as pd
from nltk.tokenize import word_tokenize
import json

# Define start and end tokens
SOS = 0
EOS = 1
pad = 2
unk = 3

class QuestionSentenceDataset:
    def __init__(self):
        self.word2idx = {'<SOS>': SOS, '<EOS>': EOS, '<pad>': pad, '<unk>': unk}
        self.word2freq = {}
        self.idx2word = {0: "<SOS>", 1: "<EOS>", 2:"<pad>", 3:"<unk>"}
        self.word_count = 4

    def add_to_vocab(self, sentence):
        # preprocess words
        for word in word_tokenize(sentence):
            word = word.lower()
            
            if word not in self.word2idx:
                self.word2idx[word] = self.word_count
                self.word2freq[word] = 1
                self.idx2word[self.word_count] = word
                self.word_count += 1
            else:
                self.word2freq[word] += 1

if __name__=="__main__":
    DATA_DIR = '../data/'
    df = pd.read_csv(DATA_DIR+"qg_train.csv")
    
    train = QuestionSentenceDataset()
    
    for i, row in df.iterrows():
        input_sent = row['sentence']
        output_qn = row['question']
        train.add_to_vocab(input_sent)
        train.add_to_vocab(output_qn)
    
    # pd.DataFrame().from_dict(train.word2idx, orient='index', columns=["word2idx"]).to_csv(DATA_DIR+'vocab.csv', encoding='utf-8')
    with open(DATA_DIR+'vocab.json', 'w') as f:
        json.dump(train.word2idx, f)