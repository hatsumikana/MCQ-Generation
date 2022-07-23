import pandas as pd 
import torch
import random

#find gpu otherwise use cpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

'''
load and split the data
'''
data = pd.read_csv("qg_train.csv", sep="\t")
data.head()

question = data['question'].values.tolist()
sentence = data['sentence'].values.tolist()

data = [(question[i], sentence[i]) for i in range(len(question))]


#Train-Test split
import numpy as np
from sklearn.model_selection import train_test_split

train_dataset, test_dataset = train_test_split(
    data, test_size=1/10, random_state=179)
train_dataset, valid_dataset = train_test_split(
    train_dataset, test_size=1/9, random_state=179)

'''
Tokenization
'''
import torchtext
from torchtext.data.utils import get_tokenizer
from collections import Counter

# tokenizer type
tokenizer = get_tokenizer("basic_english")

# vocab
counter = Counter()
for (sentence, question) in train_dataset:
  counter.update(tokenizer(question))

#define vocab
vocab = torchtext.vocab.Vocab(counter, max_size=10000,  specials=('<SOS>', '<EOS>', '<pad>', '<unk>'), specials_first=True)
print("\nVocab size:",len(vocab))

'''
Dataloader
'''
from torch.utils.data import Dataset, DataLoader

#defined maximum sentence and question length
max_question_len = 15
max_sentence_len = 50

#tokenization function
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]

#padding function
sentence_padding_pipeline = lambda tokens: [vocab.stoi['<pad>'] for p in range(max_sentence_len - len(tokens))] + tokens[-max_sentence_len:]
question_padding_pipeline = lambda tokens: [vocab.stoi['<pad>'] for p in range(max_question_len - len(tokens))] + tokens[:max_question_len]

#collate function for dataloader
def collate_batch(batch):
    #initizlize empty lists for sentence and question lists
    sentence_list, question_list = [], []

    for (sentence, question) in batch:
        #sentence -> tokens -> id -> pad to max sentence length
        sentence_ = sentence_padding_pipeline(text_pipeline(sentence))
        #question -> tokens -> ids -> pad to max question length
        question_ = question_padding_pipeline(text_pipeline(question))

        sentence_list.append(sentence_)
        question_list.append(question_)
        
    # convert to tensor
    sentence_list = torch.tensor(sentence_list, dtype=torch.int64)
    question_list = torch.tensor(question_list, dtype=torch.int64)
    return sentence_list.to(device), question_list.to(device)


BATCH_SIZE=8

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)

'''
Download glove
'''
from gensim.models import KeyedVectors
import gensim.downloader as api

try:
    print("Loading saved word vectors...")
    glove_50dim = KeyedVectors.load("./glove_50dim.w2v")
except:
    print("Downloading word vectors...")
    glove_50dim = api.load("glove-wiki-gigaword-50")
    glove_50dim.save('glove_50dim.w2v')

print("Number of word vectors:", glove_50dim.vectors.shape)

#Initialise model embedding with glove
for word in vocab.stoi.keys():
    if word in glove_50dim.key_to_index.keys():
        word_vec = glove_50dim[word]
        model.embedding.weight.data[vocab.stoi[word]] = torch.tensor(word_vec)


'''
Training and validation
'''