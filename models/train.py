import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import json

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import contractions
import re

stop_words = set(stopwords.words('english'))
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

from tqdm import tqdm
BATCH_SIZE=4

from utils import *
from seq2seq import EncoderRNN, AttnDecoderRNN

DATA_DIR = "../data/"
with open(DATA_DIR+"vocab.json", 'r') as f:
    WORD2IDX = json.load(f)

#defined maximum sentence and question length
max_question_len = 15
max_sentence_len = 45

# special token ids
SOS_id = WORD2IDX.get('<SOS>')
EOS_id = WORD2IDX.get('<EOS>')
unk_id = WORD2IDX['<unk>']

# preprocessing for input sentence only
def preprocess_sent(sentence):
    res = []
    for word in word_tokenize(sentence):
        word = word.lower()
        word =  re.sub('\W+','', word)
        word = contractions.fix(word)
        if word not in stop_words and word not in punctuations:
            res.append(word)
    return " ".join(res)

#tokenization function
text_pipeline = lambda x: [WORD2IDX.get(token.lower(), unk_id) for token in word_tokenize(x)]

#padding function
sentence_padding_pipeline = lambda tokens: tokens[:max_sentence_len]+[WORD2IDX['<pad>'] for p in range(max_sentence_len - len(tokens))]
question_padding_pipeline = lambda tokens: tokens[:max_question_len]+[WORD2IDX['<pad>'] for p in range(max_question_len - len(tokens))]

#prepend
sentence_prep_pipeline = lambda sent, ans: ans+sent

def collate_batch(batch):
    #initizlize empty lists for sentence and question lists
    sentence_list, question_list, sentence_length_list = [], [], []

    for (sentence, answer, question) in batch:
        #answers -> tokens -> ids
        ans_ = text_pipeline(answer)
        #sentence -> tokens -> ids -> add ans and special tokens -> pad to max sentence length 
        proc_sentence_ = text_pipeline(preprocess_sent(sentence))
        sentence_ = sentence_padding_pipeline(sentence_prep_pipeline(proc_sentence_, ans_))
        #question -> tokens -> ids -> pad to max question length
        question_ = question_padding_pipeline(text_pipeline(question))
        
        #sentence is truncated if too long
        length_ = min(len(sentence_), max_sentence_len)
        
        sentence_list.append(sentence_)
        question_list.append(question_)
        sentence_length_list.append(length_)
        
    # convert to tensor
    sentence_list = torch.tensor(sentence_list, dtype=torch.int64)
    question_list = torch.tensor(question_list, dtype=torch.int64)
    sentence_length_list = torch.tensor(sentence_length_list, dtype=torch.int64)
     
    return sentence_list.to(device), question_list.to(device), sentence_length_list.to(device)

def trainIters(encoder, decoder, training_data, num_epochs, learning_rate=0.01):
    print_every=1
    plot_every=100
    
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    criterion = nn.NLLLoss()

    for epoch in range(num_epochs):
        for _, (input_tensor, target_tensor, input_length_tensor) in tqdm(enumerate(training_data)):
            # print(input_tensor)
            loss = train(input_tensor, target_tensor, input_length_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%d %d%%) %.4f' % (epoch, epoch / num_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

def sent2idx(sentence, word2idx=WORD2IDX):
    tokens = word_tokenize(sentence)[:13]
    idx_vector = [word2idx['<SOS>']]
    idx_vector += [word2idx.get(word.lower(), word2idx['<unk>']) for word in tokens]
    idx_vector.append(word2idx['<EOS>'])
    return idx_vector
    
def sent2tensor(sentence, word2idx=WORD2IDX):
    idx_vector = sent2idx(sentence)
    return torch.tensor(idx_vector, dtype=torch.long, device=device).view(-1, 1)

if __name__=="__main__":
    n_words = len(WORD2IDX)
    hidden_size = 50
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    df = pd.read_csv(DATA_DIR+"qg_train.csv")[['sentence', 'answer', 'question']][:1000]
    train_dataloader = DataLoader(df.values, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
    
    encoder1 = EncoderRNN(n_words, hidden_size, WORD2IDX).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, n_words, WORD2IDX, dropout_p=0.1, max_length=max_sentence_len).to(device)

    print("-- Training --")
    trainIters(encoder1, attn_decoder1, train_dataloader, num_epochs=10)

    evalution_input = "Oxygen is the eighth element in the periodic table and has an atomic weight of 16."
    evaluateAndShowAttention(encoder1, attn_decoder1, evalution_input)
