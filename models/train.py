import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import json
from nltk.tokenize import word_tokenize

BATCH_SIZE=3

from utils import *
from seq2seq import EncoderRNN, AttnDecoderRNN

DATA_DIR = "../data/"
with open(DATA_DIR+"vocab.json", 'r') as f:
    WORD2IDX = json.load(f)
    
def trainIters(encoder, decoder, training_data, num_epochs, learning_rate=0.01):
    print_every=1000
    plot_every=100
    
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    criterion = nn.NLLLoss()

    for epoch in range(num_epochs):
        for i in range(len(training_data)):
            training_pair = training_data[i]
            input_tensor = sent2tensor(training_pair[0])
            target_tensor = sent2tensor(training_pair[1])
        # for _, (input_tensor, target_tensor) in enumerate(training_data):
            loss = train(input_tensor, target_tensor, encoder,
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
    idx_vector = [word2idx['SOS']]
    idx_vector += [word2idx.get(word.lower(),86267) for word in word_tokenize(sentence)]
    idx_vector.append(word2idx['EOS'])
    return idx_vector
    
def sent2tensor(sentence, word2idx=WORD2IDX):
    idx_vector = sent2idx(sentence)
    return torch.tensor(idx_vector, dtype=torch.long, device=device).view(-1, 1)

def collate_batch(batch):
    data_list, label_list = [], []

    #for each element in the batch 
    seq_len = 5
    for ele in batch:
      data_list.append(sent2idx(ele[0][:seq_len]))
      label_list.append(sent2idx(ele[1][:seq_len]))
         
    
    data_list = torch.Tensor(data_list)
    label_list = torch.Tensor(label_list).view(-1,1)
    return data_list, label_list

if __name__=="__main__":
    n_words = len(WORD2IDX)
    hidden_size = 256
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    df = pd.read_csv(DATA_DIR+"qg_train.csv")[['sentence', 'question']][:10]
    # train_dataloader = DataLoader(df.values, batch_size=BATCH_SIZE,
    #                           shuffle=True, collate_fn=collate_batch)
    
    train_dataloader = df.values
    
    encoder1 = EncoderRNN(n_words, hidden_size, WORD2IDX).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, n_words, WORD2IDX, dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, train_dataloader, 10)
    evaluateAndShowAttention(encoder1, attn_decoder1, "The pound-force has a metric counterpart, less commonly used than the newton: the kilogram-force (kgf) (sometimes kilopond), is the force exerted by standard gravity on one kilogram of mass") 
