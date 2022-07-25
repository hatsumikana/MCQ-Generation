import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import json
from nltk.tokenize import word_tokenize

from tqdm import tqdm
BATCH_SIZE=4

from utils import *
from seq2seq import EncoderRNN, AttnDecoderRNN

DATA_DIR = "../data/"
with open(DATA_DIR+"vocab.json", 'r') as f:
    WORD2IDX = json.load(f)

#defined maximum sentence and question length
max_question_len = 15
max_sentence_len = 60

#tokenization function
text_pipeline = lambda x: [WORD2IDX.get(token.lower(), WORD2IDX['<unk>']) for token in word_tokenize(x)]

#padding function
sentence_padding_pipeline = lambda tokens: tokens[:max_sentence_len]+[WORD2IDX['<pad>'] for p in range(max_sentence_len - len(tokens))]
question_padding_pipeline = lambda tokens: tokens[:max_question_len]+[WORD2IDX['<pad>'] for p in range(max_question_len - len(tokens))]

def collate_batch(batch):
    #initizlize empty lists for sentence and question lists
    sentence_list, question_list, sentence_length_list = [], [], []

    for (sentence, question) in batch:
        tokens = text_pipeline(sentence)
        #sentence -> tokens -> id -> pad to max sentence length
        sentence_ = sentence_padding_pipeline(tokens)
        #question -> tokens -> ids -> pad to max question length
        question_ = question_padding_pipeline(tokens)
        #sentence is truncated if too long
        length_ = min(len(tokens), max_sentence_len)
        
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
        # for i in range(len(training_data)):
        #     training_pair = training_data[i]
        #     input_tensor = sent2tensor(training_pair[0])
        #     target_tensor = sent2tensor(training_pair[1])
         
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
    
    df = pd.read_csv(DATA_DIR+"qg_train.csv")[['sentence', 'question']][:100]
    train_dataloader = DataLoader(df.values, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
    
    encoder1 = EncoderRNN(n_words, hidden_size, WORD2IDX).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, n_words, WORD2IDX, dropout_p=0.1, max_length=max_sentence_len).to(device)

    print("-- Training --")
    trainIters(encoder1, attn_decoder1, train_dataloader, num_epochs=5)

    evalution_input = "Born and raised in Houston Texas she performed in various singing and dancing competitions as a child and rose to fame in the late 1990s as lead singer of R&B girl group Destiny's Child."
    evaluateAndShowAttention(encoder1, attn_decoder1, evalution_input)
