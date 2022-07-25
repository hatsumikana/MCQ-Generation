from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
glove_path = '../data/glove.6B.50d.txt'

MAX_LENGTH = 15

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, word2idx):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.word2idx = word2idx

        self.glove_weights = self.load_glove_embeddings(glove_path, word2idx)
        self.embedding = nn.Embedding.from_pretrained(self.glove_weights, padding_idx=word2idx["<pad>"])
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=False, batch_first=True)

    def forward(self, input, hidden):
        # print(input.shape, hidden[0].shape)
        # print(self.embedding(input).shape)
        
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded)
        
        return output, hidden

    def initHidden(self):
        # return tuple (h_n, c_n)
        init_tensor = torch.zeros(1, 1, self.hidden_size, device=device)
        return (init_tensor, init_tensor)
    
    def load_glove_embeddings(self, path, word2idx, embedding_dim=50):
        with open(path, encoding='utf-8') as f:
            embeddings = np.zeros((len(word2idx), embedding_dim))
            for line in f.readlines():
                values = line.split()
                word = values[0]
                index = word2idx.get(word)
                if index:
                    vector = np.array(values[1:], dtype='float32')
                    embeddings[index] = vector
            return torch.from_numpy(embeddings).float()


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, word2idx, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.word2idx = word2idx

        self.glove_weights = self.load_glove_embeddings(glove_path, word2idx)
        self.embedding = nn.Embedding.from_pretrained(self.glove_weights)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=False)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        # print(input, embedded)
        # print(input.shape, embedded.shape, encoder_outputs.shape, hidden[0].shape)
        
        h_n = hidden[0]
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], h_n[0]), dim=-1)), 
                                 dim=1)
        # print(attn_weights.shape)
        # print(attn_weights.unsqueeze(0).shape)
        # print(encoder_outputs.unsqueeze(0).shape)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        # print(attn_applied.shape)
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        # print(output.shape)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    def load_glove_embeddings(self, path, word2idx, embedding_dim=50):
        with open(path, encoding='utf-8') as f:
            embeddings = np.zeros((len(word2idx), embedding_dim))
            for line in f.readlines():
                values = line.split()
                word = values[0]
                index = word2idx.get(word)
                if index:
                    vector = np.array(values[1:], dtype='float32')
                    embeddings[index] = vector
            return torch.from_numpy(embeddings).float()
    
    
