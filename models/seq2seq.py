from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
glove_path = './data/glove/glove.6B.50d.txt'

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, word2idx):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.word2idx = word2idx

        # TODO: use pretrained weights
        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.weights = torch.tensor(self.load_glove_embeddings(glove_path, word2idx))
        self.embedding = nn.Embedding.from_pretrained(self.weights)

        # TODO: change to bidirectional LSTM
        # self.gru = nn.GRU(hidden_size, hidden_size)
        self.bilstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.bilstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    def load_glove_embeddings(path, word2idx, embedding_dim=50):
        with open(path) as f:
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

        self.weights = torch.tensor(self.load_glove_embeddings(glove_path, word2idx))
        self.embedding = nn.Embedding.from_pretrained(self.weights)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        # TODO: change to LSTM
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=False)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), 
                                 dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    def load_glove_embeddings(path, word2idx, embedding_dim=50):
        with open(path) as f:
            embeddings = np.zeros((len(word2idx), embedding_dim))
            for line in f.readlines():
                values = line.split()
                word = values[0]
                index = word2idx.get(word)
                if index:
                    vector = np.array(values[1:], dtype='float32')
                    embeddings[index] = vector
            return torch.from_numpy(embeddings).float()
    
    
