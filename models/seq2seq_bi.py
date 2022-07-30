from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
# device = 'cpu'
glove_path = 'data/glove.6B.50d.txt'

MAX_LENGTH = 15

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, word2idx):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.word2idx = word2idx

        self.glove_weights = self.load_glove_embeddings(glove_path, word2idx)
        self.embedding = nn.Embedding.from_pretrained(self.glove_weights, padding_idx=word2idx["<pad>"])
        self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.W_h = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)
        self.W_c = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)
        self.W_o = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)

    def forward(self, input, input_length, hidden):
        # print(input.shape, hidden[0].shape)
        # print(self.embedding(input).shape)
        
        # basic implementation
        # embedded = self.embedding(input)
        # output, hidden = self.lstm(embedded)
        
        # with pack_padded_sequence
        batch_size, max_seq_len = input.size(0), input.size(-1)
        embedded = self.embedding(input)
        # print(input.shape, embedded.shape, input_length.shape)
        #need to explicitly put lengths on cpu!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_length.cpu().numpy(), batch_first=True, enforce_sorted=False)     
        packed_output, (last_hidden, last_cell) = self.lstm(packed_embedded)
        last_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1) 
        last_cell = torch.cat((last_cell[0], last_cell[1]), dim=1)  
        hidden = (self.W_h(last_hidden)[None, :], self.W_c(last_cell)[None, :])   
        # final_state = hidden[0].view(1, 2, batch_size, self.hidden_size)[-1] 
        # print(hidden.shape)
        # print(final_state[0].shape)
        # h_1, h_2 = final_state[0], final_state[1]      
        # hidden = torch.cat((h_1, h_2), 1) 
        #packed_output is a packed sequence containing all hidden states
        #hidden is now from the final non-padded element in the batch
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) 
        output = self.W_o(output)
        #output is now a non-packed sequence, all hidden states obtained when the input is a pad token are all zeros
        # ISSUE --> max_seq_len in output is not original padded length
        if output.size(1)<max_seq_len:
            dummy_tensor = torch.zeros(batch_size, max_seq_len-output.size(1), self.hidden_size).to(device)
            # print(output.shape, dummy_tensor.shape)
            output = torch.cat([output, dummy_tensor], 1)
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

    def forward(self, input, hidden, encoder_outputs, mask):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded) # [1, 1, embed_dim]
        # print(input, embedded)
        # print(input.shape, embedded.shape, encoder_outputs.shape, hidden[0].shape)
        
        # encoder_outputs --> [enc_input_len, embed_dim]
        # encoder_hidden --> tuple with [1, 1, embed_dim]
        
        h_n = hidden[0]
        # print(embedded[0].shape, h_n[0].shape)
        attn = self.attn(torch.cat((embedded[0], h_n[0]), dim=-1))  # [1, enc_input_len]
        attn = attn.masked_fill(mask == 0, -1e10)
        attn_weights = F.softmax(attn, dim=1)
        # print(attn.shape, attn_weights.shape)
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
    
    
