import torch
import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')

import random

import json
from nltk.tokenize import word_tokenize
# Define constant values
teacher_forcing_ratio = 0.5
SOS_token = 0
EOS_token = 1
pad_token = 2
MAX_LENGTH = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

DATA_DIR = "data/"
with open(DATA_DIR+"vocab.json", 'r') as f:
    WORD2IDX = json.load(f)
    
max_sentence_len = 60
#tokenization function
text_pipeline = lambda x: [WORD2IDX.get(token.lower(), WORD2IDX['<unk>']) for token in word_tokenize(x)]

#padding function
sentence_padding_pipeline = lambda tokens: tokens[:max_sentence_len]+[WORD2IDX['<pad>'] for p in range(max_sentence_len - len(tokens))]
question_padding_pipeline = lambda tokens: tokens[:max_question_len]+[WORD2IDX['<pad>'] for p in range(max_question_len - len(tokens))]
    
def train(input_tensor, target_tensor, orig_length_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    batch_size = input_tensor.size(0)
    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)
    
    target_tensor = target_tensor.view(batch_size, target_length, 1)
    
    loss = 0
    encoder_outputs, encoder_hidden = encoder(input_tensor, orig_length_tensor, encoder_hidden)
    # print(encoder_outputs.shape, encoder_hidden[0].shape)
    
    # create mask
    masks = (input_tensor != pad_token) # [batch_size, input_len]
    
    decoder_input = torch.tensor([SOS_token], device=device)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    for bi in range(batch_size):
        encoder_hidden_ = (encoder_hidden[0].permute(1,0,2)[bi], encoder_hidden[1].permute(1,0,2)[bi])
        encoder_outputs_ = encoder_outputs[bi]
        
        decoder_hidden = (encoder_hidden_[0].view(1, 1, -1), encoder_hidden_[1].view(1, 1, -1))
        mask = masks[bi]
        # print(mask.shape)
        
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs_, mask)
                # print(decoder_output.shape, target_tensor[bi].shape)
                loss += criterion(decoder_output, target_tensor[bi][di])
                decoder_input = target_tensor[bi][di]  # Teacher forcing
                # print("Decoder input", decoder_input.shape)
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs_, mask)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                # print(decoder_output.shape, target_tensor[bi].shape)
                # print("Decoder input", decoder_input.shape)
                loss += criterion(decoder_output, target_tensor[bi][di])
                if decoder_input.item() == EOS_token:
                    break
                    

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / (batch_size*target_length)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(points)


def evaluate(encoder, decoder, sentence, word2idx=WORD2IDX, max_length=MAX_LENGTH):
    idx2word = {idx: word for (word, idx) in word2idx.items()}
    with torch.no_grad():
        tokens = text_pipeline(sentence)
        sentence_ = sentence_padding_pipeline(tokens)
        sentence_length = len(tokens)
        
        sentence_length = torch.tensor([sentence_length], dtype=torch.int64)
        # input_tensor = torch.tensor(sentence_, dtype=torch.int64).reshape((max_sentence_len, 1))
        input_tensor = torch.tensor([sentence_], dtype=torch.int64).to(device)
        input_length = input_tensor.size(1)
        encoder_hidden = encoder.initHidden()

        encoder_outputs, encoder_hidden = encoder(input_tensor, sentence_length, encoder_hidden)
        # print(encoder_outputs.shape, encoder_hidden[0].shape)
        encoder_outputs = encoder_outputs[0]
        # encoder_outputs = encoder_outputs.permute(1, 0, 2)[0]
        # encoder_hidden = (encoder_hidden[0].permute(1, 0, 2)[0], encoder_hidden[1].permute(1, 0, 2)[0])
        
        decoder_input = torch.tensor([SOS_token], device=device) 
        # print(decoder_input.shape)
        decoder_hidden = encoder_hidden
        # decoder_hidden = (encoder_hidden[0].view(1, 1, -1), encoder_hidden[1].view(1, 1, -1))

        # create mask
        mask = (input_tensor != pad_token) # [1, input_len]
        # print(mask.shape)
        
        decoded_words = []
        # decoder_attentions = torch.zeros(max_length, max_length)
        decoder_attentions = torch.zeros(max_sentence_len, max_sentence_len)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, mask)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(idx2word[topi.item()])

            decoder_input = topi.squeeze().detach()
            # print("[E] Decoder input", decoder_input, decoder_input.shape)

        return decoded_words, decoder_attentions[:di + 1]


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    plt.show()


def evaluateAndShowAttention(encoder, decoder, input_sentence):
    output_words, attentions = evaluate(encoder, decoder, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def sent2idx(sentence, word2idx=WORD2IDX):
    idx_vector = [word2idx['<SOS>']]
    idx_vector += [word2idx.get(word.lower(), 3) for word in word_tokenize(sentence)]
    idx_vector.append(word2idx['<EOS>'])
    return idx_vector
    
def sent2tensor(sentence, word2idx=WORD2IDX):
    idx_vector = sent2idx(sentence)
    return torch.tensor(idx_vector, dtype=torch.long, device=device).view(-1, 1)
