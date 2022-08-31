from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import argparse
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tokenizer import get_symbols, get_words
import pickle


EOS_token = 1
SOS_token = 0

class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "SOS", 1: "EOS"}
        self.n_chars = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for charater in list(sentence):
            self.addChar(charater)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1
    


def readLangs(lang1,lang2, tokens='chars', datapath="data/terms.txt"):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open( datapath, encoding='utf-8').read().strip().split('\n')
    #lines = Datastring.strip().split('\n')
    print("total number of terms read:",len(lines))
    lines = list(set(lines))
    print("total number of unique terms:",len(lines))
    # Split every line into pairs and normalize
    print("Token Type = ", tokens)
    if tokens == 'chars' :
        pairs = [ [l,l] for l in lines]
    elif tokens == 'words' :
        pairs = []
        for l in lines :
            curr_line = get_words(l)
            pairs.append([curr_line,curr_line])
    elif tokens == 'symbols' :
        pairs = []
        for l in lines :
            curr_line = get_symbols(l)
            pairs.append([curr_line,curr_line])


    all_lens = [ len(l) for l in lines]
    max_len = max(all_lens)

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang,output_lang,pairs, max_len + 1


def prepareData(lang1, lang2, tokens = 'chars'):
    input_lang, output_lang, pairs,max_len = readLangs(lang1, lang2,tokens = tokens )
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, pairs,max_len
  

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden, cell):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, (hidden, cell) = self.lstm(output, (hidden,cell))
        return output, hidden, cell

    def initHidden(self,device):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    def initCell(self,device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


  
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden,cell):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, (hidden, cell) = self.lstm(output, (hidden,cell))
        output = self.softmax(self.out(output[0]))
        return output, hidden, cell

    def initHidden(self,device):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    def initCell(self,device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromSentence(lang, sentence, ignore_missing = False):

    if ignore_missing :
        to_ret = []
        for word in list(sentence) :
            if word in lang.char2index :
                to_ret.append(lang.char2index[word])
        return to_ret
    else :
        return [lang.char2index[word] for word in list(sentence)]
  
def tensorFromSentence(lang, sentence, device, ignore_missing = False):
    indexes = indexesFromSentence(lang, sentence, ignore_missing = ignore_missing)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0],device)
    target_tensor = tensorFromSentence(output_lang, pair[1],device)
    return (input_tensor, target_tensor)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=None):
    if max_length == None :
        max_length = MAX_LENGTH

    teacher_forcing_ratio = 0
    encoder_hidden = encoder.initHidden(device)
    encoder_cell = encoder.initCell(device)
    decoder_cell = decoder.initCell(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden,encoder_cell = encoder(input_tensor[ei], encoder_hidden,encoder_cell)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_cell = decoder(decoder_input, decoder_hidden, decoder_cell)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, tokens = 'chars'):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(0, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%d %d%%) %.4f, CurrLoss = %.4f' % (iter, iter / n_iters * 100, print_loss_avg, loss))
            torch.save(encoder1, "data/encoder_%s.model"%tokens)
            torch.save(decoder1, "data/decoder_%s.model"%tokens)
            if print_loss_avg < 0.001 :
                break

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--tokens', type=str)
    parser.add_argument('-r','--resume',action='store_true')
    args = parser.parse_args()
    tokens = args.tokens

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    

    lang1 = 'coqlan'
    lang2 = 'coqlan'


    Datastring = """eval_simple_lvalue e m (Evar x ty) bx Ptrofs.zero
    forall e l m l' m', lred ge e l m l' m' -> simple l -> simple l'.
    eq (PTree.get x e) (Some (pair bx ty))
    block
    Mem\.mem
    type
    positive
    env
    genv
    eq (PTree.get x e) (Some (pair bx ty))
    forall e l m l' m', lred ge e l m l' m' -> simple l -> simple l'.
    eq (Genv.find_symbol ge x) (Some b)
    eq (PTree.get x e) None
    block"""

    input_lang, output_lang, pairs,MAX_LENGTH = prepareData(lang1, lang2, tokens = tokens)
    
    with open("data/encoder_language_%s.pkl"%tokens,"wb") as f:
        pickle.dump(input_lang,f,protocol = pickle.HIGHEST_PROTOCOL)
    with open("data/decoder_language_%s.pkl"%tokens,"wb") as f:
        pickle.dump(output_lang,f,protocol = pickle.HIGHEST_PROTOCOL)

    print(random.choice(pairs))
    hidden_size = 2048
    print("Resume Value :", args.resume)
    if args.resume :
        
        encoder1 = torch.load("data/encoder_%s.model"%tokens).to(device)
        decoder1 = torch.load("data/decoder_%s.model"%tokens).to(device)
    else :
        encoder1 = EncoderRNN(input_lang.n_chars, hidden_size).to(device)
        decoder1 = DecoderRNN(hidden_size, output_lang.n_chars).to(device)

    trainIters(encoder1, decoder1, 1500000, print_every=5000,tokens=tokens)

    # torch.save(encoder1, "data/encoder_%s.model"%tokens)
    # torch.save(decoder1, "data/decoder_%s.model"%tokens)
    print("done")
