#!/usr/bin/env python3

import re
import random
import string
import sys

import time
import math
import argparse

from format import read_pair

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1

teacher_forcing_ratio = 0.5

MAX_LENGTH=20000

class TacticPredictor:
    def __init__(self):
        hidden_size = 256
        output_size = 256
        self.encoder=EncoderRNN(output_size, hidden_size)
        self.decoder=DecoderRNN(hidden_size, output_size)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        if use_cuda:
            self.embedding = self.embedding.cuda()
            self.gru = self.gru.cuda()

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

        if use_cuda:
            self.out = self.out.cuda()
            self.embedding = self.embedding.cuda()
            self.gru = self.gru.cuda()
            self.softmax = self.softmax.cuda()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        if use_cuda:
            output = output.cuda()
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

def train(input_variable, target_variable,
          encoder, decoder,
          encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break;

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "{}m {}s".format(m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return "{} (- {})".format(asMinutes(s), asMinutes(rs))

def read_data(data_path, max_size=None):
    data_set = []
    with open(data_path, mode="r") as data_file:
        pair = read_pair(data_file)
        counter = 0
        while pair and (not max_size or counter < max_size):
            context, tactic = pair
            counter += 1
            context_ids = [ord(x) for x in context]
            tactic_ids = [ord(x) for x in tactic]

            data_set.append([context_ids, tactic_ids])

            pair = read_pair(data_file)
    return data_set

def variableFromSentence(sentence):
    sentence = Variable(torch.LongTensor(sentence).view(-1, 1))
    if len(sentence) > MAX_LENGTH:
        sentence = sentence[:MAX_LENGTH]
    if use_cuda:
        return sentence.cuda()
    else:
        return sentence

def variablesFromPair(pair):
    return variableFromSentence(pair[0]), variableFromSentence(pair[1])

def commandLinePredict(encoder, decoder, max_length=MAX_LENGTH):
    sentence = ""
    next_line = sys.stdin.readline()
    while next_line != "+++++\n":
        sentence += next_line
        next_line = sys.stdin.readline()
    print (predictTactic_inner(encoder, decoder, sentence))

def predictTactic(predictor, sentence):
    return predictTactic_inner(predictor.encoder, predictor.decoder, sentence)

def predictTactic_inner(encoder, decoder, sentence, max_length=MAX_LENGTH):

    input_variable = variableFromSentence([ord(x) for x in sentence])
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    if use_cuda:
        encoder_outputs = encoder_outputs.cuda()

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    if use_cuda:
        decoder_input = decoder_input.cuda()

    decoder_hidden = encoder_hidden

    decoded_words = []
    # decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token or ni == ord('.'):
            decoded_words.append('.')
            break
        else:
            decoded_words.append(chr(ni))

        decoder_input = Variable(torch.LongTensor([[ni]]))
        if use_cuda:
            decoder_input = decoder_input.cuda()

    return ''.join(decoded_words)

def trainIters(encoder, decoder, n_iters, scrapefile,
               print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    pairs = read_data(scrapefile)
    training_pairs = [variablesFromPair(random.choice(pairs))
                      for i in range(n_iters)]

    criterion = nn.NLLLoss()

    for idx in range(1, n_iters + 1):
        training_pair = training_pairs[idx - 1]
        context_variable, tactic_variable = training_pair

        loss = train(context_variable, tactic_variable, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss

        if idx % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print("{} ({} {}%) {:.4f}".format(timeSince(start, idx / n_iters),
                                              idx, idx / n_iters * 100,
                                              print_loss_avg))


def main():
    parser = argparse.ArgumentParser(description=
                                     "pytorch model for proverbot")
    parser.add_argument("--niters", default=75000, type=int)
    parser.add_argument("--save", default=None, required=True)
    parser.add_argument("--train", default=False, const=True, action='store_const')
    parser.add_argument("--scrapefile", default="scrape.txt")
    args = parser.parse_args()
    hidden_size = 256
    output_size = 256
    encoder1 = EncoderRNN(output_size, hidden_size)
    decoder1 = DecoderRNN(hidden_size, output_size, 1)
    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()
    if args.train:
        trainIters(encoder1, decoder1, args.niters, args.scrapefile, print_every=100)
        with open(args.save + ".enc", "wb") as f:
            torch.save(encoder1.state_dict(), f)
        with open(args.save + ".dec", "wb") as f:
            torch.save(decoder1.state_dict(), f)
    else:
        encoder1.load_state_dict(torch.load(args.save + ".enc"))
        decoder1.load_state_dict(torch.load(args.save + ".dec"))
        commandLinePredict(encoder1, decoder1)

def loadPredictor(path_stem):
    predictor = TacticPredictor()
    predictor.encoder.load_state_dict(torch.load(path_stem + ".enc"))
    predictor.decoder.load_state_dict(torch.load(path_stem + ".dec"))
    return predictor

if __name__ == "__main__":
    main()
