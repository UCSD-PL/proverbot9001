#!/usr/bin/env python3

import re
import random
import string
import sys
import signal

import time
import math
import argparse

from format import read_pair
from text_encoder import encode_tactic, encode_context, \
    decode_tactic, decode_context, \
    text_vocab_size, \
    get_encoder_state, set_encoder_state
import text_encoder

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.utils.data as data

from itertools import takewhile
from tactic_predictor import TacticPredictor

use_cuda = torch.cuda.is_available()
# assert use_cuda

SOS_token = 1
EOS_token = 0

teacher_forcing_ratio = 0.5

MAX_LENGTH=200

class EncDecRNNPredictor(TacticPredictor):
    def load_saved_state(filename):
        checkpoint = torch.load(options["filename"] + ".tar")
        assert checkpoint['hidden_size']
        assert checkpoint['text_encoder_dict']
        assert checkpoint['encoder']
        assert checkpoint['decoder']
        assert checkpoint['max_length']

        hidden_size = checkpoint['hidden_size']
        set_encoder_state(checkpoint['text_encoder_dict'])
        output_size = text_vocab_size()
        self.encoder = EncoderRNN(output_size, hidden_size, 1)
        self.decoder = DecoderRNN(hidden_size, output_size, 1)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.max_length = checkpoint["max_length"]
        self.beam_width = options["beam_width"]
        pass

    def __init__(self, options):
        assert options["filename"]
        assert options["beam_width"]
        load_saved_state(options["filename"])

        if use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.vocab_size = output_size
        pass

    def predictKTactics(self, in_data, k):
        return [decode_tactic(tokenlist) for tokenlist in
                predictKTokenlist(self, encode_context(in_data["goal"]),
                                  self.beam_width, self.max_length)[:k]]

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, n_layers=3):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

        if use_cuda:
            self.cuda()
            self.embedding = self.embedding.cuda()
            self.gru = self.gru.cuda()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, self.batch_size, -1)
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        zeroes = torch.zeros(1, self.batch_size, self.hidden_size)
        if use_cuda:
            zeroes = zeroes.cuda()
        return Variable(zeroes)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, n_layers, width=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(1)
        self.k = width
        self.batch_size = batch_size

        if use_cuda:
            self.cuda()
            self.embedding = self.embedding.cuda()
            self.gru = self.gru.cuda()
            self.out = self.out.cuda()
            self.softmax = self.softmax.cuda()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, self.batch_size * self.k, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        zeroes = torch.zeros(1, 1, self.hidden_size)
        if use_cuda:
            zeroes = zeroes.cuda()
        return Variable(zeroes)

def LongTensor(arr):
    if use_cuda:
        return torch.cuda.LongTensor(arr)
    else:
        return torch.LongTensor(arr)

def FloatTensor(k, val):
    if use_cuda:
        return torch.cuda.FloatTensor(k, val)
    else:
        return torch.FloatTensor(k, val)

def run_predictor_teach(input_variable, output_variable, encoder, decoder):
    batch_size = input_variable.size()[0]
    input_length = input_variable.size()[1]
    output_length = output_variable.size()[1]
    decoder_input = Variable(LongTensor([[SOS_token] * batch_size]))
    encoder_hidden = encoder.initHidden()
    decoder_hidden = encoder_hidden
    prediction = []

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[:,ei], encoder_hidden)

    for di in range(output_length):
        decoder_output, decoder_hidden = decoder(output_variable[:,di-1],
                                                 decoder_hidden)
        prediction.append(decoder_output)
    return prediction

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "{}m {:.2f}s".format(m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return "{} (- {})".format(asMinutes(s), asMinutes(rs))

def read_text_data(data_path, max_size=None):
    data_set = []
    with open(data_path, mode="r") as data_file:
        pair = read_pair(data_file)
        counter = 0
        while pair and (not max_size or counter < max_size):
            context, tactic = pair
            counter += 1
            data_set.append([encode_context(context),
                             encode_tactic(tactic)])

            pair = read_pair(data_file)
    assert(len(data_set) > 0)
    return data_set

def read_num_data(data_path, max_size=None):
    data_set = []
    with open(data_path, mode="r") as data_file:
        context = data_file.readline()
        counter = 0
        while(context != "" and (not max_size or counter < max_size)):
            tactic = data_file.readline()
            blank_line = data_file.readline()
            assert tactic != "" and (blank_line == "\n" or blank_line == ""), "tactic line: {}\nblank line: {}".format(tactic, blank_line)
            context_ids = [int(num) for num in context.split(" ")]
            tactic_ids = [int(num) for num in tactic.split(" ")]

            data_set.append([context_ids, tactic_ids])
            context = data_file.readline()
    return data_set

def inputFromSentence(sentence):
    if len(sentence) > MAX_LENGTH:
        sentence = sentence[:MAX_LENGTH]
    if len(sentence) < MAX_LENGTH:
        sentence.extend([0] * (MAX_LENGTH - len(sentence)))
    return sentence

def variableFromSentence(sentence):
    sentence = inputFromSentence(sentence)
    sentence = Variable(LongTensor(sentence).view(1, -1))
    return sentence

def commandLinePredict(predictor, numfile, k, max_length):
    predictor.decoder.k = k
    if numfile:
        sentence = sys.stdin.readline()
        tokenlist = [int(w) for w in sentence.split()]
    else:
        sentence = ""
        next_line = sys.stdin.readline()
        while next_line != "+++++\n":
            sentence += next_line
            next_line = sys.stdin.readline()
        tokenlist = [ord(x) for x in sentence]

    tokensresults = predictKTokenlist(predictor, tokenlist, k, max_length)

    if numfile:
        for result in tokensresults:
            print(list(result))
    else:
        for result in tokensresults:
            print(''.join([chr(x) for x in result]))

def predictKTokenlist(predictor, tokenlist, k, max_length):
    if len(tokenlist) < max_length:
        tokenlist.extend([0] * (max_length - len(tokenlist)))
    encoder_hidden = encodeContext(predictor.encoder, tokenlist)
    return decodeKTactics(predictor.decoder, encoder_hidden, k, predictor.vocab_size)

def encodeContext(encoder, tokenlist):
    input_variable = variableFromSentence(tokenlist)
    input_length = input_variable.size()[1]
    encoder_hidden = encoder.initHidden()

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[:, ei], encoder_hidden)

    return encoder_hidden

def decodeTactic(decoder, encoder_hidden, vocab_size):
    decoder_hidden = encoder_hidden
    decoded_tokens = []

    decoder_input = Variable(LongTensor([[SOS_token]]))

    for _ in range(MAX_LENGTH):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        decoded_tokens.append(ni)

        decoder_input = Variable(LongTensor([[ni]]))

    return decoded_tokens

def adjustLearningRates(initial, optimizers, epoch):
    for optimizer in optimizers:
        lr = initial * (0.5 ** (epoch // 20))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def maybe_cuda(component):
    if use_cuda:
        return component.cuda()
    else:
        return component

def train(dataset,
          hidden_size, output_size,
          learning_rate, num_layers, max_length,
          num_epochs, batch_size, print_every):
    print("Initializing PyTorch...")
    in_stream = [inputFromSentence(datum[0]) for datum in dataset]
    out_stream = [inputFromSentence(datum[1]) for datum in dataset]
    data_loader = data.DataLoader(data.TensorDataset(torch.LongTensor(out_stream),
                                                     torch.LongTensor(in_stream)),
                                  batch_size=batch_size, num_workers=0,
                                  shuffle=True, pin_memory=True,
                                  drop_last=True)

    encoder = EncoderRNN(output_size, hidden_size, batch_size, num_layers)
    decoder = DecoderRNN(hidden_size, output_size, batch_size, num_layers)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    optimizers = [encoder_optimizer, decoder_optimizer]
    criterion = maybe_cuda(nn.NLLLoss())

    start = time.time()
    num_items = len(dataset) * num_epochs
    total_loss = 0

    print("Training...")
    for epoch in range(num_epochs):
        print("Epoch {}".format(epoch))
        adjustLearningRates(learning_rate, optimizers, epoch)
        for batch_num, (out_batch, in_batch) in enumerate(data_loader):
            target_length = out_batch.size()[1]
            in_var = maybe_cuda(Variable(in_batch))
            out_var = maybe_cuda(Variable(out_batch))

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            predictor_output = run_predictor_teach(in_var, out_var,
                                                   encoder, decoder)
            loss = 0
            for i in range(target_length):
                loss += criterion(predictor_output[i], out_var[:,i])
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += (loss.data.item() / target_length) * batch_size

            if batch_num % print_every == 0:
                items_processed = (batch_num + 1) * batch_size + epoch * len(dataset)
                progress = items_processed / num_items
                print("{} ({} {:.2f}%) {:.4f}".
                      format(timeSince(start, progress),
                             items_processed, progress * 100,
                             total_loss / items_processed))

        yield encoder.state_dict(), decoder.state_dict()

def exit_early(signal, frame):
    sys.exit(0)

def take_args():
    parser = argparse.ArgumentParser(description=
                                     "pytorch model for proverbot")
    parser.add_argument("command")
    args = parser.parse_args(sys.argv[1:2])
    if args.command == "train":
        parser = argparse.ArgumentParser(description=
                                         "pytorch model for proverbot")
        parser.add_argument("scrape_file")
        parser.add_argument("save_file")
        parser.add_argument("--num-epochs", dest="num_epochs", default=50, type=int)
        parser.add_argument("--batch-size", dest="batch_size", default=256, type=int)
        parser.add_argument("--max-length", dest="max_length", default=100, type=int)
        parser.add_argument("--print-every", dest="print_every", default=10, type=int)
        parser.add_argument("--hidden-size", dest="hidden_size", default=None, type=int)
        parser.add_argument("--learning-rate", dest="learning_rate",
                            default=0.003, type=float)
        parser.add_argument("--num-layers", dest="num_layers", default=3, type=int)
        return args.command, parser.parse_args(sys.argv[2:])
    elif args.command == "predict":
        parser.add_argument("save_file")
        parser.add_argument("--num-predictions", default=3, type=int)
        parser.add_argument("--beam-width", dest="beam_width", default=None, type=int)
        return args.command, parser.parse_args(sys.argv[2:])
    else:
        print("Unrecognized subcommand!")
        parser.print_help()
        sys.exit(1)

def main():
    # Set up cleanup handler for Ctrl-C
    signal.signal(signal.SIGINT, exit_early)
    subcommand, args = take_args()
    if subcommand == "train":
        print("Reading dataset...")
        dataset = read_text_data(args.scrape_file)
        output_size = text_vocab_size()
        if args.hidden_size:
            hidden_size = args.hidden_size
        else:
            hidden_size = output_size * 2

        checkpoints = train(dataset, hidden_size, output_size,
                            args.learning_rate, args.num_layers, args.max_length,
                            args.num_epochs, args.batch_size, args.print_every)

        for epoch, (encoder_state, decoder_state) in enumerate(checkpoints):
            state = {'epoch':epoch,
                     'text-encoder':get_encoder_state(),
                     'neural-encoder':encoder_state,
                     'neural-decoder':decoder_state,
                     'hidden-size':hidden_size,
                     'max-length': args.max_length}
            with open(args.save_file, 'wb') as f:
                print("=> Saving checkpoint at epoch {}".
                      format(epoch))
                torch.save(state, f)
    else:
        if args.beam_width:
            beam_width = args.beam_width
        else:
            beam_width = args.num_predictions * args.num_predictions
        predictor = EncDecRNNPredictor({'filename': args.save_file,
                                        'beam_width': beam_width})
        commandLinePredict(predictor, args.num_predictions)

# The code below here was copied from
# https://ibm.github.io/pytorch-seq2seq/public/_modules/seq2seq/models/TopKDecoder.html
# and modified. This code is available under the apache license.
def decodeKTactics(decoder, encoder_hidden, k, v):
    pos_index = Variable(LongTensor([0]) * k).view(-1, 1)

    hidden = _inflate(encoder_hidden, k)

    sequence_scores = FloatTensor(k, 1)
    sequence_scores.fill_(-float('Inf'))
    sequence_scores.index_fill_(0, LongTensor([0]), 0.0)
    sequence_scores = Variable(sequence_scores)

    input_var = Variable(LongTensor([[SOS_token] * k]))

    stored_predecessors = list()
    stored_emitted_symbols = list()

    decoder.k = k

    for j in range(MAX_LENGTH):
        decoder_output, hidden = decoder(input_var, hidden)

        sequence_scores = _inflate(sequence_scores, v)
        sequence_scores += decoder_output

        scores, candidates = sequence_scores.view(1, -1).topk(k)

        input_var = (candidates % v).view(1, k)
        sequence_scores = scores.view(k, 1)

        predecessors = (candidates / v +
                        pos_index.expand_as(candidates)).view(k, 1)
        hidden = hidden.index_select(1, predecessors.squeeze())

        eos_indices = input_var.data.eq(EOS_token)
        if eos_indices.nonzero().dim() > 0:
            sequence_scores.data.masked_fill_(torch.transpose(eos_indices, 0, 1),
                                              -float('inf'))

        stored_predecessors.append(predecessors)
        stored_emitted_symbols.append(torch.transpose(input_var, 0, 1))


    # Trace back from the final three highest scores
    _, next_idxs = sequence_scores.view(k).sort(descending=True)
    seqs = []
    eos_found = 0
    for i in range(MAX_LENGTH - 1, -1, -1):
        # The next column of symbols from the end
        next_symbols = stored_emitted_symbols[i].view(k).index_select(0, next_idxs).data
        # The predecessors of that column
        next_idxs = stored_predecessors[i].view(k).index_select(0, next_idxs)

        # Handle sequences that ended early
        eos_indices = stored_emitted_symbols[i].data.squeeze(1).eq(EOS_token).nonzero()
        if eos_indices.dim() > 0:
            for j in range(eos_indices.size(0)-1, -1, -1):
                idx = eos_indices[j]

                res_k_idx = k - (eos_found % k) - 1
                eos_found += 1
                res_idx = res_k_idx

                next_idxs[res_idx] = stored_predecessors[i][idx[0]]
                next_symbols[res_idx] = stored_emitted_symbols[i][idx[0]].data[0]

        # Commit the result
        seqs.insert(0, next_symbols)

    # Transpose
    seqs = [[data[i].item() for data in seqs] for i in range(k)]
    # Cut off EOS tokens
    seqs = [list(takewhile(lambda x: x != EOS_token, seq)) for seq in seqs]

    return seqs

def _inflate(tensor, times):
    tensor_dim = len(tensor.size())
    if tensor_dim == 3:
        b = tensor.size(1)
        return tensor.repeat(1, 1, times).view(tensor.size(0), b * times, -1)
    elif tensor_dim == 2:
        return tensor.repeat(1, times)
    elif tensor_dim == 1:
        b = tensor.size(0)
        return tensor.repeat(times).view(b, -1)
    else:
        raise ValueError("Tensor can be of 1D, 2D, or 3D only. "
                         "This one is {}D.".format(tensor_dim))

def _mask_symbol_scores(self, score, idx, masking_score=-float('inf')):
    score[idx] = masking_score

def _mask(tensor, idx, dim=0, masking_score=-float('inf')):
    if len(idx.size()) > 0:
        indices = idx[:,0]
        tensor.index_fill_(dim, indices, masking_score)

## ENDING HERE

if __name__ == "__main__":
    main()
