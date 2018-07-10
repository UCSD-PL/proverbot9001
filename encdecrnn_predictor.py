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
import torch.cuda

from itertools import takewhile
from tactic_predictor import TacticPredictor

from typing import Dict, List, Union

use_cuda = torch.cuda.is_available()
assert use_cuda

SOS_token = 1
EOS_token = 0

SomeLongTensor = Union[torch.cuda.LongTensor, torch.LongTensor]

class EncDecRNNPredictor(TacticPredictor):
    def load_saved_state(self, filename, beam_width):
        checkpoint = torch.load(filename)
        assert checkpoint['hidden-size']
        assert checkpoint['text-encoder']
        assert checkpoint['neural-encoder']
        assert checkpoint['neural-decoder']
        # For testing only!
        checkpoint['num-encoder-layers'] = 3
        checkpoint['num-decoder-layers'] = 3
        assert checkpoint['num-encoder-layers']
        assert checkpoint['num-decoder-layers']
        assert checkpoint['max-length']

        hidden_size = checkpoint['hidden-size']
        set_encoder_state(checkpoint['text-encoder'])
        self.vocab_size = text_vocab_size()
        self.encoder = maybe_cuda(EncoderRNN(self.vocab_size, hidden_size,
                                             checkpoint["num-encoder-layers"]))
        self.decoder = maybe_cuda(DecoderRNN(hidden_size, self.vocab_size,
                                             checkpoint["num-decoder-layers"],
                                             beam_width=beam_width))
        self.encoder.load_state_dict(checkpoint['neural-encoder'])
        self.decoder.load_state_dict(checkpoint['neural-decoder'])
        self.max_length = checkpoint["max-length"]
        self.beam_width = beam_width
        pass

    def __init__(self, options):
        assert options["filename"]
        assert options["beam-width"]
        self.load_saved_state(options["filename"], options["beam-width"])

        pass

    def predictKTactics(self, in_data : Dict[str, str], k : int) -> List[str]:
        in_sentence = LongTensor(inputFromSentence(encode_context(in_data["goal"]),
                                                   self.max_length))
        feature_vector = self.encoder.run(in_sentence)
        prediction_sentences = decodeKTactics(self.decoder,
                                              feature_vector, self.beam_width)[:k]
        return [decode_tactic(sentence) for sentence in prediction_sentences]

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size=1):
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = maybe_cuda(nn.Embedding(input_size, hidden_size))
        self.gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))

        if use_cuda:
            self.cuda()
        pass

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, self.batch_size, -1)
        for i in range(self.num_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        zeroes = torch.zeros(1, self.batch_size, self.hidden_size)
        if use_cuda:
            zeroes = zeroes.cuda()
        return Variable(zeroes)

    def run(self, sentence : SomeLongTensor) -> SomeLongTensor:
        input_variable = maybe_cuda(Variable(sentence))
        assert input_variable.size()[0] == self.batch_size
        input_length = input_variable.size()[1]
        encoder_hidden = self.initHidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = self(
                input_variable[:, ei], encoder_hidden)
        return encoder_hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, batch_size=1, beam_width=1):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = maybe_cuda(nn.Embedding(output_size, hidden_size))
        self.gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self.out = maybe_cuda(nn.Linear(hidden_size, output_size))
        self.softmax = maybe_cuda(nn.LogSoftmax(1))
        self.beam_width = beam_width
        self.batch_size = batch_size

        if use_cuda:
            self.cuda()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, self.batch_size * self.beam_width, -1)
        for i in range(self.num_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initInput(self):
        return Variable(LongTensor([[SOS_token] * self.batch_size]))

    def initHidden(self):
        zeroes = torch.zeros(1, 1, self.hidden_size)
        if use_cuda:
            zeroes = zeroes.cuda()
        return Variable(zeroes)
    def run_teach(self, hidden : SomeLongTensor,
                  output_batch : SomeLongTensor) -> List[SomeLongTensor]:
        output_variable = maybe_cuda(Variable(output_batch))
        decoder_hidden = hidden
        decoder_input = self.initInput()
        prediction = []

        for di in range(output_variable.size()[1]):
            decoder_output, decoder_hidden = self(output_variable[:,di-1],
                                                  decoder_hidden)
            prediction.append(decoder_output)
        return prediction

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

def inputFromSentence(sentence, max_length):
    if len(sentence) > max_length:
        sentence = sentence[:max_length]
    if len(sentence) < max_length:
        sentence.extend([0] * (max_length - len(sentence)))
    return sentence

def commandLinePredict(predictor, k):
    sentence = ""
    next_line = sys.stdin.readline()
    while next_line != "+++++\n":
        sentence += next_line
        next_line = sys.stdin.readline()
    for result in predictor.predictKTactics({"goal": sentence}, k):
        print(result)

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

def train(dataset, hidden_size, output_size, learning_rate,
          num_encoder_layers, num_decoder_layers, max_length,
          num_epochs, batch_size, print_every):
    print("Initializing PyTorch...")
    in_stream = [inputFromSentence(datum[0], max_length) for datum in dataset]
    out_stream = [inputFromSentence(datum[1], max_length) for datum in dataset]
    data_loader = data.DataLoader(data.TensorDataset(torch.LongTensor(out_stream),
                                                     torch.LongTensor(in_stream)),
                                  batch_size=batch_size, num_workers=0,
                                  shuffle=True, pin_memory=True,
                                  drop_last=True)

    encoder = EncoderRNN(output_size, hidden_size, num_encoder_layers,
                         batch_size=batch_size)
    decoder = DecoderRNN(hidden_size, output_size, num_decoder_layers,
                         batch_size=batch_size)
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
        for batch_num, (output_batch, input_batch) in enumerate(data_loader):
            target_length = output_batch.size()[1]

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            predictor_output = decoder.run_teach(encoder.run(input_batch), output_batch)
            loss = 0
            output_var = maybe_cuda(Variable(output_batch))
            for i in range(target_length):
                loss += criterion(predictor_output[i], output_var[:,i])
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
        parser.add_argument("--num-encoder-layers", dest="num_encoder_layers",
                            default=3, type=int)
        parser.add_argument("--num-decoder-layers", dest="num_decoder_layers",
                            default=3, type=int)
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
                            args.learning_rate,
                            args.num_encoder_layers,
                            args.num_decoder_layers, args.max_length,
                            args.num_epochs, args.batch_size,
                            args.print_every)

        for epoch, (encoder_state, decoder_state) in enumerate(checkpoints):
            state = {'epoch':epoch,
                     'text-encoder':get_encoder_state(),
                     'neural-encoder':encoder_state,
                     'neural-decoder':decoder_state,
                     'num-encoder-layers':args.num_encoder_layers,
                     'num-decoder-layers':args.num_decoder_layers,
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
def decodeKTactics(decoder, encoder_hidden, k):
    v = decoder.output_size
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
