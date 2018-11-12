#!/usr/bin/env python3

import re
import random
import string
import sys
import signal

import time
import math
import argparse
import itertools

from tokenizer import KeywordTokenizer, context_keywords, tactic_keywords
from data import read_text_data, encode_seq_seq_data, Sentence, \
    SequenceSequenceDataset, EOS_token, SOS_token
from util import *
from util import _inflate

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.cuda

from itertools import takewhile
from models.tactic_predictor import TacticPredictor

from typing import Dict, List, Union, Any, Tuple, Iterable, cast, overload

SomeLongTensor = Union[torch.cuda.LongTensor, torch.LongTensor]
SomeFloatTensor = Union[torch.cuda.FloatTensor, torch.FloatTensor]

class EncDecRNNPredictor(TacticPredictor):
    def load_saved_state(self, filename : str, beam_width : int) -> None:
        checkpoint = torch.load(filename)
        assert checkpoint['hidden-size']
        assert checkpoint['context-tokenizer']
        assert checkpoint['tactic-tokenizer']
        assert checkpoint['neural-encoder']
        assert checkpoint['neural-decoder']
        assert checkpoint['num-encoder-layers']
        assert checkpoint['num-decoder-layers']
        assert checkpoint['max-length']

        hidden_size = checkpoint['hidden-size']
        self.context_tokenizer = checkpoint['context-tokenizer']
        self.tactic_tokenizer = checkpoint['tactic-tokenizer']
        self.encoder = maybe_cuda(EncoderRNN(self.context_tokenizer.numTokens(),
                                             hidden_size,
                                             checkpoint["num-encoder-layers"]))
        self.decoder = maybe_cuda(DecoderRNN(hidden_size,
                                             self.tactic_tokenizer.numTokens(),
                                             checkpoint["num-decoder-layers"],
                                             beam_width=beam_width))
        self.encoder.load_state_dict(checkpoint['neural-encoder'])
        self.decoder.load_state_dict(checkpoint['neural-decoder'])
        self.max_length = checkpoint["max-length"]
        self.beam_width = beam_width
        pass

    def __init__(self, options : Dict["str", Any]) -> None:
        assert options["filename"]
        assert options["beam-width"]
        self.load_saved_state(options["filename"], options["beam-width"])

        pass

    def predictKTactics(self, in_data : Dict[str, Union[List[str], str]], k : int) -> \
        List[Tuple[str, float]]:
        in_sentence = LongTensor(inputFromSentence(
            self.context_tokenizer.toTokenList(in_data["goal"]),
            self.max_length)).view(1, -1)
        feature_vector = self.encoder.run(in_sentence)
        prediction_sentences = decodeKTactics(self.decoder,
                                              feature_vector,
                                              self.beam_width,
                                              self.max_length)[:k]
        return [(self.tactic_tokenizer.toString(sentence), .5 **i)
                for sentence, i in zip(prediction_sentences, itertools.count())]

class EncoderRNN(nn.Module):
    def __init__(self, input_size : int, hidden_size : int,
                 num_layers : int, batch_size : int=1) -> None:
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = maybe_cuda(nn.Embedding(input_size, hidden_size))
        self.gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))

        pass

    def forward(self, input : torch.LongTensor, hidden : torch.LongTensor) \
        -> Tuple[SomeLongTensor, SomeLongTensor] :
        output = self.embedding(input).view(1, self.batch_size, -1)
        for i in range(self.num_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self) -> torch.FloatTensor:
        zeroes = cast(torch.FloatTensor, maybe_cuda(
            torch.zeros(1, self.batch_size, self.hidden_size)))
        return Variable(zeroes)

    def run(self, sentence : torch.LongTensor) -> torch.FloatTensor:
        input_variable = maybe_cuda(Variable(sentence))
        assert input_variable.size()[0] == self.batch_size, \
            "input var has size {}, batch_size is {}".format(input_variable.size()[0],
                                                             self.batch_size)
        input_length = input_variable.size()[1]
        encoder_hidden = self.initHidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = self(
                input_variable[:, ei], encoder_hidden)
        return encoder_hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size : int, output_size : int, num_layers : int,
                 batch_size : int =1, beam_width : int =1) -> None:
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


    def forward(self, input : SomeLongTensor, hidden : SomeLongTensor) \
        -> Tuple[SomeLongTensor, SomeLongTensor]:
        output = self.embedding(input).view(1, self.batch_size * self.beam_width, -1)
        for i in range(self.num_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initInput(self) -> SomeLongTensor:
        return Variable(LongTensor([[SOS_token] * self.batch_size]))

    def initHidden(self) -> SomeLongTensor:
        zeroes = cast(torch.LongTensor, maybe_cuda(torch.zeros(1, 1, self.hidden_size)))
        return Variable(zeroes)
    def run_teach(self, hidden : torch.FloatTensor,
                  output_batch : torch.LongTensor) -> List[torch.FloatTensor]:
        output_variable = maybe_cuda(Variable(output_batch))
        decoder_hidden = hidden
        decoder_input = self.initInput()
        prediction = []

        for di in range(output_variable.size()[1]):
            decoder_output, decoder_hidden = self(decoder_input, decoder_hidden)
            decoder_input = output_variable[:,di]
            prediction.append(decoder_output)
        return prediction

def inputFromSentence(sentence : Sentence, max_length : int) -> Sentence:
    if len(sentence) > max_length:
        sentence = sentence[:max_length]
    if len(sentence) < max_length:
        sentence.extend([0] * (max_length - len(sentence)))
    return sentence

def commandLinePredict(predictor : EncDecRNNPredictor, k : int) -> None:
    sentence = ""
    next_line = sys.stdin.readline()
    while next_line != "+++++\n":
        sentence += next_line
        next_line = sys.stdin.readline()
    for result in predictor.predictKTactics({"goal": sentence}, k):
        print(result)

def adjustLearningRates(initial : float,
                        optimizers : List[torch.optim.SGD],
                        epoch : int):
    for optimizer in optimizers:
        lr = initial * (0.5 ** (epoch // 20))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

Checkpoint = Tuple[Dict[Any, Any], Dict[Any, Any]]

def train(dataset : SequenceSequenceDataset, hidden_size : int,
          learning_rate : float, num_encoder_layers : int,
          num_decoder_layers : int, max_length : int, num_epochs : int, batch_size : int,
          print_every : int, context_vocab_size : int, tactic_vocab_size : int) -> Iterable[Checkpoint]:
    print("Initializing PyTorch...")
    in_stream = [inputFromSentence(datum[0], max_length) for datum in dataset]
    out_stream = [inputFromSentence(datum[1], max_length) for datum in dataset]
    data_loader = data.DataLoader(data.TensorDataset(torch.LongTensor(out_stream),
                                                     torch.LongTensor(in_stream)),
                                  batch_size=batch_size, num_workers=0,
                                  shuffle=True, pin_memory=True,
                                  drop_last=True)

    encoder = EncoderRNN(context_vocab_size, hidden_size, num_encoder_layers,
                         batch_size=batch_size)
    decoder = DecoderRNN(hidden_size, tactic_vocab_size, num_decoder_layers,
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
            predictor_output = decoder.run_teach(encoder
                                                 .run(cast(SomeLongTensor, input_batch)),
                                                 cast(SomeLongTensor, output_batch))
            loss = maybe_cuda(Variable(LongTensor(0)))
            output_var = maybe_cuda(Variable(output_batch))
            for i in range(target_length):
                loss += criterion(predictor_output[i], output_var[:,i])
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += (loss.data[0] / target_length) * batch_size

            if (batch_num + 1) % print_every == 0:
                items_processed = (batch_num + 1) * batch_size + epoch * len(dataset)
                progress = items_processed / num_items
                print("{} ({} {:.2f}%) {:.4f}".
                      format(timeSince(start, progress),
                             items_processed, progress * 100,
                             total_loss / items_processed))

        yield encoder.state_dict(), decoder.state_dict()

def exit_early(signal, frame):
    sys.exit(0)

def take_args(args : List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=
                                     "pytorch model for proverbot")
    parser.add_argument("scrape_file")
    parser.add_argument("save_file")
    parser.add_argument("--num-epochs", dest="num_epochs", default=50, type=int)
    parser.add_argument("--batch-size", dest="batch_size", default=256, type=int)
    parser.add_argument("--max-length", dest="max_length", default=100, type=int)
    parser.add_argument("--print-every", dest="print_every", default=10, type=int)
    parser.add_argument("--hidden-size", dest="hidden_size", default=256, type=int)
    parser.add_argument("--learning-rate", dest="learning_rate",
                        default=0.003, type=float)
    parser.add_argument("--num-encoder-layers", dest="num_encoder_layers",
                        default=3, type=int)
    parser.add_argument("--num-decoder-layers", dest="num_decoder_layers",
                        default=3, type=int)
    return parser.parse_args(args)

def main(args_list : List[str]) -> None:
    # Set up cleanup handler for Ctrl-C
    signal.signal(signal.SIGINT, exit_early)
    args = take_args(args_list)
    print("Reading dataset...")
    raw_dataset = read_text_data(args.scrape_file)
    dataset, context_tokenizer, tactic_tokenizer = \
                encode_seq_seq_data(raw_dataset,
                                    lambda keywords, num_reserved:
                                    KeywordTokenizer(context_keywords, num_reserved),
                                    lambda keywords, num_reserved:
                                    KeywordTokenizer(tactic_keywords, num_reserved),
                                    0, 2)


    checkpoints = train(dataset, args.hidden_size,
                        args.learning_rate,
                        args.num_encoder_layers,
                        args.num_decoder_layers, args.max_length,
                        args.num_epochs, args.batch_size,
                        args.print_every,
                        context_tokenizer.numTokens(),
                        tactic_tokenizer.numTokens())

    for epoch, (encoder_state, decoder_state) in enumerate(checkpoints):
        state = {'epoch':epoch,
                 'context-tokenizer':context_tokenizer,
                 'tactic-tokenizer':tactic_tokenizer,
                 'neural-encoder':encoder_state,
                 'neural-decoder':decoder_state,
                 'num-encoder-layers':args.num_encoder_layers,
                 'num-decoder-layers':args.num_decoder_layers,
                 'hidden-size':args.hidden_size,
                 'max-length': args.max_length}
        with open(args.save_file, 'wb') as f:
            print("=> Saving checkpoint at epoch {}".
                  format(epoch))
            torch.save(state, f)

# The code below here was copied from
# https://ibm.github.io/pytorch-seq2seq/public/_modules/seq2seq/models/TopKDecoder.html
# and modified. This code is available under the apache license.
def decodeKTactics(decoder : DecoderRNN, encoder_hidden : torch.FloatTensor,
                   beam_width : int, max_length : int):
    v = decoder.output_size
    pos_index = Variable(LongTensor([0]) * beam_width).view(-1, 1)

    hidden = _inflate(encoder_hidden, beam_width)

    sequence_scores = FloatTensor(beam_width, 1)
    sequence_scores.fill_(-float('Inf'))
    sequence_scores.index_fill_(0, LongTensor([0]), 0.0)
    sequence_scores = Variable(sequence_scores)

    input_var = Variable(LongTensor([[SOS_token] * beam_width]))

    stored_predecessors = list()
    stored_emitted_symbols = list()

    for j in range(max_length):
        decoder_output, hidden = decoder(input_var, hidden)

        sequence_scores = _inflate(sequence_scores, v)
        sequence_scores += decoder_output

        scores, candidates = sequence_scores.view(1, -1).topk(beam_width)

        input_var = (candidates % v).view(1, beam_width)
        sequence_scores = scores.view(beam_width, 1)

        predecessors = (candidates / v +
                        pos_index.expand_as(candidates)).view(beam_width, 1)
        hidden = hidden.index_select(1, cast(torch.LongTensor, predecessors.squeeze()))

        eos_indices = input_var.data.eq(EOS_token)
        if eos_indices.nonzero().dim() > 0:
            sequence_scores.data.masked_fill_(torch.transpose(eos_indices, 0, 1),
                                              -float('inf'))

        stored_predecessors.append(predecessors)
        stored_emitted_symbols.append(torch.transpose(input_var, 0, 1))


    # Trace back from the final three highest scores
    _, next_idxs = sequence_scores.view(beam_width).sort(descending=True)
    seqs = [] # type: List[List[SomeLongTensor]]
    eos_found = 0
    for i in range(max_length - 1, -1, -1):
        # The next column of symbols from the end
        next_symbols = stored_emitted_symbols[i].view(beam_width) \
                                                .index_select(0, next_idxs).data
        # The predecessors of that column
        next_idxs = stored_predecessors[i].view(beam_width).index_select(0, next_idxs)

        # Handle sequences that ended early
        eos_indices = stored_emitted_symbols[i].data.squeeze(1).eq(EOS_token).nonzero()
        if eos_indices.dim() > 0:
            for j in range(eos_indices.size(0)-1, -1, -1):
                idx = eos_indices[j]

                res_k_idx = beam_width - (eos_found % beam_width) - 1
                eos_found += 1
                res_idx = res_k_idx

                next_idxs[res_idx] = stored_predecessors[i][idx[0]]
                next_symbols[res_idx] = stored_emitted_symbols[i][idx[0]].data[0]

        # Commit the result
        seqs.insert(0, next_symbols)

    # Transpose
    int_seqs = [[data[i][0] for data in seqs] for i in range(beam_width)]
    # Cut off EOS tokens
    int_seqs = [list(takewhile(lambda x: x != EOS_token, seq)) for seq in int_seqs]

    return int_seqs

def _mask_symbol_scores(self, score : List[float], idx : int,
                        masking_score : float=-float('inf')) -> None:
    score[idx] = masking_score

def _mask(tensor : torch.Tensor, idx : SomeLongTensor,
          dim : int=0, masking_score : float=-float('inf')):
    if len(idx.size()) > 0:
        indices = idx[:,0]
        tensor.index_fill_(dim, indices, masking_score)
