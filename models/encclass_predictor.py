#!/usr/bin/env python3

import signal
import argparse
import time
import signal
import sys

from format import read_pair
from text_encoder import encode_context, decode_context, context_vocab_size, \
    get_encoder_state, set_encoder_state

from models.encdecrnn_predictor import inputFromSentence

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
import torch.utils.data as data
import torch.cuda

from models.tactic_predictor import TacticPredictor
from typing import Dict, List, Union, Any, Tuple, Iterable, cast

from util import *

SomeLongTensor = Union[torch.cuda.LongTensor, torch.LongTensor]
SomeFloatTensor = Union[torch.cuda.FloatTensor, torch.FloatTensor]

class EncClassPredictor(TacticPredictor):
    def load_saved_state(self, filename : str) -> None:
        global idx_to_stem
        global stem_to_idx
        checkpoint = torch.load(filename)
        assert checkpoint['text-encoder']
        assert checkpoint['neural-encoder']
        assert checkpoint['num-encoder-layers']
        assert checkpoint['max-length']
        assert checkpoint['num-tactic-stems']
        assert checkpoint['hidden-size']
        assert checkpoint['stem-to-idx']
        assert checkpoint['idx-to-stem']
        idx_to_stem = checkpoint['idx-to-stem']
        stem_to_idx = checkpoint['stem-to-idx']

        set_encoder_state(checkpoint['text-encoder'])
        self.vocab_size = context_vocab_size()
        self.encoder = maybe_cuda(RNNClassifier(self.vocab_size,
                                                checkpoint['hidden-size'],
                                                checkpoint['num-tactic-stems'],
                                                checkpoint['num-encoder-layers']))
        self.encoder.load_state_dict(checkpoint['neural-encoder'])
        self.max_length = checkpoint["max-length"]

    def __init__(self, options : Dict[str, Any]) -> None:
        assert(options["filename"])
        self.load_saved_state(options["filename"])

    def predictKTactics(self, in_data : Dict[str, str], k : int) -> List[str]:
        in_sentence = LongTensor(inputFromSentence(encode_context(in_data["goal"]), self.max_length))\
                      .view(1, -1)
        prediction_distribution = self.encoder.run(in_sentence)
        _, stem_idxs = prediction_distribution.view(-1).topk(k)
        return [decode_stem(stem_idx.data[0]) for stem_idx in stem_idxs]

class RNNClassifier(nn.Module):
    def __init__(self, input_vocab_size : int, hidden_size : int, output_vocab_size: int,
                 num_layers : int, batch_size : int=1) -> None:
        super(RNNClassifier, self).__init__()
        self.num_layers = num_layers
        self.input_vocab_size = input_vocab_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = maybe_cuda(nn.Embedding(input_vocab_size, hidden_size))
        self.gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self.out = maybe_cuda(nn.Linear(hidden_size, output_vocab_size))
        self.softmax = maybe_cuda(nn.LogSoftmax(dim=1))

    def forward(self, input : torch.FloatTensor, hidden : torch.FloatTensor) \
        -> Tuple[torch.FloatTensor, torch.FloatTensor] :
        output = self.embedding(input).view(1, self.batch_size, -1)
        for i in range(self.num_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return maybe_cuda(Variable(torch.zeros(1, self.batch_size, self.hidden_size)))

    def run(self, input : torch.LongTensor):
        in_var = maybe_cuda(Variable(input))
        hidden = self.initHidden()
        for i in range(in_var.size()[1]):
            output, hidden = self(in_var[:,i], hidden)
        return output

def read_text_data(data_path : str, max_size:int=None) -> DataSet:
    data_set = []
    with open(data_path, mode="r") as data_file:
        pair = read_pair(data_file)
        counter = 0
        while pair and (not max_size or counter < max_size):
            context, tactic = pair
            counter += 1
            data_set.append([encode_context(context),
                             encode_stem(tactic)])
            pair = read_pair(data_file)
    assert len(data_set) > 0
    return data_set

Checkpoint = Dict[Any, Any]

def train(dataset : DataSet,
          input_vocab_size : int, output_vocab_size : int, hidden_size : int,
          learning_rate : float, num_encoder_layers : int,
          max_length : int, num_epochs : int, batch_size : int,
          print_every : int) -> Iterable[Checkpoint]:
    print("Initializing PyTorch...")
    in_stream = [inputFromSentence(datum[0], max_length) for datum in dataset]
    out_stream = [datum[1] for datum in dataset]
    dataset = data.TensorDataset(torch.LongTensor(in_stream),
                                 torch.LongTensor(out_stream))
    dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=0,
                                 shuffle=True, pin_memory=True, drop_last=True)

    encoder = maybe_cuda(
        RNNClassifier(input_vocab_size, hidden_size, output_vocab_size,
                      num_encoder_layers,
                      batch_size=batch_size))
    optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    criterion = maybe_cuda(nn.NLLLoss())
    adjuster = scheduler.StepLR(optimizer, 5, gamma=0.5)
    lsoftmax = maybe_cuda(nn.LogSoftmax(1))

    start=time.time()
    num_items = len(dataset) * num_epochs
    total_loss = 0

    print("Training...")
    for epoch in range(num_epochs):
        print("Epoch {}".format(epoch))
        adjuster.step()
        for batch_num, (input_batch, output_batch) in enumerate(dataloader):

            optimizer.zero_grad()

            prediction_distribution = encoder.run(
                cast(torch.LongTensor, input_batch))
            loss = 0
            output_var = maybe_cuda(Variable(output_batch))
            # print("Got distribution: {}"
            #       .format(str_1d_float_tensor(prediction_distribution[0])))
            loss += criterion(prediction_distribution, output_var)
            # print("Correct answer: {}".format(output_var.data[0]))
            loss.backward()

            optimizer.step()

            total_loss += loss.data[0] * batch_size

            if (batch_num + 1) % print_every == 0:

                items_processed = (batch_num + 1) * batch_size + epoch * len(dataset)
                progress = items_processed / num_items
                print("{} ({:7} {:5.2f}%) {:.4f}".
                      format(timeSince(start, progress),
                             items_processed, progress * 100,
                             total_loss / items_processed))

        yield encoder.state_dict()

def exit_early(signal, frame):
    sys.exit(0)

def take_args(args) -> Tuple[str, Any]:
    parser = argparse.ArgumentParser(description=
                                     "pytorch model for proverbot")
    parser.add_argument("scrape_file")
    parser.add_argument("save_file")
    parser.add_argument("--num-epochs", dest="num_epochs", default=50, type=int)
    parser.add_argument("--batch-size", dest="batch_size", default=256, type=int)
    parser.add_argument("--max-length", dest="max_length", default=100, type=int)
    parser.add_argument("--print-every", dest="print_every", default=10, type=int)
    parser.add_argument("--hidden-size", dest="hidden_size", default=128, type=int)
    parser.add_argument("--learning-rate", dest="learning_rate",
                        default=.4, type=float)
    parser.add_argument("--num-encoder-layers", dest="num_encoder_layers",
                        default=3, type=int)
    return parser.parse_args(args)

def main(args) -> None:
    signal.signal(signal.SIGINT, exit_early)
    args = take_args(args)
    print("Reading dataset...")
    dataset = read_text_data(args.scrape_file)

    checkpoints = train(dataset,
                        context_vocab_size(), num_stems(), args.hidden_size,
                        args.learning_rate, args.num_encoder_layers,
                        args.max_length, args.num_epochs, args.batch_size,
                        args.print_every)

    for epoch, encoder_state in enumerate(checkpoints):
        state = {'epoch':epoch,
                 'text-encoder':get_encoder_state(),
                 'neural-encoder':encoder_state,
                 'num-encoder-layers':args.num_encoder_layers,
                 'num-tactic-stems':num_stems(),
                 'max-length': args.max_length,
                 'hidden-size' : args.hidden_size,
                 'stem-to-idx' : stem_to_idx,
                 'idx-to-stem' : idx_to_stem}
        with open(args.save_file, 'wb') as f:
            print("=> Saving checkpoint at epoch {}".
                  format(epoch))
            torch.save(state, f)

stem_to_idx = {}
idx_to_stem = {}
def encode_stem(tactic):
    stem = get_stem(tactic)
    if stem in stem_to_idx:
        return stem_to_idx[stem]
    else:
        new_idx = num_stems()
        stem_to_idx[stem] = new_idx
        idx_to_stem[new_idx] = stem
        return new_idx

def decode_stem(idx):
    return idx_to_stem[idx] + "."

def num_stems():
    return len(idx_to_stem)
