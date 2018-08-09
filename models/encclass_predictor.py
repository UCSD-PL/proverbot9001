#!/usr/bin/env python3

import signal
import argparse
import time
import signal
import sys

from models.encdecrnn_predictor import inputFromSentence
from tokenizer import Tokenizer, tokenizers
from data import read_text_data, encode_seq_classify_data, ClassifySequenceDataset
from util import *

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

SomeLongTensor = Union[torch.cuda.LongTensor, torch.LongTensor]
SomeFloatTensor = Union[torch.cuda.FloatTensor, torch.FloatTensor]

class EncClassPredictor(TacticPredictor):
    def load_saved_state(self, filename : str) -> None:
        checkpoint = torch.load(filename)
        assert checkpoint['tokenizer']
        assert checkpoint['tokenizer-name']
        assert checkpoint['embedding']
        assert checkpoint['neural-encoder']
        assert checkpoint['num-encoder-layers']
        assert checkpoint['max-length']
        assert checkpoint['hidden-size']
        assert checkpoint['num-keywords']

        self.options = [("tokenizer", checkpoint['tokenizer-name']),
                        ("# encoder layers", checkpoint['num-encoder-layers']),
                        ("input length", checkpoint['max-length']),
                        ("hidden size", checkpoint['hidden-size']),
                        ("# keywords", checkpoint['num-keywords'])]

        self.tokenizer = checkpoint['tokenizer']
        self.embedding = checkpoint['embedding']
        self.encoder = maybe_cuda(RNNClassifier(self.tokenizer.numTokens(),
                                                checkpoint['hidden-size'],
                                                self.embedding.numTokens(),
                                                checkpoint['num-encoder-layers']))
        self.encoder.load_state_dict(checkpoint['neural-encoder'])
        self.max_length = checkpoint["max-length"]

    def __init__(self, options : Dict[str, Any]) -> None:
        assert(options["filename"])
        self.load_saved_state(options["filename"])

    def predictKTactics(self, in_data : Dict[str, str], k : int) -> List[str]:
        in_sentence = LongTensor(inputFromSentence(
            self.tokenizer.toTokenList(in_data["goal"]),
            self.max_length))\
            .view(1, -1)
        prediction_distribution = self.encoder.run(in_sentence)
        _, stem_idxs = prediction_distribution.view(-1).topk(k)
        return [self.embedding.decode_token(stem_idx.data[0]) for stem_idx in stem_idxs]

    def getOptions(self) -> List[Tuple[str, str]]:
        return self.options

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

Checkpoint = Dict[Any, Any]

def train(dataset : ClassifySequenceDataset,
          input_vocab_size : int, output_vocab_size : int, hidden_size : int,
          learning_rate : float, num_encoder_layers : int,
          max_length : int, num_epochs : int, batch_size : int,
          print_every : int) -> Iterable[Checkpoint]:
    print("Initializing PyTorch...")
    in_stream = [inputFromSentence(datum[0], max_length) for datum in dataset]
    out_stream = [datum[1] for datum in dataset]
    dataloader = data.DataLoader(data.TensorDataset(torch.LongTensor(in_stream),
                                                    torch.LongTensor(out_stream)),
                                 batch_size=batch_size, num_workers=0,
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
            loss = cast(torch.FloatTensor, 0)
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

def take_args(args) -> argparse.Namespace:
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
    parser.add_argument("--tokenizer",
                        choices=list(tokenizers.keys()), type=str,
                        default=list(tokenizers.keys())[0])
    parser.add_argument("--num-keywords", dest="num_keywords", default=100, type=int)
    return parser.parse_args(args)

def main(arg_list : List[str]) -> None:
    signal.signal(signal.SIGINT, exit_early)
    args = take_args(arg_list)
    print("Reading dataset...")

    raw_data = read_text_data(args.scrape_file)
    dataset, tokenizer, embedding = encode_seq_classify_data(raw_data,
                                                             tokenizers[args.tokenizer],
                                                             args.num_keywords, 2)

    checkpoints = train(dataset, tokenizer.numTokens(), embedding.num_tokens(),
                        args.hidden_size,
                        args.learning_rate, args.num_encoder_layers,
                        args.max_length, args.num_epochs, args.batch_size,
                        args.print_every)

    for epoch, encoder_state in enumerate(checkpoints):
        state = {'epoch':epoch,
                 'tokenizer':tokenizer,
                 'tokenizer-name':args.tokenizer,
                 'embedding': embedding,
                 'neural-encoder':encoder_state,
                 'num-encoder-layers':args.num_encoder_layers,
                 'max-length': args.max_length,
                 'hidden-size' : args.hidden_size,
                 'num-keywords' : args.num_keywords,
        }
        with open(args.save_file, 'wb') as f:
            print("=> Saving checkpoint at epoch {}".
                  format(epoch))
            torch.save(state, f)
