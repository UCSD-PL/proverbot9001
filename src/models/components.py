#!/usr/bin/env python3.7

from typing import Dict, Any, List
from abc import ABCMeta, abstractmethod

class Embedding:
    @abstractmethod
    def __init__(self) -> None: pass
    @abstractmethod
    def encode_token(self, token : str) -> int : pass
    @abstractmethod
    def decode_token(self, idx : int) -> str: pass
    @abstractmethod
    def num_tokens(self) -> int: pass
    @abstractmethod
    def has_token(self, token : str) -> bool : pass
    pass

class SimpleEmbedding(Embedding):
    def __init__(self) -> None:
        self.tokens_to_indices = {} #type: Dict[str, int]
        self.indices_to_tokens = {} #type: Dict[int, str]
    def encode_token(self, token : str) -> int :
        if token in self.tokens_to_indices:
            return self.tokens_to_indices[token]
        else:
            new_idx = len(self.tokens_to_indices)
            self.tokens_to_indices[token] = new_idx
            self.indices_to_tokens[new_idx] = token
            return new_idx

    def decode_token(self, idx : int) -> str:
        return self.indices_to_tokens[idx]
    def num_tokens(self) -> int:
        return len(self.indices_to_tokens)
    def has_token(self, token : str) -> bool :
        return token in self.tokens_to_indices

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import *
import argparse
def add_nn_args(parser : argparse.ArgumentParser,
                default_values : Dict[str, Any] = {}) -> None:
    parser.add_argument("--num-epochs", dest="num_epochs", type=int,
                        default=default_values.get("num-epochs", 20))
    parser.add_argument("--batch-size", dest="batch_size", type=int,
                        default=default_values.get("batch-size", 256))
    parser.add_argument("--start-from", dest="start_from", type=str,
                        default=default_values.get("start-from", None))
    parser.add_argument("--print-every", dest="print_every", type=int,
                        default=default_values.get("print-every", 5))
    parser.add_argument("--learning-rate", dest="learning_rate", type=float,
                        default=default_values.get("learning-rate", .7))
    parser.add_argument("--epoch-step", dest="epoch_step", type=int,
                        default=default_values.get("epoch-step", 10))
    parser.add_argument("--hidden-size", dest="hidden_size", type=int,
                        default=default_values.get("hidden-size", 128))
    parser.add_argument("--num-layers", dest="num_layers", type=int,
                        default=default_values.get("num-layers", 3))
    parser.add_argument("--gamma", dest="gamma", type=float,
                        default=default_values.get("gamma", 0.8))
    parser.add_argument("--optimizer",
                        choices=list(optimizers.keys()), type=str,
                        default=default_values.get("optimizer",
                                                   list(optimizers.keys())[0]))


class DNNClassifier(nn.Module):
    def __init__(self, input_vocab_size : int, hidden_size : int, output_vocab_size : int,
                 num_layers : int) -> None:
        super(DNNClassifier, self).__init__()
        self.num_layers = num_layers
        self.in_layer = maybe_cuda(nn.Linear(input_vocab_size, hidden_size))
        for i in range(num_layers - 1):
            self.add_module("_layer{}".format(i),
                            maybe_cuda(nn.Linear(hidden_size, hidden_size)))
        self.out_layer = maybe_cuda(nn.Linear(hidden_size, output_vocab_size))
        self.softmax = maybe_cuda(nn.LogSoftmax(dim=1))

    def forward(self, input : torch.FloatTensor) -> torch.FloatTensor:
        layer_values = self.in_layer(maybe_cuda(Variable(input)))
        for i in range(self.num_layers - 1):
            layer_values = F.relu(layer_values)
            layer_values = getattr(self, "_layer{}".format(i))(layer_values)
        layer_values = F.relu(layer_values)
        return self.softmax(self.out_layer(layer_values)).view(input.size()[0], -1)

class EncoderDNN(nn.Module):
    def __init__(self, input_vocab_size : int, hidden_size : int, output_vocab_size : int,
                 num_layers : int, batch_size : int=1) -> None:
        super(EncoderDNN, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.in_layer = maybe_cuda(nn.Linear(input_vocab_size, hidden_size))
        self.layers = [maybe_cuda(nn.Linear(hidden_size, hidden_size))
                       for _ in range(num_layers)]
        self.out_layer = maybe_cuda(nn.Linear(hidden_size, output_vocab_size))

    def forward(self, input : torch.FloatTensor) -> torch.FloatTensor:
        layer_values = self.in_layer(input)
        for i in range(self.num_layers):
            layer_values = F.relu(layer_values)
            layer_values = self.layers[i](layer_values)
        return self.out_layer(layer_values)

class DecoderGRU(nn.Module):
    def __init__(self, input_size : int, hidden_size : int,
                 num_layers : int, batch_size : int=1) -> None:
        super(DecoderGRU, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.gru = maybe_cuda(nn.GRU(input_size, hidden_size,
                                     num_layers=self.num_layers, batch_first=True))
        self.softmax = maybe_cuda(nn.LogSoftmax(dim=1))

    def forward(self, input : torch.FloatTensor, hidden : torch.FloatTensor) \
        -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        input = input.contiguous()
        hidden = hidden.expand(self.num_layers, -1, -1).contiguous()
        output, hidden = self.gru(input, hidden)
        return self.softmax(input), hidden[0]
