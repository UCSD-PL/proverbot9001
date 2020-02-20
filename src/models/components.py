#!/usr/bin/env python3.7
##########################################################################
#
#    This file is part of Proverbot9001.
#
#    Proverbot9001 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Proverbot9001 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Proverbot9001.  If not, see <https://www.gnu.org/licenses/>.
#
#    Copyright 2019 Alex Sanchez-Stern and Yousef Alhessi
#
##########################################################################

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
from typing import TypeVar, Generic
import argparse

S = TypeVar("S")

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
    parser.add_argument("--max-premises", dest="max_premises", type=int,
                        default=default_values.get("max-premises", 20))

class StraightlineClassifierModel(Generic[S], metaclass=ABCMeta):
    @staticmethod
    def add_args_to_parser(parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) \
                           -> None:
        pass
    @abstractmethod
    def __init__(self, args : argparse.Namespace,
                 input_vocab_size : int, output_vocab_size : int) -> None:
        pass
    @abstractmethod
    def checkpoints(self, inputs : List[List[float]], outputs : List[int]) \
        -> Iterable[S]:
        pass
    @abstractmethod
    def setState(self, state : S) -> None:
        pass

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

class DNNScorer(nn.Module):
    def __init__(self, input_vocab_size : int, hidden_size : int, num_layers) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.in_layer = maybe_cuda(nn.Linear(input_vocab_size, hidden_size))
        for i in range(num_layers - 1):
            self.add_module("_layer{}".format(i),
                            maybe_cuda(nn.Linear(hidden_size, hidden_size)))
        self.out_layer = maybe_cuda(nn.Linear(hidden_size, 1))

    def forward(self, input : torch.FloatTensor) -> torch.FloatTensor:
        layer_values = self.in_layer(maybe_cuda(Variable(input)))
        for i in range(self.num_layers - 1):
            layer_values = F.relu(layer_values)
            layer_values = getattr(self, "_layer{}".format(i))(layer_values)
        layer_values = F.relu(layer_values)
        return self.out_layer(layer_values)


class WordFeaturesEncoder(nn.Module):
    def __init__(self, input_vocab_sizes : List[int],
                 hidden_size : int, num_layers : int,
                 output_vocab_size : int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_word_features = len(input_vocab_sizes)
        for i, vocab_size in enumerate(input_vocab_sizes):
            self.add_module("_word_embedding{}".format(i),
                            maybe_cuda(nn.Embedding(vocab_size, hidden_size)))
        self._in_layer = maybe_cuda(nn.Linear(hidden_size * len(input_vocab_sizes),
                                              hidden_size))
        for i in range(num_layers - 1):
            self.add_module("_layer{}".format(i),
                            maybe_cuda(nn.Linear(hidden_size, hidden_size)))
        self._out_layer = maybe_cuda(nn.Linear(hidden_size, output_vocab_size))

    def forward(self, input_vec : torch.LongTensor) -> torch.FloatTensor:
        batch_size = input_vec.size()[0]
        word_embedded_features = []
        for i in range(self.num_word_features):
            word_feature_var = maybe_cuda(Variable(input_vec[:,i]))
            embedded = getattr(self, "_word_embedding{}".format(i))(word_feature_var)\
                .view(batch_size, self.hidden_size)
            word_embedded_features.append(embedded)
        word_embedded_features_vec = \
            torch.cat(word_embedded_features, dim=1)
        vals = self._in_layer(word_embedded_features_vec)
        for i in range(self.num_layers - 1):
            vals = F.relu(vals)
            vals = getattr(self, "_layer{}".format(i))(vals)
        vals = F.relu(vals)
        result = self._out_layer(vals).view(batch_size, -1)
        return result

class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size : int,
                 hidden_size : int,
                 output_vocab_size : int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self._word_embedding = maybe_cuda(nn.Embedding(input_vocab_size, hidden_size))
        self._gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self._out_layer = maybe_cuda(nn.Linear(hidden_size, output_vocab_size))
    def forward(self, input_seq : torch.LongTensor) -> torch.FloatTensor:
        input_var = maybe_cuda(Variable(input_seq))
        batch_size = input_seq.size()[0]
        hidden = maybe_cuda(Variable(torch.zeros(1, batch_size, self.hidden_size)))
        for i in range(input_seq.size()[1]):
            token_batch = self._word_embedding(input_var[:,i])\
                .view(1, batch_size, self.hidden_size)
            token_batch = F.relu(token_batch)
            token_out, hidden = self._gru(token_batch, hidden)
        result = self._out_layer(token_out.view(batch_size, self.hidden_size))
        return result

class EncoderDNN(nn.Module):
    def __init__(self, input_vocab_size : int, hidden_size : int, output_vocab_size : int,
                 num_layers : int, batch_size : int=1) -> None:
        super(EncoderDNN, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.in_layer = maybe_cuda(nn.Linear(input_vocab_size, hidden_size))
        for i in range(num_layers - 1):
            self.add_module("_layer{}".format(i),
                            maybe_cuda(nn.Linear(hidden_size, hidden_size)))
        self.out_layer = maybe_cuda(nn.Linear(hidden_size, output_vocab_size))

    def forward(self, input : torch.FloatTensor) -> torch.FloatTensor:
        layer_values = self.in_layer(input)
        for i in range(self.num_layers - 1):
            layer_values = F.relu(layer_values)
            layer_values = getattr(self, "_layer{}".format(i))(layer_values)
        return self.out_layer(F.relu(layer_values))

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

import sys
from sklearn import svm

svm_kernels = [
    "rbf",
    "linear",
]
class SVMClassifierModel(StraightlineClassifierModel[svm.SVC]):
    @staticmethod
    def add_args_to_parser(parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) \
                           -> None:
        parser.add_argument("--kernel", choices=svm_kernels, type=str,
                            default=svm_kernels[0])
        parser.add_argument("--gamma",type=float,
                            default=svm_kernels[0])
    def __init__(self, args : argparse.Namespace,
                 input_vocab_size : int, output_vocab_size : int) -> None:
        self._model = svm.SVC(gamma=args.gamma, kernel=args.kernel,
                              probability=args.probability,
                              verbose=args.verbose)
    def checkpoints(self, inputs : List[List[float]], outputs : List[int]) \
        -> Iterable[svm.SVC]:
        curtime = time.time()
        print("Training SVM...", end="")
        sys.stdout.flush()
        self._model.fit(inputs, outputs)
        print(" {:.2f}s".format(time.time() - curtime))
        loss = self._model.score(inputs, outputs)
        print("Training loss: {}".format(loss))
        yield self._model
    def predict(self, inputs : List[List[float]]) -> List[List[float]]:
        return self._model.predict_log_proba(inputs)
    def setState(self, state : svm.SVC) -> None:
        self._model = state

import threading
from torch import optim
import torch.optim.lr_scheduler as scheduler
import torch.utils.data as data
from dataclasses import dataclass
from typing import NamedTuple

optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
}

@dataclass(init=True)
class PredictorState:
    epoch : int

@dataclass(init=True)
class NeuralPredictorState(PredictorState):
    epoch : int
    loss : float
    weights : Dict[str, Any]

class DNNClassifierModel(StraightlineClassifierModel[NeuralPredictorState]):
    @staticmethod
    def add_args_to_parser(parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) \
                           -> None:
        add_nn_args(parser)
    def __init__(self, args : argparse.Namespace,
                 input_vocab_size : int, output_vocab_size : int) -> None:
        self._model = maybe_cuda(DNNClassifier(input_vocab_size,
                                               args.hidden_size, output_vocab_size,
                                               args.num_layers))
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.gamma = args.gamma
        self.epoch_step = args.epoch_step
        self.print_every = args.print_every
        self.optimizer_name = args.optimizer
        self._optimizer = optimizers[args.optimizer](self._model.parameters(),
                                                     lr=args.learning_rate)
        self.adjuster = scheduler.StepLR(self._optimizer, args.epoch_step,
                                         gamma=args.gamma)
        self._criterion = maybe_cuda(nn.NLLLoss())
        self._lock = threading.Lock()
        pass
    def checkpoints(self, inputs : List[List[float]], outputs : List[int]) \
        -> Iterable[NeuralPredictorState]:
        print("Building tensors")
        dataloader = data.DataLoader(data.TensorDataset(torch.FloatTensor(inputs),
                                                        torch.LongTensor(outputs)),
                                     batch_size=self.batch_size, num_workers=0,
                                     shuffle=True,pin_memory=True,drop_last=True)
        num_batches = int(len(inputs) / self.batch_size)
        dataset_size = num_batches * self.batch_size

        print("Initializing model...")
        training_start = time.time()
        for epoch in range(1, self.num_epochs):
            self.adjuster.step()
            print("Epoch {} (learning rate {:.6f})"
                  .format(epoch, self._optimizer.param_groups[0]['lr']))
            epoch_loss = 0.
            for batch_num, data_batch in enumerate(dataloader, start=1):
                self._optimizer.zero_grad()
                input_batch, output_batch = data_batch
                with autograd.detect_anomaly():
                    predictionDistribution = self._model(input_batch)
                    output_var = maybe_cuda(Variable(output_batch))
                    loss = self._criterion(predictionDistribution, output_var)
                    loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()
                if batch_num % self.print_every == 0:
                    items_processed = batch_num * self.batch_size + \
                        (epoch - 1) * dataset_size
                    progress = items_processed / (dataset_size * self.num_epochs)
                    print("{} ({:7} {:5.2f}%) {:.4f}"
                          .format(timeSince(training_start, progress),
                                  items_processed, progress * 100,
                                  epoch_loss / batch_num))
            state = self._model.state_dict()
            loss = epoch_loss / num_batches
            checkpoint = NeuralPredictorState(epoch, loss, state)
            yield checkpoint
    def predict(self, inputs : List[List[float]]) -> List[List[float]]:
        with self._lock:
            return self._model(FloatTensor(inputs)).data
    def setState(self, state : NeuralPredictorState) -> None:
        self._model.load_state_dict(state.weights)
