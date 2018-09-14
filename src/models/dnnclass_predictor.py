#!/usr/bin/env python3

import signal
import argparse
import time
import sys
import threading
import math

from tokenizer import Tokenizer, tokenizers
from data import read_text_data, filter_data, \
    encode_bag_classify_data, encode_bag_classify_input, ClassifyBagDataset
from context_filter import get_context_filter
from util import *

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
import torch.utils.data as data
import torch.cuda
from torch.optim import Optimizer

from models.tactic_predictor import TacticPredictor
from typing import Dict, List, Union, Any, Tuple, Iterable, Callable, cast

from serapi_instance import get_stem

class DNNClassPredictor(TacticPredictor):
    def load_saved_state(self, filename : str) -> None:
        checkpoint = torch.load(filename)
        assert checkpoint['tokenizer']
        assert checkpoint['embedding']
        assert checkpoint['network-state']
        assert checkpoint['training-args']

        args = checkpoint['training-args']
        self.options = [
            ("tokenizer", args.tokenizer),
            ("# network layers", args.num_layers),
            ("hidden size", args.hidden_size),
            ("# keywords", args.num_keywords),
            ("learning rate", args.learning_rate),
            ("# epochs", args.num_epochs),
            ("optimizer", args.optimizer),
            ("gamma", args.gamma),
            ("epoch step", args.epoch_step),
            ("context filter", args.context_filter),
        ]

        self.tokenizer = checkpoint['tokenizer']
        self.embedding = checkpoint['embedding']

        self.network = maybe_cuda(DNNClassifier(self.tokenizer.numTokens(),
                                                args.hidden_size,
                                                self.embedding.num_tokens(),
                                                args.num_layers))
        self.network.load_state_dict(checkpoint['network-state'])
        self.criterion = maybe_cuda(nn.NLLLoss())
        self.lock = threading.Lock()

    def __init__(self, options : Dict[str, Any]) -> None:
        assert options["filename"]
        self.load_saved_state(options["filename"])

    def predictDistribution(self, in_data : Dict[str, str]) -> torch.FloatTensor:
        in_vec = maybe_cuda(Variable(torch.FloatTensor(
            encode_bag_classify_input(in_data["goal"], self.tokenizer))))\
            .view(1, -1)
        return self.network(in_vec)

    def predictKTactics(self, in_data : Dict[str, str], k : int) \
        -> List[Tuple[str, float]]:
        self.lock.acquire()
        distribution = self.predictDistribution(in_data)
        certainties_and_idxs = distribution.squeeze().topk(k)
        results = [(self.embedding.decode_token(idx.data[0]) + ".",
                    math.exp(certainty.data[0]))
                   for certainty, idx in zip(*certainties_and_idxs)]
        self.lock.release()
        return results
    def predictKTacticsWithLoss(self, in_data : Dict[str, str], k : int,
                                correct : str) -> Tuple[List[Tuple[str, float]], float]:
        self.lock.acquire()
        distribution = self.predictDistribution(in_data)
        stem = get_stem(correct)
        if self.embedding.has_token(stem):
            output_var = maybe_cuda(
                Variable(torch.LongTensor([self.embedding.encode_token(stem)])))
            loss = self.criterion(distribution.view(1, -1), output_var).data[0]
        else:
            loss = 0

        certainties_and_idxs = distribution.squeeze().topk(k)
        predictions_and_certainties = [(self.embedding.decode_token(idx.data[0]) + ".",
                                        math.exp(certainty.data[0]))
                                       for certainty, idx in certainties_and_idxs]
        self.lock.release()

        return predictions_and_certainties, loss
    def getOptions(self) -> List[Tuple[str, str]]:
        return self.options

class DNNClassifier(nn.Module):
    def __init__(self, input_vocab_size : int, hidden_size : int, output_vocab_size : int,
                 num_layers : int, batch_size : int=1) -> None:
        super(DNNClassifier, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.in_layer = maybe_cuda(nn.Linear(input_vocab_size, hidden_size))
        self.layers = [maybe_cuda(nn.Linear(hidden_size, hidden_size))
                       for _ in range(num_layers)]
        self.out_layer = maybe_cuda(nn.Linear(hidden_size, output_vocab_size))
        self.softmax = maybe_cuda(nn.LogSoftmax(dim=1))

    def forward(self, input : torch.FloatTensor) -> torch.FloatTensor:
        layer_values = self.in_layer(input)
        for i in range(self.num_layers):
            layer_values = F.relu(layer_values)
            layer_values = self.layers[i](layer_values)
        return self.softmax(self.out_layer(layer_values))

optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
}

Checkpoint = Tuple[Dict[Any, Any], float]

def train(dataset : ClassifyBagDataset,
          input_vocab_size : int, hidden_size : int, output_vocab_size : int,
          num_layers : int, batch_size : int, learning_rate : float, gamma : float,
          epoch_step : int, num_epochs : int,
          print_every : int, optimizer_f : Callable[..., Optimizer]) \
          -> Iterable[Checkpoint]:
    print("Initializing PyTorch...")
    inputs, outputs = zip(*dataset)
    dataloader = data.DataLoader(data.TensorDataset(torch.FloatTensor(inputs),
                                                    torch.LongTensor(outputs)),
                                 batch_size = batch_size, num_workers = 0,
                                 shuffle=True, pin_memory=True, drop_last=True)
    network = maybe_cuda(DNNClassifier(input_vocab_size, hidden_size, output_vocab_size,
                                       num_layers, batch_size=batch_size))

    optimizer = optimizer_f(network.parameters(), lr=learning_rate)
    criterion = maybe_cuda(nn.NLLLoss())
    adjuster = scheduler.StepLR(optimizer, epoch_step, gamma=gamma)

    start = time.time()
    num_items = len(dataset) * num_epochs
    total_loss = 0

    print("Training...")
    for epoch in range(num_epochs):
        print("Epoch {}".format(epoch))
        adjuster.step()

        for batch_num, (input_batch, output_batch) in enumerate(dataloader):

            optimizer.zero_grad()
            input_var = maybe_cuda(Variable(input_batch))
            output_var = maybe_cuda(Variable(output_batch))

            prediction_distribution = network(input_var)

            loss = cast(torch.FloatTensor, 0)
            # print("prediction_distribution.size(): {}"
            #       .format(prediction_distribution.size()))
            loss += criterion(prediction_distribution.squeeze(), output_var)
            loss.backward()

            optimizer.step()
            total_loss += loss.data[0] * batch_size

            if (batch_num + 1) % print_every == 0:
                items_processed = (batch_num + 1) * batch_size + epoch * len(dataset)
                progress = items_processed / num_items
                print("{} ({:7} {:5.2f}%) {:.4f}"
                      .format(timeSince(start, progress),
                              items_processed, progress * 100,
                              total_loss / items_processed))
        yield (network.state_dict(), total_loss / items_processed)

def exit_early(signal, frame):
    sys.exit(0)

def take_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=
                                     "non-recurrent neural network "
                                     "model for Proverbot9001")
    parser.add_argument("scrape_file")
    parser.add_argument("save_file")
    parser.add_argument("--tokenizer",
                        choices=list(tokenizers.keys()), type=str,
                        default=list(tokenizers.keys())[0])
    parser.add_argument("--num-keywords", dest="num_keywords", default=100, type=int)
    parser.add_argument("--hidden-size", dest="hidden_size", default=256, type=int)
    parser.add_argument("--num-layers", dest="num_layers", default=3, type=int)
    parser.add_argument("--batch-size", dest="batch_size", default=256, type=int)
    parser.add_argument("--learning-rate", dest="learning_rate", default=.4, type=float)
    parser.add_argument("--num-epochs", dest="num_epochs", default=50, type=int)
    parser.add_argument("--epoch-step", dest="epoch_step", default=5, type=int)
    parser.add_argument("--gamma", dest="gamma", default=0.5, type=float)
    parser.add_argument("--print-every", dest="print_every", default=10, type=int)
    parser.add_argument("--optimizer", choices=list(optimizers.keys()), type=str,
                        default=list(optimizers.keys())[0])
    parser.add_argument("--context-filter", dest="context_filter",
                        type=str, default="default")
    return parser.parse_args(args)

def main(arg_list : List[str]) -> None:
    signal.signal(signal.SIGINT, exit_early)
    args = take_args(arg_list)

    print("Reading dataset...")
    raw_data = read_text_data(args.scrape_file)
    print("Encoding/Filtering dataset...")
    filtered_data = filter_data(raw_data, get_context_filter(args.context_filter))
    dataset, tokenizer, embedding = encode_bag_classify_data(filtered_data,
                                                             tokenizers[args.tokenizer],
                                                             args.num_keywords, 2)
    checkpoints = train(dataset,
                        tokenizer.numTokens(), args.hidden_size, embedding.num_tokens(),
                        args.num_layers, args.batch_size, args.learning_rate,
                        args.gamma, args.epoch_step,
                        args.num_epochs, args.print_every, optimizers[args.optimizer])

    for epoch, (network_state, training_loss) in enumerate(checkpoints):
        state = {'epoch': epoch,
                 'training-loss': training_loss,
                 'tokenizer':tokenizer,
                 'embedding':embedding,
                 'network-state':network_state,
                 'training-args': args,
        }
        with open(args.save_file, 'wb') as f:
            print("=> Saving checkpoint at epoch {}"
                  .format(epoch))
            torch.save(state, f)
