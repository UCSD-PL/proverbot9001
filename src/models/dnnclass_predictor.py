#!/usr/bin/env python3.7

import signal
import argparse
import time
import sys
import threading
import math

from tokenizer import Tokenizer, tokenizers
from data import get_text_data, \
    encode_bag_classify_data, encode_bag_classify_input, ClassifyBagDataset
from context_filter import get_context_filter
from util import *
from models.args import take_std_args
from models.components import DNNClassifier

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
import torch.utils.data as data
import torch.cuda
from torch.optim import Optimizer

from models.tactic_predictor import TacticPredictor, Prediction, TacticContext
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
            ("# network layers", args.num_decoder_layers),
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
                                                args.num_decoder_layers))
        self.network.load_state_dict(checkpoint['network-state'])
        self.criterion = maybe_cuda(nn.NLLLoss())
        self.lock = threading.Lock()

    def __init__(self, options : Dict[str, Any]) -> None:
        assert options["filename"]
        assert options["skip-nochange-tac"]
        self.load_saved_state(options["filename"])
        self.skip_nochange_tac = options["skip-nochange-tac"]

    def predictDistribution(self, in_data : TacticContext) \
        -> torch.FloatTensor:
        in_vec = maybe_cuda(Variable(torch.FloatTensor(
            encode_bag_classify_input(in_data.goal, self.tokenizer))))\
            .view(1, -1)
        return self.network(in_vec)

    def predictKTactics(self, in_data : TacticContext, k : int) \
        -> List[Prediction]:
        self.lock.acquire()
        distribution = self.predictDistribution(in_data)
        certainties_and_idxs = distribution.squeeze().topk(k)
        results = [Prediction(self.embedding.decode_token(idx.data[0]) + ".",
                              math.exp(certainty.data[0]))
                   for certainty, idx in zip(*certainties_and_idxs)]
        self.lock.release()
        return results
    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int,
                                correct : str) -> Tuple[List[Prediction], float]:
        self.lock.acquire()
        distribution = self.predictDistribution(in_data)
        stem = get_stem(correct)
        if self.embedding.has_token(stem):
            output_var = maybe_cuda(
                Variable(torch.LongTensor([self.embedding.encode_token(stem)])))
            loss = self.criterion(distribution.view(1, -1), output_var).item()
        else:
            loss = 0

        certainties, idxs = distribution.squeeze().topk(k)
        predictions_and_certainties = \
            [Prediction(self.embedding.decode_token(idx.item()) + ".",
                        math.exp(certainty.item()))
             for certainty, idx in zip(list(certainties), list(idxs))]
        self.lock.release()

        return predictions_and_certainties, loss
    def getOptions(self) -> List[Tuple[str, str]]:
        return self.options

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
                                       num_layers))

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

def main(arg_list : List[str]) -> None:
    args = take_std_args(arg_list, "non-recurrent neural network "
                                     "model for Proverbot9001")

    raw_dataset = get_text_data(args)
    dataset, tokenizer, embedding = encode_bag_classify_data(raw_dataset,
                                                             tokenizers[args.tokenizer],
                                                             args.num_keywords, 2)
    checkpoints = train(dataset,
                        tokenizer.numTokens(), args.hidden_size, embedding.num_tokens(),
                        args.num_decoder_layers, args.batch_size, args.learning_rate,
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
