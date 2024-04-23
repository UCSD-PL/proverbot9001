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

import argparse
import time
import math
from typing import Dict, Any, List, Tuple, Iterable, cast, Union

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.optim.lr_scheduler as scheduler
import torch.utils.data as data

from models.tactic_predictor import TacticPredictor, Prediction

from tokenizer import tokenizers
from data import (get_text_data, encode_bag_classify_data,
                  encode_bag_classify_input)
from context_filter import get_context_filter
from util import *
from coq_serapy.contexts import TacticContext
from coq_serapy import get_stem

class WordBagClassifyPredictor(TacticPredictor):
    def load_saved_state(self, filename : str) -> None:
        checkpoint = torch.load(filename)
        assert checkpoint['stem-embeddings']

        self.embedding = checkpoint['stem-embeddings']
        self.tokenizer = checkpoint['tokenizer']
        self.linear = maybe_cuda(nn.Linear(self.tokenizer.numTokens(),
                                           self.embedding.num_tokens()))
        self.linear.load_state_dict(checkpoint['linear-state'])
        self.lsoftmax = maybe_cuda(nn.LogSoftmax(dim=1))

        self.options = checkpoint['options']
        self.criterion = maybe_cuda(nn.NLLLoss())
        pass

    def getOptions(self) -> List[Tuple[str, str]]:
        return [("classifier", str(True)),
                ("num-stems", str(self.embedding.num_tokens())),
                ] + self.options

    def __init__(self, options : Dict[str, Any]) -> None:
        assert options["filename"]
        self.load_saved_state(options["filename"])

    def predictDistribution(self, in_data : TacticContext) \
        -> torch.FloatTensor:
        in_vec = Variable(FloatTensor(encode_bag_classify_input(
            in_data.goal, self.tokenizer))) .view(1, -1)
        return self.lsoftmax(self.linear(in_vec))

    def predictKTactics(self, in_data : TacticContext, k : int) \
        -> List[Prediction]:
        distribution = self.predictDistribution(in_data)
        probs_and_indices = distribution.squeeze().topk(k)
        return [Prediction(self.embedding.decode_token(idx.data[0]) + ".",
                           math.exp(certainty.data[0]))
                for certainty, idx in probs_and_indices]

    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int,
                                correct : str) -> Tuple[List[Prediction], float]:
        distribution = self.predictDistribution(in_data)
        stem = get_stem(correct)
        if self.embedding.has_token(stem):
            output_var = maybe_cuda(
                Variable(torch. LongTensor([self.embedding.encode_token(stem)])))
            loss = self.criterion(distribution, output_var).data[0]
        else:
            loss = 0

        probs_and_indices = distribution.squeeze().topk(k)
        predictions = [Prediction(self.embedding.decode_token(idx.data[0]) + ".",
                                  math.exp(certainty.data[0]))
                       for certainty, idx in probs_and_indices]
        return predictions, loss

Checkpoint = Tuple[Dict[Any, Any], float]

def main(args_list : List[str]) -> None:
    parser = argparse.ArgumentParser(description=
                                     "A second-tier predictor which predicts tactic "
                                     "stems based on word frequency in the goal")
    parser.add_argument("--learning-rate", dest="learning_rate", default=.5, type=float)
    parser.add_argument("--num-epochs", dest="num_epochs", default=10, type=int)
    parser.add_argument("--batch-size", dest="batch_size", default=256, type=int)
    parser.add_argument("--print-every", dest="print_every", default=10, type=int)
    parser.add_argument("--epoch-step", dest="epoch_step", default=5, type=int)
    parser.add_argument("--gamma", dest="gamma", default=0.5, type=float)
    parser.add_argument("--optimizer", default="SGD",
                        choices=list(optimizers.keys()), type=str)
    parser.add_argument("--context-filter", dest="context_filter",
                        type=str, default="default")
    parser.add_argument("scrape_file")
    parser.add_argument("save_file")
    args = parser.parse_args(args_list)
    print("Loading dataset...")

    text_dataset = get_text_data(args)
    samples, tokenizer, embedding = encode_bag_classify_data(text_dataset,
                                                             tokenizers["char-fallback"],
                                                             100, 2)

    checkpoints = train(samples, args.learning_rate,
                        args.num_epochs, args.batch_size,
                        embedding.num_tokens(), args.print_every,
                        args.gamma, args.epoch_step, args.optimizer)

    for epoch, (linear_state, loss) in enumerate(checkpoints, start=1):
        state = {'epoch':epoch,
                 'text-encoder':tokenizer,
                 'linear-state': linear_state,
                 'stem-embeddings': embedding,
                 'options': [
                     ("# epochs", str(epoch)),
                     ("learning rate", str(args.learning_rate)),
                     ("batch size", str(args.batch_size)),
                     ("epoch step", str(args.epoch_step)),
                     ("gamma", str(args.gamma)),
                     ("dataset size", str(len(samples))),
                     ("optimizer", args.optimizer),
                     ("training loss", "{:10.2f}".format(loss)),
                     ("context filter", args.context_filter),
                 ]}
        with open(args.save_file, 'wb') as f:
            print("=> Saving checkpoint at epoch {}".
                  format(epoch))
            torch.save(state, f)

optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "RMSprop": optim.RMSprop
}

def train(dataset, learning_rate : float, num_epochs : int,
          batch_size : int, num_stems: int, print_every : int,
          gamma : float, epoch_step : int, optimizer_type : str) -> Iterable[Checkpoint]:
    print("Initializing PyTorch...")
    assert len(dataset[0][0]) > 10 and len(dataset[0][0]) < 1000
    linear = maybe_cuda(nn.Linear(len(dataset[0][0]), num_stems))
    lsoftmax = maybe_cuda(nn.LogSoftmax(1))

    inputs, outputs = zip(*dataset)
    dataloader = data.DataLoader(
        data.TensorDataset(
            torch.FloatTensor(inputs),
            torch.LongTensor(outputs)),
        batch_size=batch_size, num_workers=0,
        shuffle=True, pin_memory=True, drop_last=True)

    optimizer = optimizers[optimizer_type](linear.parameters(), lr=learning_rate)
    criterion = maybe_cuda(nn.NLLLoss())
    adjuster = scheduler.StepLR(optimizer, epoch_step, gamma=gamma)

    start=time.time()
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

            prediction_distribution = lsoftmax(linear(input_var))

            loss = cast(torch.FloatTensor, 0) # type: torch.FloatTensor

            loss += criterion(prediction_distribution, output_var)

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
        yield (linear.state_dict(), total_loss / items_processed)
