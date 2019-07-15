#!/usr/bin/env python3.7

import signal
import argparse
from argparse import Namespace
import time
import sys
import threading
import math
import random
import multiprocessing
import functools
from dataclasses import dataclass
from itertools import chain

from models.encdecrnn_predictor import inputFromSentence
from models.components import Embedding
from tokenizer import Tokenizer, tokenizers, make_keyword_tokenizer_relevance
from data import get_text_data, filter_data, \
    encode_seq_classify_data, ScrapedTactic, Sentence, Dataset, TokenizedDataset
from util import *
from context_filter import get_context_filter
from serapi_instance import get_stem
from models.args import start_std_args, optimizers

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim import Optimizer
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
import torch.utils.data as data
import torch.cuda

from models.tactic_predictor import Prediction, NeuralClassifier, TacticContext
from typing import Dict, List, Union, Any, Tuple, NamedTuple, Iterable, cast, Callable, Optional, Sequence

class ECSample(NamedTuple):
    goal : Sentence
    tactic : int

@dataclass(init=True, repr=True)
class ECDataset(Dataset):
    data : List[ECSample]
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i : Any):
        return self.data[i]

class EncClassPredictor(NeuralClassifier[ECDataset, 'RNNClassifier']):
    def _predictDistributions(self, in_datas : List[TacticContext]) \
        -> torch.FloatTensor:
        assert self.training_args
        tokenized_goals = [self._tokenizer.toTokenList(in_data.goal)
                           for in_data in in_datas]
        input_list = [inputFromSentence(tokenized_goal, self.training_args.max_length)
                      for tokenized_goal in tokenized_goals]
        input_tensor = LongTensor(input_list).view(len(in_datas), -1)
        return self._model.run(input_tensor)

    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, corrects : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        assert self.training_args
        if len(in_data) == 0:
            return [], 0
        with self._lock:
            tokenized_goals = [self._tokenizer.toTokenList(goal)
                               for prev_tactics, hypotheses, goal in in_data]
            input_tensor = LongTensor([inputFromSentence(tokenized_goal,
                                                         self.training_args.max_length)
                                      for tokenized_goal in tokenized_goals])
            prediction_distributions = self._model.run(input_tensor,
                                                       batch_size=len(in_data))
            correct_stems = [get_stem(correct) for correct in corrects]
            output_var = maybe_cuda(Variable(
                torch.LongTensor([self._embedding.encode_token(correct_stem)
                                  if self._embedding.has_token(correct_stem)
                                  else 0
                                  for correct_stem in correct_stems])))
            loss = self._criterion(prediction_distributions, output_var).item()

            if k > self._embedding.num_tokens():
                k = self._embedding.num_tokens()

            certainties_and_idxs_list = [single_distribution.view(-1).topk(k)
                                         for single_distribution in
                                         list(prediction_distributions)]
            results = [[Prediction(self._embedding.decode_token(stem_idx.item()) + ".",
                                   math.exp(certainty.item()))
                        for certainty, stem_idx in zip(*certainties_and_idxs)]
                       for certainties_and_idxs in certainties_and_idxs_list]
        return results, loss

    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) -> None:
        super().add_args_to_parser(parser, default_values)
        parser.add_argument("--max-length", dest="max_length", type=int,
                            default=default_values.get("max-length", 100))
        parser.add_argument("--hidden-size", dest="hidden_size", type=int,
                            default=default_values.get("hidden-size", 128))
        parser.add_argument("--num-encoder-layers", dest="num_encoder_layers", type=int,
                            default=default_values.get("num-encoder-layers", 3))
    def _encode_tokenized_data(self, data : TokenizedDataset, arg_values : Namespace,
                               tokenizer : Tokenizer, embedding : Embedding) \
        -> ECDataset:
        return ECDataset([ECSample(goal, tactic) for prev_tactics, goal, tactic in
                          data])
    def _data_tensors(self, encoded_data : ECDataset,
                      arg_values : Namespace) \
        -> List[torch.Tensor]:
        in_stream = torch.LongTensor([inputFromSentence(datum.goal, arg_values.max_length)
                                      for datum in encoded_data])
        out_stream = torch.LongTensor([datum.tactic for datum in encoded_data])
        return [in_stream, out_stream]
    def _get_model(self, arg_values : Namespace,
                   tactic_vocab_size : int, term_vocab_size : int) \
        -> 'RNNClassifier':
        return RNNClassifier(term_vocab_size, arg_values.hidden_size, tactic_vocab_size,
                             arg_values.num_encoder_layers)
    def _getBatchPredictionLoss(self, data_batch : Sequence[torch.Tensor],
                                model : 'RNNClassifier') \
        -> torch.FloatTensor:
        input_batch, output_batch = cast(Tuple[torch.LongTensor, torch.LongTensor],
                                         data_batch)
        predictionDistribution = model.run(input_batch, batch_size=len(input_batch))
        output_var = maybe_cuda(Variable(output_batch))
        return self._criterion(predictionDistribution, output_var)
    def getOptions(self) -> List[Tuple[str, str]]:
        return list(vars(self.training_args).items()) + \
            [("training loss", self.training_loss),
             ("# epochs", self.num_epochs)]
    def _description(self) -> str:
       return "a classifier pytorch model for proverbot"

class RNNClassifier(nn.Module):
    def __init__(self, input_vocab_size : int, hidden_size : int, output_vocab_size: int,
                 num_encoder_layers : int, num_decoder_layers : int =1,
                 batch_size : int=1) -> None:
        super(RNNClassifier, self).__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = maybe_cuda(nn.Embedding(input_vocab_size, hidden_size))
        self.gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self.decoder_out = maybe_cuda(nn.Linear(hidden_size, output_vocab_size))
        self.softmax = maybe_cuda(nn.LogSoftmax(dim=1))

    def forward(self, input : torch.FloatTensor, hidden : torch.FloatTensor) \
        -> Tuple[torch.FloatTensor, torch.FloatTensor] :
        output = self.embedding(input).view(1, self.batch_size, -1)
        for i in range(self.num_encoder_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        return output[0], hidden

    def initHidden(self):
        return maybe_cuda(Variable(torch.zeros(1, self.batch_size, self.hidden_size)))

    def run(self, input : torch.LongTensor, batch_size : int=1):
        self.batch_size = batch_size
        in_var = maybe_cuda(Variable(input))
        hidden = self.initHidden()
        for i in range(in_var.size()[1]):
            output, hidden = self(in_var[:,i], hidden)
        decoded = self.decoder_out(output)
        return self.softmax(decoded).view(self.batch_size, -1)

def main(arg_list : List[str]) -> None:
    predictor = EncClassPredictor()
    predictor.train(arg_list)
