#!/usr/bin/env python3

from models.tactic_predictor import (NeuralClassifier, TacticContext,
                                     Prediction)
from models.components import Embedding
from data import (Sentence, Dataset, TokenizedDataset,
                  normalizeSentenceLength)
from serapi_instance import get_stem
from util import *
from tokenizer import Tokenizer

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from typing import (List, Any, Tuple, NamedTuple, Dict, Sequence,
                    cast)
from dataclasses import dataclass
import threading
import argparse
from argparse import Namespace

class PECSample(NamedTuple):
    prev_tactic : int
    goal : Sentence
    next_tactic : int

@dataclass(init=True, repr=True)
class PECDataset(Dataset):
    data : List[PECSample]
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i : Any):
        return self.data[i]

class PEClassifier(nn.Module):
    def __init__(self, goal_vocab_size : int, hidden_size : int,
                 tactic_vocab_size : int, num_encoder_layers : int) -> None:
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.hidden_size = hidden_size
        self.goal_embedding = maybe_cuda(nn.Embedding(goal_vocab_size, hidden_size))
        self.tactic_embedding = maybe_cuda(nn.Embedding(tactic_vocab_size, hidden_size))
        self.gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self.decoder_out = maybe_cuda(nn.Linear(hidden_size * 2, tactic_vocab_size))
        self.softmax = maybe_cuda(nn.LogSoftmax(dim=1))

    def forward(self, goal_token : torch.LongTensor, hidden :
                torch.FloatTensor) \
        -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        batch_size = goal_token.size()[0]
        output = self.goal_embedding(goal_token).view(1, batch_size, self.hidden_size)
        for i in range(self.num_encoder_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        return output[0], hidden

    def run(self,
            goal_tokens : torch.LongTensor,
            prev_tactic : torch.LongTensor):
        batch_size = goal_tokens.size()[0]
        assert prev_tactic.size()[0] == batch_size
        in_var = maybe_cuda(Variable(goal_tokens))
        hidden = maybe_cuda(Variable(torch.zeros(1, batch_size, self.hidden_size)))

        for i in range(in_var.size()[1]):
            goal_output, hidden = self(in_var[:,i], hidden)

        embedded_prev = self.tactic_embedding(maybe_cuda(prev_tactic))\
                            .view(batch_size, self.hidden_size)
        catted = torch.cat((goal_output, embedded_prev), dim=1)
        full_output = self.decoder_out(catted)

        return self.softmax(full_output).view(batch_size, -1)

class PECPredictor(NeuralClassifier[PECDataset, 'PEClassifier']):
    def _get_prev(self, in_data : TacticContext) -> int:
        stem = get_stem(in_data.prev_tactics[-1]) \
            if len(in_data.prev_tactics) > 1 else "Proof"
        if self._embedding.has_token(stem):
            return self._embedding.encode_token(stem)
        else:
            return self._embedding.encode_token("eauto")
    def _predictDistribution(self, in_data : TacticContext) \
        -> torch.FloatTensor:
        tokenized_goal = self._tokenizer.toTokenList(in_data.goal)
        goal_list = normalizeSentenceLength(tokenized_goal, self.training_args.max_length)
        goal_tensor = LongTensor(goal_list).view(1, -1)
        prev_tensor = LongTensor([self._get_prev(in_data)]).view(1, -1)
        return self._model.run(goal_tensor, prev_tensor)

    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, corrects : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        if len(in_data) == 0:
            return [], 0
        with self._lock:
            goals_tensor = LongTensor([
                normalizeSentenceLength(self._tokenizer.toTokenList(goal),
                                        self.training_args.max_length)
                for prev_tactics, hypotheses, goal in in_data])
            prevs_tensor = LongTensor([self._get_prev(in_datum) for in_datum in in_data])
            correct_stems = [get_stem(correct) for correct in corrects]
            prediction_distributions = self._model.run(goals_tensor, prevs_tensor)
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
        -> PECDataset:
        return PECDataset([PECSample(embedding.encode_token(
                get_stem(prev_tactics[-1]) if len(prev_tactics) > 1 else "Proof"),
                                     goal, tactic)
                           for prev_tactics, goal, tactic in data])
    def _data_tensors(self, encoded_data : PECDataset, arg_values : Namespace) \
        -> List[torch.Tensor]:
        prevs, goals, nexts = zip(*encoded_data)
        goal_stream = torch.LongTensor([
            normalizeSentenceLength(goal, arg_values.max_length)
            for goal in goals])
        prev_stream = torch.LongTensor(prevs)
        out_stream = torch.LongTensor(nexts)
        return [goal_stream, prev_stream, out_stream]

    def _get_model(self, arg_values : Namespace, tactic_vocab_size : int,
                   term_vocab_size : int) \
                   -> PEClassifier:
        return PEClassifier(term_vocab_size, arg_values.hidden_size, tactic_vocab_size,
                            arg_values.num_encoder_layers)

    def _getBatchPredictionLoss(self, data_batch : Sequence[torch.Tensor],
                                model : PEClassifier) \
                                -> torch.FloatTensor:
        goal_batch, prev_batch, output_batch = \
            cast(Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor], data_batch)
        predictionDistribution = model.run(goal_batch, prev_batch)
        output_var = maybe_cuda(Variable(output_batch))
        return self._criterion(predictionDistribution, output_var)

    def _description(self) -> str:
       return "another classifier pytorch model for proverbot"

def main(arg_list : List[str]) -> None:
    predictor = PECPredictor()
    predictor.train(arg_list)
