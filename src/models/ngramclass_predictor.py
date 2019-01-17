#!/usr/bin/env python3

import argparse
from argparse import Namespace
import time
import math
import threading
from typing import Dict, Any, List, Tuple, NamedTuple, Iterable, cast, Union, Sequence

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.optim.lr_scheduler as scheduler
import torch.utils.data as data

from models.tactic_predictor import NeuralPredictor, NeuralPredictorState, Prediction, TacticContext

from tokenizer import tokenizers
from data import get_text_data, Sentence, getNGramTokenbagVector, ScrapedTactic, \
    TokenizedDataset, encode_ngram_classify_input, Dataset, NGram, NGramSample, NGramDataset
from context_filter import get_context_filter
from util import *
from serapi_instance import get_stem
from models.args import start_std_args

from dataclasses import dataclass

class NGramClassifyPredictor(NeuralPredictor[NGramDataset, 'nn.Linear']):
    def __init__(self) -> None:
        self._criterion = maybe_cuda(nn.NLLLoss())
        self._lsoftmax = nn.LogSoftmax(1)
        self._lock = threading.Lock()
    def predictDistribution(self, in_data : TacticContext) \
        -> torch.FloatTensor:
        in_vec = Variable(FloatTensor(encode_ngram_classify_input(
            in_data.goal, self.training_args.num_grams, self._tokenizer)))\
                 .view(1, -1)
        return self._lsoftmax(self._model(in_vec))

    def predictKTactics(self, in_data : TacticContext, k : int) \
        -> List[Prediction]:
        with self._lock:
            distribution = self.predictDistribution(in_data)
            if k > self._embedding.num_tokens():
                k = self._embedding.num_tokens()
            probs_and_indices = distribution.squeeze().topk(k)
        return [Prediction(self._embedding.decode_token(idx.data[0]) + ".",
                           math.exp(certainty.data[0]))
                for certainty, idx in probs_and_indices]

    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int,
                                correct : str) -> Tuple[List[Prediction], float]:
        with self._lock:
            distribution = self.predictDistribution(in_data)
            stem = get_stem(correct)
            if self._embedding.has_token(stem):
                output_var = maybe_cuda(
                    Variable(torch. LongTensor([self._embedding.encode_token(stem)])))
                loss = self._criterion(distribution, output_var).item()
            else:
                loss = 0

            if k > self._embedding.num_tokens():
                k = self._embedding.num_tokens()
            probs_and_indices = distribution.squeeze().topk(k)
            predictions = [Prediction(self._embedding.decode_token(idx.item()) + ".",
                                      math.exp(certainty.item()))
                           for certainty, idx in zip(*probs_and_indices)]
        return predictions, loss

    def predictKTacticsWithLoss_batch(self, in_data : List[TacticContext],
                                      k : int, corrects : List[str]):
        with self._lock:
            input_tensor = Variable(FloatTensor([encode_ngram_classify_input(
                in_data_point.goal, self.training_args.num_grams, self._tokenizer)
                                                 for in_data_point in in_data]))
            prediction_distributions = self._lsoftmax(self._model(input_tensor))
            correct_stems = [get_stem(correct) for correct in corrects]
            output_var = maybe_cuda(Variable(torch.LongTensor(
                [self._embedding.encode_token(correct_stem)
                 if self._embedding.has_token(correct_stem)
                 else 0
                 for correct_stem in correct_stems])))
            loss = self._criterion(prediction_distributions, output_var).item()
            if k > self._embedding.num_tokens():
                k = self._embedding.num_tokens()

            certainties_and_idxs_list = \
                [single_distribution.view(-1).topk(k)
                 for single_distribution in list(prediction_distributions)]
            results = [[Prediction(self._embedding.decode_token(stem_idx.item()) + ".",
                                   math.exp(certainty.item()))
                        for certainty, stem_idx in zip(*certainties_and_idxs)]
                       for certainties_and_idxs in certainties_and_idxs_list]
        return results, loss
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) -> None:
        super().add_args_to_parser(parser, {"num-epochs": 50,
                                            "learning-rate": 0.0008})
        parser.add_argument("--num-grams", dest="num_grams", default=1, type=int)
    def _encode_tokenized_data(self, data : TokenizedDataset, arg_values : Namespace,
                               term_vocab_size : int, tactic_vocab_size : int) \
        -> NGramDataset:
        return NGramDataset([NGramSample(getNGramTokenbagVector(arg_values.num_grams,
                                                                term_vocab_size,
                                                                goal),
                                         tactic) for prev_tactic, goal, tactic in data])
    def _data_tensors(self, encoded_data : NGramDataset,
                      arg_values : Namespace) \
        -> List[torch.Tensor]:
        in_stream = torch.FloatTensor([datum.goal for datum in encoded_data])
        out_stream = torch.LongTensor([datum.tactic for datum in encoded_data])
        return [in_stream, out_stream]
    def _get_model(self, arg_values : Namespace,
                   tactic_vocab_size : int, term_vocab_size : int) \
        -> 'nn.Linear':
        return nn.Linear( term_vocab_size ** arg_values.num_grams, tactic_vocab_size)
    def _getBatchPredictionLoss(self, data_batch : Sequence[torch.Tensor],
                                model : 'nn.Linear') \
        -> torch.FloatTensor:
        input_batch, output_batch = cast(Tuple[torch.FloatTensor, torch.LongTensor],
                                         data_batch)
        input_var = maybe_cuda(Variable(input_batch))
        output_var = maybe_cuda(Variable(output_batch))
        prediction_distribution = self._lsoftmax(model(input_var))
        return self._criterion(prediction_distribution, output_var)
    def _description(self) -> str:
        return "A second-tier predictor which predicts tactic stems " \
            "based on word frequency in the goal"

def main(args_list : List[str]) -> None:
    predictor = NGramClassifyPredictor()
    predictor.train(args_list)
