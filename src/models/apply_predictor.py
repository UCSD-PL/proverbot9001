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
from models.tactic_predictor import \
    (NeuralPredictorState, TrainablePredictor,
     Prediction, save_checkpoints, optimize_checkpoints, embed_data,
     predictKTactics, predictKTacticsWithLoss,
     predictKTacticsWithLoss_batch, add_tokenizer_args,
     tokenize_goals, tokenize_hyps)
from models.components import (Embedding, SimpleEmbedding, DNNClassifier, add_nn_args)
from data import (Sentence, ListDataset, RawDataset,
                  normalizeSentenceLength, getNGramTokenbagVector,
                  TacticContext)
from tokenizer import Tokenizer
import serapi_instance
from format import ScrapedTactic, TacticContext, strip_scraped_output
from util import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from typing import (List, Any, Tuple, NamedTuple, Dict, Sequence,
                    cast, Optional)
from dataclasses import dataclass
import threading
import argparse
from argparse import Namespace
import multiprocessing
import functools

TermType = List[int]

class HypothesisRelevanceSample(NamedTuple):
    hypothesis : TermType
    goal : TermType
    isRelevant : bool

class ApplyDataset(ListDataset[HypothesisRelevanceSample]):
    pass

class ApplyPredictor(TrainablePredictor[ApplyDataset,
                                        Tokenizer,
                                        NeuralPredictorState]):
    def __init__(self) -> None:
        self._criterion = maybe_cuda(nn.NLLLoss())
        self._lock = threading.Lock()

    def _encode_term(self, term : str) -> List[int]:
        assert self.training_args
        return getNGramTokenbagVector(self.training_args.num_grams,
                                      self._tokenizer.numTokens(),
                                      self._tokenizer.toTokenList(term))

    def _predictDistribution(self, in_data : TacticContext) -> torch.FloatTensor:
        hyp_terms = [serapi_instance.get_hyp_type(hyp) for hyp in in_data.hypotheses]
        encoded_hyps = FloatTensor([self._encode_term(term) for term in hyp_terms])
        encoded_goals = FloatTensor(self._encode_term(in_data.goal)) \
            .view(1, -1).expand(len(in_data.hypotheses), -1)

        relevance_predictions = \
            self._model(torch.cat((encoded_hyps, encoded_goals), dim=1))
        return relevance_predictions[:,1]
    def predictKTactics(self, in_data : TacticContext, k : int) -> List[Prediction]:
        if len(in_data.hypotheses) == 0:
            return [Prediction("eauto", 0)]
        with self._lock:
            distribution = self._predictDistribution(in_data)
            if k > len(in_data.hypotheses):
                k = len(in_data.hypotheses)
            probs, indices = distribution.squeeze().topk(k)
            if k == 1:
                probs = FloatTensor([probs])
                indices = LongTensor([indices])
        return [Prediction("apply " +
                           serapi_instance.get_first_var_in_hyp(
                               in_data.hypotheses[idx.item()]) + ".",
                           math.exp(certainty.item()))
                for certainty, idx in zip(probs, indices)]
    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) \
        -> Tuple[List[Prediction], float]:
        return self.predictKTactics(in_data, k), 0.0

    def predictKTacticsWithLoss_batch(self, in_datas : List[TacticContext],
                                      k : int, corrects : List[str]) \
        -> Tuple[List[List[Prediction]], float] :
        predictions = [self.predictKTactics(in_data, k) for in_data in in_datas]
        return predictions, 0.0
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) -> None:
        super().add_args_to_parser(parser)
        add_tokenizer_args(parser)
        add_nn_args(parser, dict([('num-epochs', 50), ('hidden-size', 256),
                                  ('batch-size', 64)]
                                 + list(default_values.items())))
        parser.add_argument("--num-grams", dest="num_grams", default=1, type=int)
        parser.add_argument("--hidden-size", dest="hidden_size", type=int,
                            default=default_values.get("hidden-size", 128))
        parser.add_argument("--num-layers", dest="num_layers", type=int,
                            default=default_values.get("num-layers", 3))
    def _determine_relevance(self, inter : ScrapedTactic) -> List[bool]:
        stem, args_string  = serapi_instance.split_tactic(inter.tactic)
        args = args_string[:-1].split()
        return [any([var.strip() in args for var in
                     serapi_instance.get_var_term_in_hyp(hyp).split(",")])
                for hyp in inter.hypotheses]

    def _encode_data(self, data : RawDataset, arg_values : Namespace) \
        -> Tuple[ApplyDataset, Tokenizer]:
        isRelevants = [self._determine_relevance(inter) for inter in data]
        embedding, embedded_data = embed_data(data)
        tokenizer, tokenized_goals = tokenize_goals(embedded_data, arg_values)
        tokenized_hyp_lists = tokenize_hyps(data, arg_values, tokenizer)
        with multiprocessing.Pool(None) as pool:
            encoded_hyp_lists = list(pool.imap(functools.partial(encodeHypList,
                                                                 arg_values.num_grams,
                                                                 tokenizer.numTokens()),
                                               tokenized_hyp_lists))
            encoded_goals = list(pool.imap(functools.partial(getNGramTokenbagVector,
                                                             arg_values.num_grams,
                                                             tokenizer.numTokens()),
                                           tokenized_goals))
        samples = ApplyDataset([
            HypothesisRelevanceSample(encoded_hyp, encoded_goal, isRelevant)
            for encoded_goal, encoded_hyps_list, relevanceList
            in zip(encoded_goals, encoded_hyp_lists, isRelevants)
            for encoded_hyp, isRelevant in zip(encoded_hyps_list, relevanceList)])

        return samples, tokenizer

    def _optimize_model_to_disc(self,
                                encoded_data : ApplyDataset,
                                metadata : Tokenizer,
                                arg_values : Namespace) \
                                -> None:
        tokenizer = metadata
        save_checkpoints("apply",
                         metadata, arg_values,
                         self._optimize_checkpoints(encoded_data, arg_values,
                                                    tokenizer))

    def _optimize_checkpoints(self,
                              encoded_data : ApplyDataset,
                              arg_values : Namespace,
                              tokenizer : Tokenizer) \
                              -> Iterable[NeuralPredictorState]:
        tensors = self._data_tensors(encoded_data, arg_values)
        model = self._get_model(arg_values, tokenizer.numTokens())
        return optimize_checkpoints(tensors, arg_values, model,
                                    lambda batch_tensors, model:
                                    self._getBatchPredictionLoss(batch_tensors, model))
    def load_saved_state(self,
                         args : Namespace,
                         unparsed_args : List[str],
                         tokenizer : Tokenizer,
                         state : NeuralPredictorState) -> None:
        self._tokenizer = tokenizer
        self._model = maybe_cuda(self._get_model(args, tokenizer.numTokens()))
        self._model.load_state_dict(state.weights)
        self.training_loss = state.loss
        self.num_epochs = state.epoch
        self.training_args = args
        self.unparsed_args = unparsed_args
    def _data_tensors(self, encoded_data : ApplyDataset,
                      arg_values : Namespace) \
        -> List[torch.Tensor]:
        hypotheses, goals, relevance = zip(*encoded_data)
        hypothesesTensor = torch.FloatTensor(hypotheses)
        goalsTensor = torch.FloatTensor(goals)
        relevanceTensor = torch.LongTensor(relevance)
        tensors = [hypothesesTensor, goalsTensor, relevanceTensor]
        return tensors
    def _get_model(self, arg_values : Namespace, num_tokens : int) \
        -> DNNClassifier:
        return DNNClassifier(2 * (num_tokens ** arg_values.num_grams),
                             arg_values.hidden_size,
                             2,
                             arg_values.num_layers)
    def _getBatchPredictionLoss(self, data_batch : Sequence[torch.Tensor],
                                model : DNNClassifier) \
        -> torch.FloatTensor:
        hypotheses_batch, goals_batch, outputs_batch = \
            cast(Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor],
                 data_batch)
        predictionDistribution = model(torch.cat((hypotheses_batch, goals_batch), dim=1))
        output_var = maybe_cuda(Variable(outputs_batch))
        return self._criterion(predictionDistribution, output_var)
    def getOptions(self) -> List[Tuple[str, str]]:
        return list(vars(self.training_args).items()) + \
            [("training loss", self.training_loss),
             ("# epochs", self.num_epochs)]

    def _description(self) -> str:
        return "A predictor that tries to apply a hypothesis to the goal"

def encodeHypList(num_grams : int, num_tokens : int, hyps_list : List[List[int]]) -> \
    List[List[int]]:
    return [getNGramTokenbagVector(num_grams, num_tokens, hyp) for hyp in hyps_list]

def train_relevance(arg_list : List[str]) -> None:
    predictor = ApplyPredictor()
    predictor.train(arg_list)
