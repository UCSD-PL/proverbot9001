from models.tactic_predictor import \
    (NeuralPredictorState, TrainablePredictor, TacticContext,
     Prediction, save_checkpoints, optimize_checkpoints, embed_data,
     predictKTactics, predictKTacticsWithLoss,
     predictKTacticsWithLoss_batch, add_nn_args, add_tokenizer_args,
     strip_scraped_output, tokenize_goals, tokenize_hyps)
from models.components import (Embedding, SimpleEmbedding)
from data import (Sentence, ListDataset, RawDataset,
                  normalizeSentenceLength, getNGramTokenbagVector)
from tokenizer import Tokenizer
import serapi_instance
from format import ScrapedTactic
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

class RelevanceClassifier(nn.Module):
    def __init__(self,
                 term_vocab_size : int,
                 hidden_size : int,
                 num_layers : int) -> None:
        super().__init__()
        self.num_layers = num_layers

        self._in_layer = maybe_cuda(nn.Linear(term_vocab_size * 2, hidden_size))
        for i in range(num_layers - 1):
            self.add_module("_layer{}".format(i),
                            maybe_cuda(nn.Linear(hidden_size, hidden_size)))
        self._out_layer = maybe_cuda(nn.Linear(hidden_size, 2))
        self._softmax = maybe_cuda(nn.LogSoftmax(dim=1))

    def forward(self,
                hypotheses_batch : torch.FloatTensor,
                goals_batch : torch.FloatTensor) -> torch.FloatTensor:
        batch_size = hypotheses_batch.size()[0]
        hypotheses_var = maybe_cuda(Variable(hypotheses_batch))
        goals_var = maybe_cuda(Variable(goals_batch))

        vals = self._in_layer(torch.cat((hypotheses_var, goals_var), dim=1))
        for i in range(self.num_layers - 1):
            vals = F.relu(vals)
            vals = getattr(self, "_layer{}".format(i))(vals)
        vals = F.relu(vals)
        result = self._softmax(self._out_layer(vals)).view(batch_size, -1)
        return result

class StemClassifier(nn.Module):
    def __init__(self, term_vocab_size : int,
                 hidden_size : int, num_layers : int,
                 output_vocab_size : int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self._in_layer = maybe_cuda(nn.Linear(term_vocab_size * 2 + 1, hidden_size))
        for i in range(num_layers - 1):
            self.add_module("_layer{}".format(i),
                            maybe_cuda(nn.Linear(hidden_size, hidden_size)))
        self._out_layer = maybe_cuda(nn.Linear(hidden_size, output_vocab_size))
        self._softmax = maybe_cuda(nn.LogSoftmax(dim=1))

    def forward(self,
                relevances_batch : torch.FloatTensor,
                hypotheses_batch : torch.FloatTensor,
                goals_batch : torch.FloatTensor) -> torch.FloatTensor:
        batch_size = relevances_batch.size()[0]
        relevances_var = maybe_cuda(Variable(relevances_batch))
        hypotheses_var = maybe_cuda(Variable(hypotheses_batch))
        goals_var = maybe_cuda(Variable(goals_batch))

        vals = self._in_layer(torch.cat((relevances_var.view(batch_size, 1),
                                         hypotheses_var, goals_var),
                                        dim=1))
        for i in range(self.num_layers - 1):
            vals = F.relu(vals)
            vals = getattr(self, "_layer{}".format(i))(vals)
        vals = F.relu(vals)
        result = self._softmax(self._out_layer(vals)).view(batch_size, -1)
        return result


class ApplyPredictor(TrainablePredictor[ApplyDataset,
                                        Tokenizer,
                                        NeuralPredictorState]):
    def __init__(self) -> None:
        self._criterion = maybe_cuda(nn.NLLLoss())
        self._lock = threading.Lock()

    def _predictDistributions(self, in_datas : List[TacticContext]) -> torch.FloatTensor:
        return torch.cat([self._predictDistribution(in_data) for in_data in in_datas])

    def _predictDistribution(self, in_data : TacticContext) -> torch.FloatTensor:
        encoded_goals = FloatTensor(getNGramTokenbagVector(
            self.training_args.num_grams,
            self._tokenizer.numTokens(),
            self._tokenizer.toTokenList(in_data.goal))) \
            .view(1, -1).expand(len(in_data.hypotheses), -1)

        hyp_terms = [hyp.partition(":")[2].strip() for hyp in in_data.hypotheses]
        encoded_hyps = FloatTensor([getNGramTokenbagVector(
            self.training_args.num_grams,
            self._tokenizer.numTokens(),
            self._tokenizer.toTokenList(term))
                                    for term in hyp_terms])
        relevance_predictions = self._model(encoded_hyps, encoded_goals)
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
                probs = [probs]
                indices = [indices]
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
        add_nn_args(parser)
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
        preprocessed_data = list(self._preprocess_data(data, arg_values))
        isRelevants = [self._determine_relevance(inter) for inter in preprocessed_data]
        embedding, embedded_data = embed_data(RawDataset(preprocessed_data))
        tokenizer, tokenized_goals = tokenize_goals(embedded_data, arg_values)
        tokenized_hyp_lists = tokenize_hyps(RawDataset(preprocessed_data), arg_values,
                                            tokenizer)
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
        save_checkpoints(metadata, arg_values,
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
                         tokenizer : Tokenizer,
                         state : NeuralPredictorState) -> None:
        self._tokenizer = tokenizer
        self._model = maybe_cuda(self._get_model(args, tokenizer.numTokens()))
        self._model.load_state_dict(state.weights)
        self.training_loss = state.loss
        self.num_epochs = state.epoch
        self.training_args = args
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
        -> RelevanceClassifier:
        return RelevanceClassifier(num_tokens ** arg_values.num_grams,
                                   arg_values.hidden_size,
                                   arg_values.num_layers)
    def _getBatchPredictionLoss(self, data_batch : Sequence[torch.Tensor],
                                model : RelevanceClassifier) \
        -> torch.FloatTensor:
        hypotheses_batch, goals_batch, outputs_batch = \
            cast(Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor],
                 data_batch)
        predictionDistribution = model(hypotheses_batch, goals_batch)
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

def main(arg_list : List[str]) -> None:
    predictor = ApplyPredictor()
    predictor.train(arg_list)
