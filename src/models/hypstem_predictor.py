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
import threading
import multiprocessing

from typing import (Dict, Any, List, Tuple, Iterable, cast, Union,
                    NamedTuple, Generic)
from argparse import Namespace
from difflib import SequenceMatcher

# Using sklearn for the actual learning
from sklearn import svm
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.tactic_predictor import (TrainablePredictor, Prediction,
                                     NeuralPredictorState,
                                     add_tokenizer_args, embed_data,
                                     tokenize_goals, NeuralPredictorState)
from models.components import (Embedding, StraightlineClassifierModel,
                               DNNClassifierModel, SVMClassifierModel,
                               PredictorState)
from tokenizer import tokenizers, Tokenizer

from data import (getNGramTokenbagVector, ListDataset, RawDataset)
from format import ScrapedTactic, TacticContext

from util import *
import serapi_instance

class HypStemSample(NamedTuple):
    hypotheses : List[int]
    relevance: float
    goal : List[int]
    tactic : int

class HypStemDataset(ListDataset[HypStemSample]):
    pass

ModelType = TypeVar("ModelType", bound=StraightlineClassifierModel)
StateType = TypeVar("StateType", bound=PredictorState)

class HypStemPredictor(TrainablePredictor[HypStemDataset, Tuple[Tokenizer, Embedding],
                                          StateType],
                       Generic[ModelType, StateType]):
    def __init__(self, modelclassObject) -> None:
        self._criterion = maybe_cuda(nn.NLLLoss())
        self._lock = threading.Lock()
        self._modelclassobject = modelclassObject

    def _encode_term(self, term : str) -> List[float]:
        assert self.training_args
        return cast(List[float],
                    getNGramTokenbagVector(self.training_args.num_grams,
                                           self._tokenizer.numTokens(),
                                           self._tokenizer.toTokenList(term)))
    def _predictDistribution(self, in_data : TacticContext) -> \
        Tuple[torch.FloatTensor, str]:
        if len(in_data.hypotheses) > 0:
            relevant_hyp, relevance = \
                max([(hyp,
                      term_relevance(in_data.goal,
                                           serapi_instance.get_hyp_type(hyp)))
                     for hyp in in_data.hypotheses], key=lambda x: x[1])
        else:
            relevant_hyp = ":"
            relevance = 0
        encoded_hyp = self._encode_term(serapi_instance.get_hyp_type(relevant_hyp))
        encoded_relevance = [relevance]
        encoded_goal = self._encode_term(in_data.goal)
        stem_distribution = self._run_model(encoded_hyp, encoded_relevance, encoded_goal)
        return FloatTensor(stem_distribution), \
            serapi_instance.get_first_var_in_hyp(relevant_hyp)
    def _encode_data(self, data : RawDataset, arg_values : Namespace) \
        -> Tuple[HypStemDataset, Tuple[Tokenizer, Embedding]]:
        embedding, embedded_data = embed_data(data)
        tokenizer, tokenized_goals = tokenize_goals(embedded_data, arg_values)
        print("Encoding hyps...")
        with multiprocessing.Pool(arg_values.num_threads) as pool:
            relevant_hyps, relevances = \
                zip(*list(pool.imap(most_relevant_hyp, data)))
        encoded_relevant_hyps = [getNGramTokenbagVector(arg_values.num_grams,
                                                        tokenizer.numTokens(),
                                                        tokenizer.toTokenList(hyp_term))
                                 for hyp_term in relevant_hyps]
        print("Encoding goals...")
        encoded_goals = [getNGramTokenbagVector(arg_values.num_grams,
                                                tokenizer.numTokens(),
                                                term) for term in tokenized_goals]
        print("Done")
        return HypStemDataset([HypStemSample(hyp, relevance, goal, inter.tactic)
                               for hyp, relevance, goal, inter in
                               zip(encoded_relevant_hyps, relevances, encoded_goals,
                                   embedded_data)]), (tokenizer, embedding)

    def predictKTactics(self, in_data : TacticContext, k : int) \
        -> List[Prediction]:
        with self._lock:
            distribution, hyp_var = self._predictDistribution(in_data)
        indices, probabilities = list_topk(list(distribution), k)
        predictions : List[Prediction] = []
        for certainty, idx in zip(probabilities, indices):
            stem = self._embedding.decode_token(idx)
            if stem == "apply" or stem == "exploit" or stem == "rewrite":
                predictions.append(Prediction(stem + " " + hyp_var + ".",
                                              math.exp(certainty)))
            else:
                predictions.append(Prediction(stem + ".", math.exp(certainty)))
        return predictions

    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int,
                                correct : str) -> Tuple[List[Prediction], float]:
        with self._lock:
            distribution, hyp_var = self._predictDistribution(in_data)
            correct_stem = serapi_instance.get_stem(correct)
            if self._embedding.has_token(correct_stem):
                loss = self._criterion(distribution.view(1, -1),
                                       Variable(LongTensor([self._embedding.encode_token(correct_stem)]))).item()
            else:
                loss = float("+inf")
        indices, probabilities = list_topk(list(distribution), k)
        predictions : List[Prediction] = []
        for certainty, idx in zip(probabilities, indices):
            stem = self._embedding.decode_token(idx)
            if serapi_instance.tacticTakesHypArgs(stem):
                predictions.append(Prediction(stem + " " + hyp_var + ".",
                                              math.exp(certainty)))
            else:
                predictions.append(Prediction(stem + ".", math.exp(certainty)))
        return predictions, loss
    def predictKTacticsWithLoss_batch(self,
                                      in_datas : List[TacticContext],
                                      k : int, corrects : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:

        results = [self.predictKTacticsWithLoss(in_data, k, correct)
                   for in_data, correct in zip(in_datas, corrects)]

        results2 : Tuple[List[List[Prediction]], List[float]] = tuple(zip(*results)) # type: ignore
        prediction_lists, losses = results2
        return prediction_lists, sum(losses)/len(losses)
    def load_saved_state(self,
                         args : Namespace,
                         unparsed_args : List[str],
                         metadata : Tuple[Tokenizer, Embedding],
                         state : StateType) -> None:
        self._tokenizer, self._embedding = metadata
        self._model = self._modelclassobject(args, self._tokenizer.numTokens() * 2 + 1,
                                             self._embedding.num_tokens())
        self._model.setState(state)
        self.training_args = args
        self.unparsed_args = args

    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) \
                           -> None:
        super().add_args_to_parser(parser, default_values)
        self._modelclassobject.add_args_to_parser(parser, default_values)
        add_tokenizer_args(parser)
        parser.add_argument("--num-grams", "-n", dest="num_grams", type=int, default=1)
        parser.add_argument("--kernel", choices=svm_kernels, type=str,
                            default=svm_kernels[0])
    def _optimize_model_to_disc(self,
                                encoded_data : HypStemDataset,
                                encdec_state : Tuple[Tokenizer, Embedding],
                                arg_values : Namespace) \
        -> None:
        hyps, rels, goals, outputs = \
            cast(Tuple[List[List[float]],
                       List[float],
                       List[List[float]],
                       List[int]],
                 zip(*encoded_data.data))
        inputs = [list(hyp) + [rel] + list(goal) for hyp, rel, goal in
                  zip(hyps, rels, goals)]
        # inputs = goals
        # inputs = hyps
        tokenizer, embedding = encdec_state
        self._model = self._modelclassobject(arg_values, tokenizer.numTokens() * 2 + 1,
                                             embedding.num_tokens())

        for checkpoint in self._model.checkpoints(inputs, outputs):
            with open(arg_values.save_file, 'wb') as f:
                torch.save((arg_values, encdec_state, checkpoint), f)

    def getOptions(self) -> List[Tuple[str, str]]:
        return list(vars(self.training_args).items())

    def _description(self) -> str:
        return "Predict tactic stems based on most similar hypothesis"

    def _run_model(self, hyp : List[float], rel : List[float], goal : List[float]) -> \
        torch.FloatTensor:
        # return FloatTensor(self._model.predict_log_proba([hyp])[0])
        # return FloatTensor(self._model.predict_log_proba([goal])[0])
        return FloatTensor(self._model.predict([list(hyp) + rel + list(goal)])[0])

def term_relevance(goal : str, term: str):
    return SequenceMatcher(None, term, goal).ratio() * len(term)
def most_relevant_hyp(inter : ScrapedTactic) -> Tuple[str, float]:
    goal, hyp_list = inter.goal, inter.hypotheses
    if len(hyp_list) == 0:
        return "", 0
    result = max([(hyp_term, term_relevance(goal, serapi_instance.get_hyp_type(hyp_term)))
                   for hyp_term in hyp_list], key=lambda x: x[1])
    return result


svm_kernels = [
    "rbf",
    "linear",
]

def main(args_list : List[str]) -> None:
    predictor = HypStemPredictor[DNNClassifierModel, NeuralPredictorState](DNNClassifierModel)
    predictor.train(args_list)
