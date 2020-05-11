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
import pickle
import sys
import threading

from typing import Dict, Any, List, Tuple, Iterable, cast, Union
from argparse import Namespace

# Using sklearn for the actual learning
from sklearn import svm

# Using Torch to get nllloss
import torch
from torch import nn
from torch.autograd import Variable

from models.tactic_predictor import (TokenizingPredictor, Prediction,
                                     TokenizerEmbeddingState)
from models.components import Embedding, PredictorState
from tokenizer import tokenizers, Tokenizer
from data import get_text_data, getNGramTokenbagVector, encode_ngram_classify_data, \
    encode_ngram_classify_input, TokenizedDataset, Dataset, NGram, NGramSample, \
    NGramDataset
from util import *
from format import TacticContext
from serapi_instance import get_stem
from dataclasses import dataclass

@dataclass
class NGramSVMClassifierState(PredictorState):
    inner : svm.SVC
#     x : int
#     y : int

class NGramSVMClassifier(TokenizingPredictor[NGramDataset, NGramSVMClassifierState]):
    def load_saved_state(self,
                         args : Namespace,
                         unpasrsed_args : List[str],
                         metadata : TokenizerEmbeddingState,
                         state : NGramSVMClassifierState) -> None:
        self._tokenizer = metadata.tokenizer
        self._embedding = metadata.embedding
        self.training_args = args
        self.context_filter = args.context_filter
        self._model = state.inner
        pass
    def getOptions(self) -> List[Tuple[str, str]]:
        return list(vars(self.training_args).items())
    def _encode_term(self, term : str) -> List[int]:
        assert self.training_args
        return getNGramTokenbagVector(self.training_args.num_grams,
                                      self._tokenizer.numTokens(),
                                      self._tokenizer.toTokenList(term))

    def __init__(self) -> None:
        self._criterion = nn.NLLLoss()
        self._lock = threading.Lock()

    def _encode_tokenized_data(self, data : TokenizedDataset, arg_values : Namespace,
                               tokenizer : Tokenizer, embedding : Embedding) \
        -> NGramDataset:
        return NGramDataset([NGramSample(getNGramTokenbagVector(arg_values.num_grams,
                                                                tokenizer.numTokens(),
                                                                goal),
                                         tactic) for prev_tactic, hyps, goal, tactic
                             in data])

    def predictDistribution(self, in_data : TacticContext) \
        -> torch.FloatTensor:
        feature_vector = cast(List[float], self._encode_term(in_data.goal))
        distribution = FloatTensor(self._model.predict_log_proba([feature_vector])[0])
        return distribution

    def predictKTactics(self, in_data : TacticContext, k : int) \
        -> List[Prediction]:
        with self._lock:
            distribution = self.predictDistribution(in_data)
            indices, probabilities = list_topk(list(distribution), k)
        return [Prediction(self._embedding.decode_token(idx) + ".",
                           math.exp(certainty))
                for certainty, idx in zip(probabilities, indices)]

    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int,
                                correct : str) -> Tuple[List[Prediction], float]:
        with self._lock:
            distribution = self.predictDistribution(in_data)
            correct_stem = get_stem(correct)
            if self._embedding.has_token(correct_stem):
                loss = self._criterion(FloatTensor(distribution).view(1, -1), Variable(LongTensor([self._embedding.encode_token(correct_stem)]))).item()
            else:
                loss = float("+inf")
            indices, probabilities = list_topk(list(distribution), k)
            predictions = [Prediction(self._embedding.decode_token(idx) + ".",
                                      math.exp(certainty))
                           for certainty, idx in zip(probabilities, indices)]
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
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) \
                           -> None:
        super().add_args_to_parser(parser, default_values)
        parser.add_argument("--num-grams", "-n", dest="num_grams", type=int, default=1)
        parser.add_argument("--kernel", choices=svm_kernels, type=str,
                            default=svm_kernels[0])
    def _optimize_model_to_disc(self,
                                encoded_data : NGramDataset,
                                encdec_state : TokenizerEmbeddingState,
                                arg_values : Namespace) \
        -> None:
        curtime = time.time()
        print("Training SVM...", end="")
        sys.stdout.flush()
        model = svm.SVC(gamma='scale', kernel=arg_values.kernel, probability=True)
        inputs, outputs = cast(Tuple[List[List[float]], List[int]],
                               zip(*encoded_data.data))
        model.fit(inputs, outputs)
        print(" {:.2f}s".format(time.time() - curtime))
        loss = model.score(inputs, outputs)
        print("Training loss: {}".format(loss))
        with open(arg_values.save_file, 'wb') as f:
            torch.save((arg_values, encdec_state, NGramSVMClassifierState(1, model)), f)

    def _description(self) -> str:
        return "A simple predictor which tries the k most common tactic stems."

svm_kernels = [
    "rbf",
    "linear",
]

def main(args_list : List[str]) -> None:
    predictor = NGramSVMClassifier()
    predictor.train(args_list)
