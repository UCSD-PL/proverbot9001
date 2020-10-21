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
    (TrainablePredictor, Prediction, optimize_checkpoints, embed_data,
     predictKTactics, predictKTacticsWithLoss,
     predictKTacticsWithLoss_batch)
from models.components import (Embedding, PredictorState)
from data import (Sentence, ListDataset, RawDataset,
                  normalizeSentenceLength)
from serapi_instance import get_stem
from util import *
from format import TacticContext, strip_scraped_output

from typing import (List, Any, Tuple, NamedTuple, Dict, Sequence,
                    cast, Optional)
from dataclasses import dataclass
import threading
import argparse
import pickle
import sys
from argparse import Namespace

from features import vec_feature_constructors, VecFeature
# Using sklearn for the actual learning
from sklearn import svm

# Using Torch to get nllloss
import torch
from torch import nn
from torch.autograd import Variable

class FeaturesSample(NamedTuple):
    features : List[float]
    next_tactic : int

@dataclass
class FeaturesSVMState(PredictorState):
    inner : svm.SVC

class FeaturesDataset(ListDataset[FeaturesSample]):
    pass
class FeaturesSVMPredictor(TrainablePredictor[FeaturesDataset,
                                              Tuple[Embedding, List[VecFeature]],
                                              FeaturesSVMState]):
    def __init__(self) \
        -> None:
        self._feature_functions : Optional[List[VecFeature]] = None
        self._criterion = maybe_cuda(nn.NLLLoss())
        self._lock = threading.Lock()
    def _predictDistributions(self, in_datas : List[TacticContext]) -> torch.FloatTensor:
        assert self._feature_functions
        feature_vectors = [self._get_features(in_data) for in_data in in_datas]
        distribution = self._model.predict_log_proba(feature_vectors)
        return FloatTensor(distribution)
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) -> None:
        super().add_args_to_parser(parser, default_values)
        parser.add_argument("--kernel", choices=svm_kernels, type=str,
                            default=svm_kernels[0])
        parser.add_argument("--print-keywords", dest="print_keywords",
                            default=False, action='store_const', const=True)
        parser.add_argument("--num-head-keywords", dest="num_head_keywords", type=int,
                            default=default_values.get("num-head-keywords", 20))
        parser.add_argument("--num-tactic-keywords", dest="num_tactic_keywords", type=int,
                            default=default_values.get("num-tactic-keywords", 50))
    def _get_features(self, context : TacticContext) -> List[float]:
        assert self._feature_functions
        return [feature_val for feature in self._feature_functions
                for feature_val in feature(context)]
    def _encode_data(self, data : RawDataset, arg_values : Namespace) \
        -> Tuple[FeaturesDataset, Tuple[Embedding, List[VecFeature]]]:
        stripped_data = [strip_scraped_output(dat) for dat in data]
        self._feature_functions = [feature_constructor(stripped_data, arg_values) for
                                   feature_constructor in vec_feature_constructors]
        embedding, embedded_data = embed_data(data)
        return (FeaturesDataset([
            FeaturesSample(self._get_features(strip_scraped_output(scraped)),
                           scraped.tactic)
            for scraped in embedded_data]),
                (embedding, self._feature_functions))
    def _optimize_model_to_disc(self,
                                encoded_data : FeaturesDataset,
                                metadata : Tuple[Embedding, List[VecFeature]],
                                arg_values : Namespace) \
        -> None:
        curtime = time.time()
        print("Training SVM...", end="")
        sys.stdout.flush()
        model = svm.SVC(gamma='scale', kernel=arg_values.kernel, probability=True)
        inputs, outputs = cast(Tuple[List[List[float]], List[int]], zip(*encoded_data))
        model.fit(inputs, outputs)
        print(" {:.2f}s".format(time.time() - curtime))
        loss = model.score(inputs, outputs)
        print("Training loss: {}".format(loss))
        with open(arg_values.save_file, 'wb') as f:
            torch.save(("featuressvm", (arg_values, metadata, FeaturesSVMState(1, model))), f)
    def _description(self) -> str:
        return "An svm predictor using only hand-engineered features"
    def load_saved_state(self,
                         args : Namespace,
                         unparsed_args : List[str],
                         metadata : Tuple[Embedding, List[VecFeature]],
                         state : FeaturesSVMState) -> None:
        self._embedding, self._feature_functions = metadata
        self._model = state.inner
        self.training_args = args
        self.unparsed_args = unparsed_args
    def getOptions(self) -> List[Tuple[str, str]]:
        return list(vars(self.training_args).items())
    def predictKTactics(self, in_data : TacticContext, k : int) \
        -> List[Prediction]:
        with self._lock:
            return predictKTactics(self._predictDistributions([in_data])[0],
                                   self._embedding, k)
    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) -> \
        Tuple[List[Prediction], float]:
        with self._lock:
            return predictKTacticsWithLoss(torch.FloatTensor(self._predictDistributions([in_data])[0]),
                                           self._embedding, k,
                                           correct, self._criterion)
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, correct : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        with self._lock:
            return predictKTacticsWithLoss_batch(FloatTensor(self._predictDistributions(in_data)),
                                                 self._embedding, k,
                                                 correct, self._criterion)
svm_kernels = [
    "rbf",
    "linear",
]

def main(arg_list : List[str]) -> None:
    predictor = FeaturesSVMPredictor()
    predictor.train(arg_list)
