from models.tactic_predictor import (TrainablePredictor,
                                     TacticContext, Prediction,
                                     save_checkpoints,
                                     optimize_checkpoints,
                                     embed_data,
                                     predictKTactics,
                                     predictKTacticsWithLoss,
                                     predictKTacticsWithLoss_batch)
from models.components import (Embedding)
from data import (Sentence, ListDataset, RawDataset,
                  normalizeSentenceLength)
from serapi_instance import get_stem
from util import *

from typing import (List, Any, Tuple, NamedTuple, Dict, Sequence,
                    cast)
from dataclasses import dataclass
import threading
import argparse
import pickle
import sys
from argparse import Namespace

from features import feature_functions
# Using sklearn for the actual learning
from sklearn import svm

# Using Torch to get nllloss
import torch
from torch import nn
from torch.autograd import Variable

class FeaturesSample(NamedTuple):
    features : List[float]
    next_tactic : int

class FeaturesDataset(ListDataset[FeaturesSample]):
    pass
class FeaturesSVMPredictor(TrainablePredictor[FeaturesDataset, Embedding, svm.SVC]):
    def __init__(self) \
        -> None:
        self._feature_functions = feature_functions
        self._criterion = maybe_cuda(nn.NLLLoss())
        self._lock = threading.Lock()
    def _predictDistributions(self, in_datas : List[TacticContext]) -> torch.FloatTensor:
        feature_vectors = [self._get_features(in_data) for in_data in in_datas]
        distribution = self._model.predict_log_proba(feature_vectors)
        return distribution
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) -> None:
        super().add_args_to_parser(parser, default_values)
        parser.add_argument("--kernel", choices=svm_kernels, type=str,
                            default=svm_kernels[0])
    def _get_features(self, context : TacticContext) -> List[float]:
        return [feature_val for feature in self._feature_functions
                for feature_val in feature(context)]
    def _encode_data(self, data : RawDataset, arg_values : Namespace) \
        -> Tuple[FeaturesDataset, Embedding]:
        preprocessed_data = self._preprocess_data(data, arg_values)
        embedding, embedded_data = embed_data(RawDataset(list(preprocessed_data)))
        return (FeaturesDataset([
            FeaturesSample([feature_val for feature in
                            self._feature_functions
                            for feature_val in feature(TacticContext(prev_tactics,
                                                                     hypotheses,
                                                                     goal))],
                           tactic)
            for prev_tactics, hypotheses, goal, tactic in
            embedded_data]),
                embedding)
    def _optimize_model_to_disc(self,
                                encoded_data : FeaturesDataset,
                                embedding : Embedding,
                                arg_values : Namespace) \
        -> None:
        curtime = time.time()
        print("Training SVM...", end="")
        sys.stdout.flush()
        model = svm.SVC(gamma='scale', kernel=arg_values.kernel, probability=True)
        inputs, outputs = zip(*encoded_data)
        model.fit(inputs, outputs)
        print(" {:.2f}s".format(time.time() - curtime))
        loss = model.score(inputs, outputs)
        print("Training loss: {}".format(loss))
        with open(arg_values.save_file, 'wb') as f:
            torch.save((arg_values, embedding, model), f)
    def _description(self) -> str:
        return "An svm predictor using only hand-engineered features"
    def load_saved_state(self,
                         args : Namespace,
                         metadata : Embedding,
                         state : svm.SVC) -> None:
        self._embedding = metadata
        self._model = state
        self.training_args = args
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
