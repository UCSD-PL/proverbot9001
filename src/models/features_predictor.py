from models.tactic_predictor import \
    (NeuralPredictorState, TrainablePredictor, TacticContext,
     Prediction, save_checkpoints, optimize_checkpoints, embed_data,
     predictKTactics, predictKTacticsWithLoss,
     predictKTacticsWithLoss_batch, add_nn_args, strip_scraped_output)
from models.components import (Embedding, SimpleEmbedding)
from data import (Sentence, ListDataset, RawDataset,
                  normalizeSentenceLength)
from serapi_instance import get_stem
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

from features import feature_constructors, Feature

class FeaturesSample(NamedTuple):
    features : List[float]
    next_tactic : int

class FeaturesDataset(ListDataset[FeaturesSample]):
    pass

class FeaturesClassifier(nn.Module):
    def __init__(self,
                 num_features : int,
                 hidden_size : int, tactic_vocab_size : int,
                 num_layers : int) -> None:
        super().__init__()
        self.num_layers = num_layers

        self._in_layer = maybe_cuda(nn.Linear(num_features, hidden_size))
        self._layers = [maybe_cuda(nn.Linear(hidden_size, hidden_size))
                       for _ in range(num_layers - 1)]
        self._out_layer = maybe_cuda(nn.Linear(hidden_size, tactic_vocab_size))
        self._softmax = maybe_cuda(nn.LogSoftmax(dim=1))

    def forward(self, features_batch : torch.FloatTensor) -> torch.FloatTensor:
        batch_size = features_batch.size()[0]
        features_var = maybe_cuda(Variable(features_batch))
        vals = self._in_layer(features_var)
        for i in range(self.num_layers - 1):
            vals = F.relu(vals)
            vals = self._layers[i](vals)
        vals = F.relu(vals)
        result = self._softmax(self._out_layer(vals)).view(batch_size, -1)
        return result

class FeaturesPredictor(TrainablePredictor[FeaturesDataset,
                                           Tuple[Embedding, List[Feature]],
                                           NeuralPredictorState]):
    def __init__(self) -> None:
        self._feature_functions : Optional[List[Feature]] = None
        self._criterion = maybe_cuda(nn.NLLLoss())
        self._lock = threading.Lock()
    def _get_features(self, context : TacticContext) -> List[float]:
        return [feature_val for feature in self._feature_functions
                for feature_val in feature(context)]
    def _predictDistributions(self, in_datas : List[TacticContext]) -> torch.FloatTensor:
        assert self._feature_functions
        feature_values = [self._get_features(in_data) for in_data in in_datas]
        return self._model(FloatTensor(feature_values))
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) -> None:
        super().add_args_to_parser(parser, default_values)
        add_nn_args(parser, default_values)
        parser.add_argument("--hidden-size", dest="hidden_size", type=int,
                            default=default_values.get("hidden-size", 128))
        parser.add_argument("--num-layers", dest="num_layers", type=int,
                            default=default_values.get("num-layers", 3))
        parser.add_argument("--print-keywords", dest="print_keywords",
                            default=False, action='store_const', const=True)
        parser.add_argument("--num-head-keywords", dest="num_head_keywords", type=int,
                            default=default_values.get("num-head-keywords", 20))
    def _encode_data(self, data : RawDataset, arg_values : Namespace) \
        -> Tuple[FeaturesDataset, Tuple[Embedding, List[Feature]]]:
        preprocessed_data = list(self._preprocess_data(data, arg_values))
        stripped_data = [strip_scraped_output(dat) for dat in preprocessed_data]
        self._feature_functions = [feature_constructor(stripped_data, arg_values) for # type: ignore
                                   feature_constructor in feature_constructors]
        embedding, embedded_data = embed_data(RawDataset(preprocessed_data))
        return (FeaturesDataset([
            FeaturesSample(self._get_features(strip_scraped_output(scraped)),
                           scraped.tactic)
            for scraped in embedded_data]),
                (embedding, self._feature_functions))
    def _optimize_model_to_disc(self,
                                encoded_data : FeaturesDataset,
                                metadata : Tuple[Embedding, List[Feature]],
                                arg_values : Namespace) \
        -> None:
        save_checkpoints(metadata, arg_values,
                         self._optimize_checkpoints(encoded_data, arg_values, metadata))
    def _optimize_checkpoints(self, encoded_data : FeaturesDataset, arg_values : Namespace,
                              metadata : Tuple[Embedding, List[Feature]]) \
        -> Iterable[NeuralPredictorState]:
        embedding, features = metadata
        return optimize_checkpoints(self._data_tensors(encoded_data, arg_values),
                                    arg_values,
                                    self._get_model(arg_values, embedding.num_tokens()),
                                    lambda batch_tensors, model:
                                    self._getBatchPredictionLoss(batch_tensors, model))
        pass
    def _description(self) -> str:
        return "A predictor using only hand-engineered features"
    def load_saved_state(self,
                         args : Namespace,
                         metadata : Tuple[Embedding, List[Feature]],
                         state : NeuralPredictorState) -> None:
        self._embedding, self._feature_functions = metadata
        print("Loading predictor with head keywords: {}"
              .format(self._feature_functions[0].headKeywords))
        self._model = maybe_cuda(self._get_model(args, self._embedding.num_tokens()))
        self._model.load_state_dict(state.weights)
        self.training_loss = state.loss
        self.num_epochs = state.epoch
        self.training_args = args
    def _data_tensors(self, encoded_data : FeaturesDataset,
                      arg_values : Namespace) \
        -> List[torch.Tensor]:
        features, tactics = zip(*encoded_data)
        return [torch.FloatTensor(features), torch.LongTensor(tactics)]
    def _get_model(self, arg_values : Namespace, tactic_vocab_size : int) \
        -> FeaturesClassifier:
        assert self._feature_functions
        return FeaturesClassifier(sum([feature.feature_size()
                                       for feature in self._feature_functions]),
                                  arg_values.hidden_size,
                                  tactic_vocab_size, arg_values.num_layers)
    def _getBatchPredictionLoss(self, data_batch : Sequence[torch.Tensor],
                                model : FeaturesClassifier) \
        -> torch.FloatTensor:
        input_batch, output_batch = cast(Tuple[torch.FloatTensor, torch.LongTensor],
                                         data_batch)
        input_var = maybe_cuda(Variable(input_batch))
        output_var = maybe_cuda(Variable(output_batch))
        predictionDistribution = model(input_var)
        return self._criterion(predictionDistribution, output_var)
    def getOptions(self) -> List[Tuple[str, str]]:
        return list(vars(self.training_args).items()) + \
            [("training loss", self.training_loss),
             ("# epochs", self.num_epochs)]
    def predictKTactics(self, in_data : TacticContext, k : int) \
        -> List[Prediction]:
        with self._lock:
            return predictKTactics(self._predictDistributions([in_data])[0],
                                   self._embedding, k)
    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) -> \
        Tuple[List[Prediction], float]:
        with self._lock:
            return predictKTacticsWithLoss(self._predictDistributions([in_data])[0],
                                           self._embedding, k,
                                           correct, self._criterion)
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, correct : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        with self._lock:
            return predictKTacticsWithLoss_batch(self._predictDistributions(in_data),
                                                 self._embedding, k,
                                                 correct, self._criterion)

def main(arg_list : List[str]) -> None:
    predictor = FeaturesPredictor()
    predictor.train(arg_list)
