from models.tactic_predictor import (NeuralPredictorState,
                                     TrainablePredictor,
                                     TacticContext, Prediction,
                                     save_checkpoints,
                                     optimize_checkpoints,
                                     predictKTactics,
                                     predictKTacticsWithLoss,
                                     predictKTacticsWithLoss_batch,
                                     add_nn_args)
from models.components import (Embedding, SimpleEmbedding)
from data import (Sentence, ListDataset, RawDataset,
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

from features import feature_functions

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
        vals = self._in_layer(features_batch)
        for i in range(self.num_layers - 1):
            vals = F.relu(vals)
            vals = self._layers[i](vals)
        result = self._softmax(self._out_layer(vals)).view(batch_size, -1)
        return result

class FeaturesPredictor(TrainablePredictor[FeaturesDataset,
                                           Embedding,
                                           NeuralPredictorState]):
    def __init__(self) \
        -> None:
        self._feature_functions = feature_functions
        self._criterion = maybe_cuda(nn.NLLLoss())
        self._lock = threading.Lock()
    def _predictDistributions(self, in_datas : List[TacticContext]) -> torch.FloatTensor:
        feature_values = [[feature_val
                           for feature in self._feature_functions
                           for feature_val in feature(in_data)]
                          for in_data in in_datas]
        return self._model(FloatTensor(feature_values))
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) -> None:
        super().add_args_to_parser(parser, default_values)
        add_nn_args(parser, default_values)
        parser.add_argument("--hidden-size", dest="hidden_size", type=int,
                            default=default_values.get("hidden-size", 128))
        parser.add_argument("--num-layers", dest="num_layers", type=int,
                            default=default_values.get("num-layers", 3))
    def _encode_data(self, data : RawDataset, arg_values : Namespace) \
        -> Tuple[FeaturesDataset, Embedding]:
        embedding = SimpleEmbedding()
        return (FeaturesDataset([
            FeaturesSample([feature_val for feature in
                            self._feature_functions
                            for feature_val in feature(TacticContext(prev_tactics,
                                                                     hypotheses,
                                                                     goal))],
                           embedding.encode_token(get_stem(tactic)))
            for prev_tactics, hypotheses, goal, tactic in
            self._preprocess_data(data, arg_values)]),
                embedding)
    def _optimize_model_to_disc(self,
                                encoded_data : FeaturesDataset,
                                embedding : Embedding,
                                arg_values : Namespace) \
        -> None:
        save_checkpoints(embedding, arg_values,
                         self._optimize_checkpoints(encoded_data, arg_values, embedding))
    def _optimize_checkpoints(self, encoded_data : FeaturesDataset, arg_values : Namespace,
                              embedding : Embedding) \
        -> Iterable[NeuralPredictorState]:
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
                         metadata : Embedding,
                         state : NeuralPredictorState) -> None:
        self._embedding = metadata
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
        return FeaturesClassifier(len(self._feature_functions), arg_values.hidden_size,
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
