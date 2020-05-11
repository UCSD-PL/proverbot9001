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
     predictKTacticsWithLoss_batch)
from models.components import (Embedding, SimpleEmbedding, add_nn_args)
from data import (Sentence, ListDataset, RawDataset,
                  normalizeSentenceLength)
from serapi_instance import get_stem
from util import *
from format import TacticContext, strip_scraped_output

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

from features import (vec_feature_constructors,
                      word_feature_constructors, VecFeature,
                      WordFeature)

class FeaturesSample(NamedTuple):
    vec_features : List[float]
    word_features : List[int]
    next_tactic : int

class FeaturesDataset(ListDataset[FeaturesSample]):
    pass

class FeaturesClassifier(nn.Module):
    def __init__(self,
                 vec_features_size : int,
                 word_feature_vocab_sizes : List[int],
                 hidden_size : int,
                 word_embedding_size : int,
                 tactic_vocab_size : int,
                 num_layers : int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.word_embedding_size = word_embedding_size
        self.num_word_features = len(word_feature_vocab_sizes)

        self._in_layer = maybe_cuda(nn.Linear(
            vec_features_size +
            len(word_feature_vocab_sizes) * word_embedding_size,
            hidden_size))
        for i, vocab_size in enumerate(word_feature_vocab_sizes):
            self.add_module("_word_embedding{}".format(i),
                            maybe_cuda(nn.Embedding(vocab_size, word_embedding_size)))
        for i in range(num_layers - 1):
            self.add_module("_layer{}".format(i),
                            maybe_cuda(nn.Linear(hidden_size, hidden_size)))

        self._out_layer = maybe_cuda(nn.Linear(hidden_size, tactic_vocab_size))
        self._softmax = maybe_cuda(nn.LogSoftmax(dim=1))

    def forward(self, vec_features_batch : torch.FloatTensor,
                word_features_batch : torch.LongTensor) -> torch.FloatTensor:
        batch_size = vec_features_batch.size()[0]
        vec_features_var = maybe_cuda(Variable(vec_features_batch))
        word_embedded_features = []
        for i in range(self.num_word_features):
            word_feature_var = maybe_cuda(Variable(word_features_batch[:,i]))
            embedded = getattr(self, "_word_embedding{}".format(i))(word_feature_var)\
                .view(batch_size, self.word_embedding_size)
            word_embedded_features.append(embedded)
        word_embedded_features_vec = \
            torch.cat(word_embedded_features, dim=1)

        vals = self._in_layer(torch.cat((vec_features_var, word_embedded_features_vec),
                                        dim=1))
        for i in range(self.num_layers - 1):
            vals = F.relu(vals)
            vals = getattr(self, "_layer{}".format(i))(vals)
        vals = F.relu(vals)
        result = self._softmax(self._out_layer(vals)).view(batch_size, -1)
        return result

class FeaturesPredictor(TrainablePredictor[FeaturesDataset,
                                           Tuple[Embedding,
                                                 List[VecFeature], List[WordFeature]],
                                           NeuralPredictorState]):
    def __init__(self) -> None:
        self._vec_feature_functions : Optional[List[VecFeature]] = None
        self._word_feature_functions : Optional[List[WordFeature]] = None
        self._criterion = maybe_cuda(nn.NLLLoss())
        self._lock = threading.Lock()
    def _get_vec_features(self, context : TacticContext) -> List[float]:
        assert self._vec_feature_functions
        return [feature_val for feature in self._vec_feature_functions
                for feature_val in feature(context)]
    def _get_word_features(self, context : TacticContext) -> List[int]:
        assert self._word_feature_functions
        return [feature(context) for feature in self._word_feature_functions]
    def _predictDistributions(self, in_datas : List[TacticContext]) -> torch.FloatTensor:
        vec_feature_values = [self._get_vec_features(in_data) for in_data in in_datas]
        word_feature_values = [self._get_word_features(in_data) for in_data in in_datas]
        return self._model(FloatTensor(vec_feature_values),
                           LongTensor(word_feature_values))
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) -> None:
        super().add_args_to_parser(parser, default_values)
        add_nn_args(parser, dict([('num-epochs', 50)] + list(default_values.items())))
        parser.add_argument("--print-keywords", dest="print_keywords",
                            default=False, action='store_const', const=True)
        parser.add_argument("--num-head-keywords", dest="num_head_keywords", type=int,
                            default=default_values.get("num-head-keywords", 100))
        parser.add_argument("--num-tactic-keywords", dest="num_tactic_keywords", type=int,
                            default=default_values.get("num-tactic-keywords", 50))
        parser.add_argument("--word-embedding-size", dest="word_embedding_size", type=int,
                            default=default_values.get("word_embedding_size", 10))
    def _encode_data(self, data : RawDataset, arg_values : Namespace) \
        -> Tuple[FeaturesDataset, Tuple[Embedding, List[VecFeature], List[WordFeature]]]:
        stripped_data = [strip_scraped_output(dat) for dat in data]
        self._vec_feature_functions = [feature_constructor(stripped_data, arg_values) for # type: ignore
                                       feature_constructor in vec_feature_constructors]
        self._word_feature_functions = [feature_constructor(stripped_data, arg_values) for # type: ignore
                                        feature_constructor in word_feature_constructors]
        embedding, embedded_data = embed_data(data)
        return (FeaturesDataset([
            FeaturesSample(self._get_vec_features(strip_scraped_output(scraped)),
                           self._get_word_features(strip_scraped_output(scraped)),
                           scraped.tactic)
            for scraped in embedded_data]),
                (embedding, self._vec_feature_functions, self._word_feature_functions))
    def _optimize_model_to_disc(self,
                                encoded_data : FeaturesDataset,
                                metadata : Tuple[Embedding,
                                                 List[VecFeature], List[WordFeature]],
                                arg_values : Namespace) \
        -> None:
        save_checkpoints("features", metadata, arg_values,
                         self._optimize_checkpoints(encoded_data, arg_values, metadata))
    def _optimize_checkpoints(self, encoded_data : FeaturesDataset, arg_values : Namespace,
                              metadata : Tuple[Embedding, List[VecFeature], List[WordFeature]]) \
        -> Iterable[NeuralPredictorState]:
        embedding, vec_features, word_features = metadata
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
                         unparsed_args : List[str],
                         metadata : Tuple[Embedding, List[VecFeature], List[WordFeature]],
                         state : NeuralPredictorState) -> None:
        self._embedding, self._vec_feature_functions, self._word_feature_functions = metadata
        self._model = maybe_cuda(self._get_model(args, self._embedding.num_tokens()))
        self._model.load_state_dict(state.weights)
        self.training_loss = state.loss
        self.num_epochs = state.epoch
        self.training_args = args
        self.unparsed_args = unparsed_args
    def _data_tensors(self, encoded_data : FeaturesDataset,
                      arg_values : Namespace) \
        -> List[torch.Tensor]:
        vec_features, word_features, tactics = zip(*encoded_data)
        return [torch.FloatTensor(vec_features), torch.LongTensor(word_features),
                torch.LongTensor(tactics)]
    def _get_model(self, arg_values : Namespace, tactic_vocab_size : int) \
        -> FeaturesClassifier:
        assert self._vec_feature_functions
        assert self._word_feature_functions
        feature_vec_size = sum([feature.feature_size()
                                for feature in self._vec_feature_functions])
        word_feature_vocab_sizes = [feature.vocab_size()
                                    for feature in self._word_feature_functions]
        return FeaturesClassifier(feature_vec_size,
                                  word_feature_vocab_sizes,
                                  arg_values.hidden_size,
                                  arg_values.word_embedding_size,
                                  tactic_vocab_size, arg_values.num_layers)
    def _getBatchPredictionLoss(self, data_batch : Sequence[torch.Tensor],
                                model : FeaturesClassifier) \
        -> torch.FloatTensor:
        vec_features_batch, word_features_batch, output_batch = \
            cast(Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor],
                 data_batch)
        vec_features_var = maybe_cuda(Variable(vec_features_batch))
        word_features_var = maybe_cuda(Variable(word_features_batch))
        output_var = maybe_cuda(Variable(output_batch))
        predictionDistribution = model(vec_features_var, word_features_var)
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
