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

from format import TacticContext
from data import ListDataset, chunks, StateScore

from features import (WordFeature, VecFeature,
                      word_feature_constructors, vec_feature_constructors)

from models.components import NeuralPredictorState, add_nn_args, WordFeaturesEncoder, DNNScorer
from models.state_evaluator import TrainableEvaluator, StateEvaluationDataset
from models.tactic_predictor import optimize_checkpoints
from util import maybe_cuda, eprint, print_time

from typing import (List, Tuple, Iterable, Sequence, Dict, Any, cast)
import argparse
import torch
import functools
import multiprocessing
import sys
from torch import nn

FeaturesDNNEvaluatorState = Tuple[List[WordFeature], List[VecFeature], NeuralPredictorState]

class FeaturesDNNEvaluatorModel(nn.Module):
    def __init__(self,
                 word_features : List[WordFeature],
                 vec_features : List[VecFeature],
                 hidden_size : int,
                 num_layers : int) -> None:
        super().__init__()
        feature_vec_size = sum([feature.feature_size()
                                for feature in vec_features])
        word_features_vocab_sizes = [features.vocab_size()
                                     for features in word_features]
        self._word_features_encoder = maybe_cuda(
            WordFeaturesEncoder(word_features_vocab_sizes,
                                hidden_size, 1, hidden_size))
        self._features_classifier = maybe_cuda(
            DNNScorer(hidden_size + feature_vec_size,
                      hidden_size, num_layers))
    def forward(self,
                word_features_batch : torch.LongTensor,
                vec_features_batch : torch.FloatTensor) -> torch.FloatTensor:
        batch_size = word_features_batch.size()[0]
        encoded_word_features = self._word_features_encoder(
            maybe_cuda(word_features_batch))
        scores = self._features_classifier(
            torch.cat((encoded_word_features, maybe_cuda(vec_features_batch)), dim=1))
        return scores.view(batch_size)

class FeaturesDNNEvaluator(TrainableEvaluator[Tuple[List[WordFeature], List[VecFeature]]]):
    def __init__(self) -> None:
        self._criterion = maybe_cuda(nn.MSELoss())
    def _add_args_to_parser(self, parser : argparse.ArgumentParser,
                            default_values : Dict[str, Any] = {}) -> None:
        super()._add_args_to_parser(parser, default_values)
        add_nn_args(parser, default_values)
        feature_set : Set[str] = set()
        all_constructors : List[Type[Feature]] = vec_feature_constructors + word_feature_constructors # type: ignore
        for feature_constructor in all_constructors:
            new_args = feature_constructor\
                .add_feature_arguments(parser, feature_set, default_values)
            feature_set = feature_set.union(new_args)

    def description(self) -> str:
        return "A state evaluator that uses the standard feature set on DNN's"

    def shortname(self) -> str:
        return "features dnn evaluator"

    def _optimize_model(self, data : StateEvaluationDataset,
                       arg_values : argparse.Namespace) -> Iterable[FeaturesDNNEvaluatorState]:
        with print_time("Initializing features", guard=arg_values.verbose):
            if arg_values.start_from:
                _, (_, word_feature_funcs, vec_feature_funcs, _) = \
                    torch.load(arg_values.start_from)
            else:
                stripped_data = [dat.state for dat in data]
                word_feature_funcs = [feature_constructor(stripped_data, arg_values)
                                      for feature_constructor in word_feature_constructors]
                vec_feature_funcs = [feature_constructor(stripped_data, arg_values)
                                     for feature_constructor in vec_feature_constructors]

        with print_time("Converting data to tensors", guard=arg_values.verbose):
            tensors = self._data_tensors(data, arg_values,
                                         word_feature_funcs, vec_feature_funcs)

        with print_Time("Building the model", guard=arg_values.verbose):
            model = self._get_model(arg_values, word_feature_funcs, vec_feature_funcs)

        return ((word_feature_funcs, vec_feature_funcs, state)
                for state in optimize_checkpoints(tensors, arg_values, model,
                                                  lambda batch_tensors, model:
                                                  self._get_batch_prediction_loss(arg_values,
                                                                                  batch_tensors,
                                                                                  model)))
    def load_saved_state(self,
                         args : argparse.Namespace,
                         state : FeaturesDNNEvaluatorState) -> None:
        word_feature_functions, vec_feature_functions, \
            neural_state = state
        self._model = maybe_cuda(self._get_model(args, word_feature_functions, vec_feature_functions))
        self._model.load_state_dict(state.weights)

        self.training_loss = state.loss
        self.num_epochs = state.epoch
        self.training_args = args

    def _data_tensors(self, data : StateEvaluationDataset, arg_values : argparse.Namespace,
                      word_features : List[WordFeature], vec_features : List[VecFeature]) \
                      -> List[torch.Tensor]:
        with multiprocessing.Pool(arg_values.num_threads) as pool:
            processed_chunks : List[Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]] = \
                pool.map(functools.partial(data_to_tensor,
                                           word_features,
                                           vec_features),
                         chunks(data, 5000))
        word_feature_chunks, vec_feature_chunks, output_chunks = zip(*processed_chunks)

        word_features = torch.cat(word_feature_chunks, 0)
        vec_features = torch.cat(vec_feature_chunks, 0)
        outputs = torch.cat(output_chunks, 0)
        return [word_features, vec_features, outputs]

    def _get_model(self, arg_values : argparse.Namespace,
                   word_feature_funcs : List[WordFeature],
                   vec_feature_funcs : List[VecFeature]) -> FeaturesDNNEvaluatorModel:
        word_feature_vocab_sizes = [feature.vocab_size()
                                    for feature in word_feature_funcs]
        feature_vec_size = sum([feature.feature_size()
                                for feature in vec_feature_funcs])
        return FeaturesDNNEvaluatorModel(word_feature_funcs, vec_feature_funcs,
                                         arg_values.hidden_size, arg_values.num_layers)

    def _get_batch_prediction_loss(self, arg_values : argparse.Namespace,
                                   data_batch : Sequence[torch.Tensor],
                                   model : FeaturesDNNEvaluatorModel) -> torch.FloatTensor:
        word_features_batch, vec_features_batch, outputs = \
            cast(Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor],
                 data_batch)
        predicted_scores = model(word_features_batch, vec_features_batch)
        return self._criterion(predicted_scores, maybe_cuda(outputs))

    def scoreState(self, state : TacticContext) -> float:
        word_features = torch.LongTensor([[word_feature(state) for word_feature in word_features]])
        vec_features = torch.FloatTensor([[feature_val for vec_feature in vec_features
                                           for feature_val in vec_features(state)]])
        return model(word_features_batch, vec_features_batch)[0][0]

def data_to_tensor(word_features : List[WordFeature], vec_features : List[VecFeature],
                   data : List[StateScore]) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
    assert isinstance(data, list), data
    assert isinstance(data[0], StateScore)
    word_features = torch.LongTensor([[word_feature(x.state) for word_feature in word_features]
                                      for x in data])
    vec_features = torch.FloatTensor([[feature_val for vec_feature in vec_features
                                       for feature_val in vec_feature(x.state)]
                                      for x in data])
    outputs = torch.FloatTensor([x.score for x in data])
    return (word_features, vec_features, outputs)

def main(arg_list : List[str]) -> None:
    predictor = FeaturesDNNEvaluator()
    predictor.train(arg_list)
