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
from dataloader import (sample_context_features,
                        features_to_total_distances_tensors,
                        features_to_total_distances_tensors_with_map,
                        tmap_to_picklable, tmap_from_picklable, features_vocab_sizes)
from dataloader import TokenMap as FeaturesTokenMap
from util import maybe_cuda, eprint, print_time

from typing import (List, Tuple, Iterable, Sequence, Dict, Any, cast)
import argparse
import torch
import functools
import multiprocessing
import sys
from torch import nn

FeaturesDNNEvaluatorState = Tuple[FeaturesTokenMap, NeuralPredictorState]

class FeaturesDNNEvaluatorModel(nn.Module):
    def __init__(self,
                 word_features_vocab_sizes : List[int],
                 vec_features_size : int,
                 hidden_size : int,
                 num_layers : int) -> None:
        super().__init__()
        self._word_features_encoder = maybe_cuda(
            WordFeaturesEncoder(word_features_vocab_sizes,
                                hidden_size, 1, hidden_size))
        self._features_classifier = maybe_cuda(
            DNNScorer(hidden_size + vec_features_size,
                      hidden_size, num_layers))
    def forward(self,
                word_features_batch : torch.LongTensor,
                vec_features_batch : torch.FloatTensor) -> torch.FloatTensor:
        encoded_word_features = self._word_features_encoder(
            maybe_cuda(word_features_batch))
        scores = self._features_classifier(
            torch.cat((encoded_word_features, maybe_cuda(vec_features_batch)), dim=1))
        return scores

class FeaturesDNNEvaluator(TrainableEvaluator[FeaturesTokenMap]):
    def __init__(self) -> None:
        self._criterion = maybe_cuda(nn.MSELoss())
    def train(self, args : List[str]) -> None:
        argparser = argparse.ArgumentParser(self.description())
        self._add_args_to_parser(argparser)
        arg_values = argparser.parse_args(args)
        # THIS PART IS DIFFERENT THAN THE SUPERCLASS, hence the overriding
        save_states = self._optimize_model(arg_values)

        for state in save_states:
            with open(arg_values.save_file, 'wb') as f:
                torch.save((self.shortname(), (arg_values, state)), f)

    def _add_args_to_parser(self, parser : argparse.ArgumentParser,
                            default_values : Dict[str, Any] = {}) -> None:
        super()._add_args_to_parser(parser, default_values)
        add_nn_args(parser, default_values)

    def description(self) -> str:
        return "A state evaluator that uses the standard feature set on DNN's"

    def shortname(self) -> str:
        return "features-dnn"

    def _optimize_model(self, arg_values : argparse.Namespace) -> Iterable[FeaturesDNNEvaluatorState]:
        with print_time("Loading data", guard=arg_values.verbose):
            if arg_values.start_from:
                _, (features_token_map, state) = torch.load(arg_values.start_from)
                word_features_data, vec_features_data, outputs,\
                    word_features_vocab_sizes, vec_features_size = features_to_total_distances_tensors_with_map(
                        arg_values.scrape_file,
                        features_token_map)
            else:
                token_map, word_features_data, vec_features_data, outputs, \
                    word_features_vocab_sizes, vec_features_size = features_to_total_distances_tensors(
                        arg_values.scrape_file)

        with print_time("Converting data to tensors", guard=arg_values.verbose):
            tensors = [torch.LongTensor(word_features_data),
                       torch.FloatTensor(vec_features_data),
                       torch.FloatTensor(outputs)]

        with print_time("Building the model", guard=arg_values.verbose):
            model = self._get_model(arg_values, word_features_vocab_sizes, vec_features_size)
            if arg_values.start_from:
                model.load_saved_state(arg_values, state)

        return ((tmap_to_picklable(token_map), state)
                for state in optimize_checkpoints(tensors, arg_values, model,
                                                  lambda batch_tensors, model:
                                                  self._get_batch_prediction_loss(arg_values,
                                                                                  batch_tensors,
                                                                                  model)))
    def load_saved_state(self,
                         args : argparse.Namespace,
                         state : FeaturesDNNEvaluatorState) -> None:
        picklable_tmap, neural_state = state
        self.features_token_map = tmap_from_picklable(picklable_tmap);
        word_features_vocab_sizes, vec_features_size = features_vocab_sizes(self.features_token_map)
        self._model = maybe_cuda(self._get_model(args, word_features_vocab_sizes, vec_features_size))
        self._model.load_state_dict(neural_state.weights)

        self.training_loss = neural_state.loss
        self.num_epochs = neural_state.epoch
        self.training_args = args
    def _get_model(self, arg_values : argparse.Namespace,
                   word_features_vocab_sizes: List[int],
                   vec_features_size: int) -> FeaturesDNNEvaluatorModel:
        return FeaturesDNNEvaluatorModel(word_features_vocab_sizes, vec_features_size,
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
        word_features_batch, vec_features_batch = sample_context_features(self.features_token_map,
                                                                          state.relevant_lemmas,
                                                                          state.prev_tactics,
                                                                          state.hypotheses,
                                                                          state.goal)
        # eprint(f"Word features: {word_features_batch}")
        # eprint(f"Vec features: {vec_features_batch}")
        model_output = self._model(torch.LongTensor([word_features_batch]),
                                   torch.FloatTensor([vec_features_batch]))
        # eprint(f"Model output: {model_output}")
        return model_output[0].item()

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
