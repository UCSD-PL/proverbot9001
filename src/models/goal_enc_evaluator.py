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

from models.components import NeuralPredictorState, DNNScorer, add_nn_args
from models.state_evaluator import TrainableEvaluator, StateEvaluationDataset
from models.tactic_predictor import optimize_checkpoints, add_tokenizer_args
from dataloader import (goals_to_total_distances_tensors,
                        goals_to_total_distances_tensors_with_meta,
                        goal_enc_get_num_tokens,
                        GoalEncMetadata,
                        DataloaderArgs)
from util import maybe_cuda, eprint, print_time, LongTensor

import argparse
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from typing import List, Tuple, Iterable, Sequence, Dict, Any, cast

GoalEncState = Tuple[GoalEncMetadata, NeuralPredictorState]

class GoalEncModel(nn.Module):
    def __init__(self,
                 input_vocab_size : int,
                 hidden_size : int,
                 num_layers : int) -> None:
        super().__init__()
        self._token_embedding = maybe_cuda(nn.Embedding(input_vocab_size, hidden_size))
        self._gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self._scorer = maybe_cuda(DNNScorer(hidden_size, hidden_size, num_layers))
        self._hidden_size = hidden_size

    def init_hidden(self, batch_size):
        return maybe_cuda(Variable(torch.zeros(1, batch_size, self._hidden_size)))

    def forward(self,
                goal_batch : torch.LongTensor):
        goal_var = maybe_cuda(Variable(goal_batch))
        batch_size = goal_batch.size()[0]

        hidden = self.init_hidden(batch_size)
        for i in range(goal_batch.size()[1]):
            token_batch = self._token_embedding(goal_var[:,i])\
                              .view(1, batch_size, self._hidden_size)
            token_batch = F.relu(token_batch)
            _, hidden = self._gru(token_batch, hidden)
            hidden = F.relu(hidden)

        end_token_embedded = self._token_embedding(LongTensor([1])
                                                   .expand(batch_size))\
                                 .view(1, batch_size, self._hidden_size)
        final_out, _ = self._gru(end_token_embedded, hidden)
        scores = self._scorer(final_out)
        return scores

class GoalEncEvaluator(TrainableEvaluator[GoalEncState]):
    def __init__(self) -> None:
        self._criterion = maybe_cuda(nn.MSELoss())
    def train(self, args : List[str]) -> None:
        argparser = argparse.ArgumentParser(self.description())
        self._add_args_to_parser(argparser)
        arg_values = argparser.parse_args(args)
        save_states = self._optimize_model(arg_values)

        for state in save_states:
            with open(arg_values.save_file, 'wb') as f:
                torch.save((self.shortname(), (arg_values, state)), f)

    def description(self) -> str:
        return "A state evaluator that uses an RNN to encode the goal"

    def shortname(self) -> str:
        return "eval-goal-enc"

    def _optimize_model(self, arg_values : argparse.Namespace) -> \
        Iterable[GoalEncState]:
        with print_time("Loading data", guard=arg_values.verbose):
            if arg_values.start_from:
                _, (arg_values, (metadata, state)) = \
                    torch.load(arg_values.start_from)
                _, tokenized_goals, outputs = \
                    goals_to_total_distances_tensors_with_meta(
                        extract_dataloader_args(arg_values),
                        str(arg_values.scrape_file), metadata)
            else:
                metadata, tokenized_goals, outputs = \
                    goals_to_total_distances_tensors(
                        extract_dataloader_args(arg_values),
                        str(arg_values.scrape_file))

        with print_time("Converting data to tensors", guard=arg_values.verbose):
            tensors = [torch.LongTensor(tokenized_goals),
                       torch.FloatTensor(outputs)]

        with print_time("Building the model", guard=arg_values.verbose):
            model = self._get_model(arg_values, goal_enc_get_num_tokens(metadata))

            if arg_values.start_from:
                model.load_saved_state(arg_values, state)

        return ((metadata, state) for state in
                optimize_checkpoints(tensors, arg_values, model,
                                     lambda batch_tensors, model:
                                     self._get_batch_prediction_loss(arg_values,
                                                                     batch_tensors,
                                                                     model)))

    def load_saved_state(self,
                         arg_values : argparse.Namespace,
                         state : GoalEncState) -> None:
        self.metadata, neural_state = state
        self._model = maybe_cuda(
            self._get_model(arg_values, goal_enc_get_num_tokens(self.metadata)))
        self._model.load_state_dict(neural_state.weights)
        self.training_loss = neural_state.loss
        self.num_epochs = neural_state.epoch
        self.training_args = arg_values

    def _get_model(self, arg_values : argparse.Namespace,
                   num_tokens : int) -> GoalEncModel:
        return GoalEncModel(num_tokens,
                            arg_values.hidden_size,
                            arg_values.num_layers)

    def _get_batch_prediction_loss(self, arg_values : argparse.Namespace,
                                   data_batch : Sequence[torch.Tensor],
                                   model : GoalEncModel) -> \
                                   torch.FloatTensor:
        tokenized_goals, outputs = \
            cast(Tuple[torch.LongTensor, torch.FloatTensor], data_batch)
        predicted_scores = model(tokenized_goals)
        return self._criterion(predicted_scores, maybe_cuda(outputs))

    def scoreState(self, state : TacticContext) -> float:
        tokenized_goal = tokenize_goal(extract_dataloader_args(self.training_args),
                                       state.goal)
        model_output = self._model(torch.LongTensor([tokenized_goal]))
        return model_output[0].item() * self.training_args.max_distance

    def _add_args_to_parser(self, parser : argparse.ArgumentParser,
                            default_values : Dict[str, Any] = {}) -> None:

        super()._add_args_to_parser(parser, default_values)
        add_nn_args(parser, default_values)
        add_tokenizer_args(parser, default_values)

        parser.add_argument("--max-distance", type=int,
                            default=default_values.get("max_distance", 10))

def extract_dataloader_args(args: argparse.Namespace) -> DataloaderArgs:
    dargs = DataloaderArgs();
    dargs.max_distance = args.max_distance
    dargs.max_length = args.max_length
    dargs.context_filter = args.context_filter
    assert args.load_tokens, "Must provide a keywords file for rust dataloader."
    dargs.keywords_file = args.load_tokens
    return dargs

def main(arg_list : List[str]) -> None:
    predictor = GoalEncEvaluator()
    predictor.train(arg_list)
