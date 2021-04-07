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

import coq_serapy as serapi_instance
import tokenizer

from models.q_estimator import QEstimator
from util import maybe_cuda, eprint
from coq_serapy.contexts import TacticContext
from models.components import WordFeaturesEncoder, DNNScorer

import torch
import torch.nn as nn
import torch.utils.data as data
from torch import optim
import torch.optim.lr_scheduler as scheduler
import argparse
import sys
from pathlib_revised import Path2

from typing import (Dict, List, Tuple, cast, BinaryIO, TypeVar, Any,
                    Optional, Sequence)


FeaturesQMetadata = Tuple[Dict[str, int], Dict[str, int]]


class FeaturesQEstimator(QEstimator):
    def __init__(self, learning_rate: float, batch_step: int, gamma: float) \
            -> None:
        self._num_tactics = 32
        self._num_tokens = 128
        self.model = FeaturesQModel(self._num_tactics, self._num_tokens,
                                    3, 128, 2)
        # self.model = SimplifiedQModel(self._num_tactics, 4, 2)
        self.optimizer = optim.SGD(self.model.parameters(), learning_rate)
        self.adjuster = scheduler.StepLR(self.optimizer, batch_step,
                                         gamma=gamma)
        self.criterion = nn.MSELoss()
        self.tactic_map: Dict[str, int] = {}
        self.token_map: Dict[str, int] = {}
        self.total_batches = 0
        pass

    def __call__(self, inputs: List[Tuple[TacticContext, str, float]]) -> List[float]:
        state_word_features_batch, vec_features_batch \
            = zip(*[self._features(state, certainty) for
                    (state, action, certainty) in inputs])
        encoded_actions_batch = [self._encode_action(state, action)
                                 for (state, action, certainty) in inputs]
        all_word_features_batch = [list(encoded_action) + state_word_features
                                   for encoded_action, state_word_features in
                                   zip(encoded_actions_batch,
                                       state_word_features_batch)]
        with torch.no_grad():
            output = self.model(torch.LongTensor(all_word_features_batch),
                                torch.FloatTensor(vec_features_batch))
        for item in output:
            assert item == item, (all_word_features_batch, vec_features_batch)
        return list(output)

    def train(self, samples: List[Tuple[TacticContext, str, float, float]],
              batch_size: Optional[int] = None,
              num_epochs: int = 1,
              show_loss: bool = False) -> None:
        for context, action, certainty, score in samples:
            assert score != float("-Inf") and score != float("Inf") and score == score
        state_word_features, vec_features = zip(*[self._features(state, certainty)
                                                  for state, _, certainty, _ in samples])
        encoded_actions = [self._encode_action(state, action)
                           for state, action, _, _ in samples]
        all_word_features = [list(ea) + swf for ea, swf in
                             zip(encoded_actions, state_word_features)]
        expected_outputs = [output for _, _, certainty, output in samples]
        if batch_size:
            batches: Sequence[Sequence[torch.Tensor]] = data.DataLoader(
                data.TensorDataset(
                    torch.LongTensor(all_word_features),
                    torch.FloatTensor(vec_features),
                    torch.FloatTensor(expected_outputs)),
                batch_size=batch_size,
                num_workers=0,
                shuffle=True, pin_memory=True,
                drop_last=True)
        else:
            batches = [[torch.LongTensor(all_word_features),
                        torch.FloatTensor(vec_features),
                        torch.FloatTensor(expected_outputs)]]
        for epoch in range(0, num_epochs):
            epoch_loss = 0.
            for idx, batch in enumerate(batches):
                self.optimizer.zero_grad()
                word_features_batch, vec_features_batch, \
                    expected_outputs_batch = batch
                outputs = self.model(word_features_batch,
                                     vec_features_batch)
                loss = self.criterion(
                    outputs, maybe_cuda(expected_outputs_batch))
                loss.backward()
                self.optimizer.step()
                self.adjuster.step()
                self.total_batches += 1
                epoch_loss += loss.item()
                eprint(epoch_loss / len(batches),
                       guard=show_loss and epoch % 10 == 0
                       and idx == len(batches) - 1)
                eprint("Batch {}: Learning rate {:.12f}".format(
                    self.total_batches,
                    self.optimizer.param_groups[0]['lr']),
                       guard=show_loss and epoch % 10 == 0
                       and idx == len(batches) - 1)

    def _features(self, context: TacticContext, certainty: float) \
            -> Tuple[List[int], List[float]]:
        if len(context.prev_tactics) > 1:
            prev_tactic = serapi_instance.get_stem(context.prev_tactics[-1])
            prev_tactic_index = emap_lookup(self.tactic_map, 32, prev_tactic)
        else:
            prev_tactic_index = 0
        if context.goal != "":
            goal_head_index = emap_lookup(self.token_map, 128,
                                          tokenizer.get_words(context.goal)[0])
        else:
            goal_head_index = 0
        goal_length_feature = min(len(tokenizer.get_words(context.goal)),
                                  100) / 100
        num_hyps_feature = min(len(context.hypotheses), 30) / 30
        return [prev_tactic_index, goal_head_index], \
            [goal_length_feature, num_hyps_feature, certainty]

    def _encode_action(self, context: TacticContext, action: str) \
            -> Tuple[int, int]:
        stem, argument = serapi_instance.split_tactic(action)
        stem_idx = emap_lookup(self.tactic_map, 32, stem)
        all_premises = context.hypotheses + context.relevant_lemmas
        stripped_arg = argument.strip(".").strip()
        if stripped_arg == "":
            arg_idx = 0
        else:
            index_hyp_vars = dict(serapi_instance.get_indexed_vars_in_hyps(
                all_premises))
            if stripped_arg in index_hyp_vars:
                hyp_varw, _, rest = all_premises[index_hyp_vars[stripped_arg]]\
                    .partition(":")
                arg_idx = emap_lookup(self.token_map, 128,
                                      tokenizer.get_words(rest)[0]) + 2
            else:
                goal_symbols = tokenizer.get_symbols(context.goal)
                if stripped_arg in goal_symbols:
                    arg_idx = emap_lookup(self.token_map, 128,
                                          stripped_arg) + 128 + 2
                else:
                    arg_idx = 1
        return stem_idx, arg_idx

    def save_weights(self, filename: Path2, args: argparse.Namespace) -> None:
        with cast(BinaryIO, filename.open('wb')) as f:
            torch.save(("features evaluator", args, sys.argv,
                        (self.tactic_map, self.token_map),
                        self.model.state_dict()),
                       f)

    def load_saved_state(self, args: argparse.Namespace,
                         unparsed_args: List[str],
                         metadata: FeaturesQMetadata,
                         state: Dict[str, Any]) -> None:
        self.tactic_map, self.token_map = metadata
        self.model.load_state_dict(state)
        pass


T = TypeVar('T')


def emap_lookup(emap: Dict[T, int], size: int, item: T):
    if item in emap:
        return emap[item]
    elif len(emap) < size - 1:
        emap[item] = len(emap) + 1
        return emap[item]
    else:
        return 0


class SimplifiedQModel(nn.Module):
    def __init__(self, num_tactics: int, hidden_size: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_tactics = num_tactics
        # self.embedding = maybe_cuda(nn.Embedding(num_tactics, hidden_size))
        self.dnn = maybe_cuda(DNNScorer(num_tactics, hidden_size, num_layers))

    def forward(self, tactics_batch: torch.LongTensor):
        # embedded = self.embedding(tactics_batch)
        one_hot = torch.nn.functional.one_hot(
            tactics_batch, self.num_tactics).float()
        return self.dnn(one_hot).view(tactics_batch.size()[0])

    def print_weights(self) -> None:
        self.dnn.print_weights()


class FeaturesQModel(nn.Module):
    def __init__(self,
                 num_tactics: int,
                 num_tokens: int,
                 vec_features_size: int,
                 hidden_size: int,
                 num_layers: int) -> None:
        super().__init__()
        # Consider making the word embedding the same for all
        # token-type inputs, also for tactic-type inputs
        self._word_features_encoder = maybe_cuda(
            WordFeaturesEncoder([num_tactics, num_tokens * 2 + 2,
                                 num_tactics, num_tokens],
                                hidden_size, 1, hidden_size))
        self._features_classifier = maybe_cuda(
            DNNScorer(hidden_size + vec_features_size,
                      hidden_size, num_layers))

    def forward(self,
                word_features_batch: torch.LongTensor,
                vec_features_batch: torch.FloatTensor) -> torch.FloatTensor:
        encoded_word_features = self._word_features_encoder(
            maybe_cuda(word_features_batch))
        scores = self._features_classifier(
            torch.cat((encoded_word_features, maybe_cuda(vec_features_batch)),
                      dim=1))\
                     .view(vec_features_batch.size()[0])
        return scores
