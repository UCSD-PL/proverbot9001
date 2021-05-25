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
import sys
from tqdm import tqdm
from typing import (Dict, List, Tuple, cast, BinaryIO, TypeVar, Any,
                    Optional, Sequence)

import torch
import torch.nn as nn
import torch.utils.data as data
from torch import optim
import torch.optim.lr_scheduler as scheduler
from pathlib_revised import Path2

import coq_serapy as serapi_instance
import tokenizer

from util import maybe_cuda, eprint
from coq_serapy.contexts import TacticContext
from models.q_estimator import QEstimator
from models.components import WordFeaturesEncoder, DNNScorer
from models.features_polyarg_predictor import FeaturesPolyargPredictor
from dataloader import (sample_context_features,
                        get_vec_features_size,
                        get_word_feature_vocab_sizes,
                        encode_fpa_stem,
                        encode_fpa_arg,
                        get_num_indices,
                        get_num_tokens,
                        get_premise_features,
                        get_premise_features_size,
                        tokenize)

PolyargQMetadata = Tuple[Dict[str, int], Dict[str, int]]


class PolyargQEstimator(QEstimator):
    def __init__(self, learning_rate: float, batch_step: int, gamma: float,
                 fpa_predictor: FeaturesPolyargPredictor) \
            -> None:
        self.predictor = fpa_predictor
        self.model = PolyargQModel(
            get_vec_features_size(fpa_predictor.metadata) +
            self.action_vec_features_size() +
            1,  # The extra 1 is for the certainty feature
            self.action_word_features_sizes() +
            get_word_feature_vocab_sizes(fpa_predictor.metadata),
            128, 2)
        self.optimizer = optim.SGD(self.model.parameters(), learning_rate)
        self.adjuster = scheduler.StepLR(self.optimizer, batch_step,
                                         gamma=gamma)
        self.criterion = nn.MSELoss()
        self.total_batches = 0

    @property
    def fpa_metadata(self):
        return self.predictor.metadata

    @property
    def dataloader_args(self):
        return self.predictor.dataloader_args

    def __call__(self, inputs: List[Tuple[TacticContext, str, float]],
                 progress: bool = False) \
            -> List[float]:
        state_word_features_batch, state_vec_features_batch \
            = zip(*[self._features(state, certainty) for
                    (state, action, certainty) in inputs])
        with torch.no_grad():
            encoded_actions_batch = [self._encode_action(state, action)
                                     for (state, action, certainty) in
                                     tqdm(inputs, desc="Encoding actions",
                                          disable=not progress)]
        all_vec_features_batch = [torch.cat((maybe_cuda(action_vec),
                                             maybe_cuda(torch.FloatTensor(svf))),
                                            dim=0).unsqueeze(0)
                                  for (_, action_vec), svf in
                                  zip(encoded_actions_batch,
                                      state_vec_features_batch)]
        all_word_features_batch = [action_words + swf
                                   for (action_words, _), swf in
                                   zip(encoded_actions_batch,
                                       state_word_features_batch)]
        with torch.no_grad():
            output = self.model(torch.LongTensor(all_word_features_batch),
                                torch.cat(all_vec_features_batch, dim=0))
        for item in output:
            assert item == item, (output,
                                  all_word_features_batch,
                                  all_vec_features_batch)
        return list(output)

    def get_input_tensors(self,
                          samples: List[Tuple[TacticContext, str,
                                              float, float]]) \
            -> List[torch.Tensor]:
        state_word_features, state_vec_features = zip(
            *[self._features(state, certainty)
              for state, _, certainty, _ in samples])
        encoded_actions = [self._encode_action(state, action)
                           for state, action, _, _ in samples]
        all_vec_features = [torch.cat((maybe_cuda(action_vec),
                                       maybe_cuda(torch.FloatTensor(svf))),
                                       dim=0).unsqueeze(0)
                            for (_, action_vec), svf in
                            zip(encoded_actions, state_vec_features)]
        all_word_features = [action_words + swf for (action_words, _), swf in
                             zip(encoded_actions, state_word_features)]
        return [torch.LongTensor(all_word_features),
                torch.cat(all_vec_features, dim=0)]


    def train(self, samples: List[Tuple[TacticContext, str, float, float]],
              batch_size: Optional[int] = None,
              num_epochs: int = 1,
              show_loss: bool = False) -> None:
        input_tensors = self.get_input_tensors(samples)
        expected_outputs = [output for _, _, _, output in samples]
        if batch_size:
            batches: Sequence[Sequence[torch.Tensor]] = data.DataLoader(
                data.TensorDataset(*(input_tensors+[torch.FloatTensor(expected_outputs)])),
                batch_size=batch_size,
                num_workers=0,
                shuffle=True, pin_memory=True,
                drop_last=True)
        else:
            batches = [input_tensors+[torch.FloatTensor(expected_outputs)]]
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
        cword_feats, cvec_feats = \
            sample_context_features(self.dataloader_args,
                                    self.fpa_metadata,
                                    context.relevant_lemmas,
                                    context.prev_tactics,
                                    context.hypotheses,
                                    context.goal)
        return cword_feats, cvec_feats + [certainty]

    def action_vec_features_size(self) -> int:
        premise_features_size = get_premise_features_size(
            self.dataloader_args,
            self.fpa_metadata)
        return 128 + premise_features_size

    def action_word_features_sizes(self) -> List[int]:
        self.predictor.metadata, num_indices = get_num_indices(self.fpa_metadata)
        return [num_indices, 3]

    def _encode_action(self, context: TacticContext, action: str) \
            -> Tuple[List[int], torch.FloatTensor]:
        stem, argument = serapi_instance.split_tactic(action)
        stem_idx = encode_fpa_stem(self.dataloader_args,
                                   self.fpa_metadata,
                                   stem)
        all_prems = context.hypotheses + context.relevant_lemmas
        arg_idx = encode_fpa_arg(self.dataloader_args,
                                 self.fpa_metadata,
                                 all_prems,
                                 context.goal,
                                 argument.strip())
        assert arg_idx is not None, (action, argument.strip(), context.goal)

        tokenized_goal = tokenize(self.dataloader_args,
                                  self.fpa_metadata,
                                  context.goal)
        premise_features_size = get_premise_features_size(
            self.dataloader_args,
            self.fpa_metadata)
        if arg_idx == 0:
            # No arg
            arg_type_idx = 0
            encoded_arg = torch.zeros(128 + premise_features_size)
        elif arg_idx <= self.dataloader_args.max_length:
            # Goal token arg
            arg_type_idx = 1
            encoded_arg = maybe_cuda(torch.cat((
                self.predictor.goal_token_encoder(
                    torch.LongTensor([stem_idx]),
                    torch.LongTensor([tokenized_goal])
                ).squeeze(0)[arg_idx].to(device=torch.device("cpu")),
               torch.zeros(premise_features_size)),
                                    dim=0))
        else:
            # Hyp arg
            arg_type_idx = 2
            arg_hyp = all_prems[
                arg_idx - (self.dataloader_args.max_length + 1)]
            entire_encoded_goal = self.predictor.entire_goal_encoder(
                torch.LongTensor([tokenized_goal]))
            tokenized_arg_hyp = tokenize(
                self.dataloader_args,
                self.fpa_metadata,
                serapi_instance.get_hyp_type(arg_hyp))
            encoded_arg = maybe_cuda(torch.cat((
                self.predictor.hyp_encoder(
                    torch.LongTensor([stem_idx]),
                    entire_encoded_goal,
                    torch.LongTensor([tokenized_arg_hyp])).to(device=torch.device("cpu")),
                torch.FloatTensor(get_premise_features(
                    self.dataloader_args,
                    self.fpa_metadata,
                    context.goal,
                    arg_hyp))),
                                     dim=0))

        return [stem_idx, arg_type_idx], encoded_arg

    def save_weights(self, filename: Path2, args: argparse.Namespace) -> None:
        with cast(BinaryIO, filename.open('wb')) as f:
            torch.save(("polyarg evaluator", args, sys.argv,
                        True,
                        self.model.state_dict()),
                       f)

    def load_saved_state(self, args: argparse.Namespace,
                         unparsed_args: List[str],
                         metadata: PolyargQMetadata,
                         state: Dict[str, Any]) -> None:
        self.model.load_state_dict(state)
        pass

    def share_memory(self):
        self.model.share_memory()


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


class PolyargQModel(nn.Module):
    def __init__(self,
                 vec_features_size: int,
                 word_feature_sizes: List[int],
                 hidden_size: int,
                 num_layers: int) -> None:
        super().__init__()
        # Consider making the word embedding the same for all
        # token-type inputs, also for tactic-type inputs
        self._word_features_encoder = maybe_cuda(
            WordFeaturesEncoder(word_feature_sizes,
                                hidden_size, 1, hidden_size))
        self._features_classifier = maybe_cuda(
            DNNScorer(hidden_size + vec_features_size,
                      hidden_size, num_layers))
        self.word_feature_sizes = word_feature_sizes

    def forward(self,
                word_features_batch: torch.LongTensor,
                vec_features_batch: torch.FloatTensor) -> torch.FloatTensor:
        try:
            encoded_word_features = self._word_features_encoder(
                maybe_cuda(word_features_batch))
        except IndexError:
            for wf_vals in  word_features_batch:
                for idx, (wf_val, wf_size) in enumerate(zip(wf_vals, self.word_feature_sizes)):
                    assert wf_val < wf_size, f"Value {wf_val} for word feature {idx} is at least size {wf_size}"
        features = torch.cat((encoded_word_features,
                              maybe_cuda(vec_features_batch)),
                             dim=1)
        scores = self._features_classifier(features)\
                     .view(vec_features_batch.size()[0])
        return scores
