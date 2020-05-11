#!/usr/bin/env python3.7
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
from argparse import Namespace
import re

import torch

from typing import Dict, Any, List, Tuple, NamedTuple, Union

from tokenizer import Tokenizer
from models.tactic_predictor import TokenizingPredictor, Prediction, TokenizerEmbeddingState
from models.components import Embedding, SimpleEmbedding, PredictorState
from format import read_tuple, ScrapedTactic, TacticContext
from util import *
from serapi_instance import get_stem
from data import get_text_data, Dataset, TokenizedDataset

from dataclasses import dataclass

class TryCommonSample(NamedTuple):
    tactic : int
@dataclass
class TryCommonState(PredictorState):
    inner : List[float]
@dataclass(init=True, repr=True)
class TryCommonDataset(Dataset):
    data : List[TryCommonSample]
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i : Any):
        return self.data[i]

class TryCommonPredictor(TokenizingPredictor[TryCommonDataset, TryCommonState]):
    def __init__(self) -> None:
        super().__init__()
    def predictKTactics(self, in_data : TacticContext, k : int) \
        -> List[Prediction]:
        return [Prediction(self._embedding.decode_token(idx) + ".", prob) for idx, prob
                in zip(*list_topk(self.probabilities, k))]
    def predictKTacticsWithLoss(self, in_data : TacticContext,
                                k : int, correct : str) -> \
        Tuple[List[Prediction], float]:
        # Try common doesn't calculate a meaningful loss
        return self.predictKTactics(in_data, k), 0
    def getOptions(self) -> List[Tuple[str, str]]:
        return list(vars(self.training_args).items()) + [("predictor", "trycommon")]
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, correct : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        return [self.predictKTactics(TacticContext([], [], [], ""), k)] * len(in_data), 0.
    def _encode_tokenized_data(self, data : TokenizedDataset, arg_values : Namespace,
                               t : Tokenizer, e : Embedding)\
                               -> TryCommonDataset:
        return TryCommonDataset([TryCommonSample(tactic)
                                 for prev_tactics, hyps, goal, tactic in
                                 data])
    def _optimize_model_to_disc(self,
                                encoded_data : TryCommonDataset,
                                encdec_state : TokenizerEmbeddingState,
                                arg_values : Namespace) \
        -> None:
        stem_counts = [0] * encdec_state.embedding.num_tokens()
        for sample in encoded_data:
            stem_counts[sample.tactic] += 1

        total_count = sum(stem_counts)
        stem_probs = [count / total_count for count in stem_counts]

        with open(arg_values.save_file, 'wb') as f:
            torch.save(("trycommon", (arg_values, encdec_state, stem_probs)), f)
    def load_saved_state(self,
                         args : Namespace,
                         unparsed_args : List[str],
                         metadata : TokenizerEmbeddingState,
                         state : TryCommonState) -> None:
        self._tokenizer = metadata.tokenizer
        self._embedding = metadata.embedding
        self.training_args = args
        self.context_filter = args.context_filter
        self.probabilities = state.inner
        self.unparsed_args = unparsed_args
        pass
    def _description(self) -> str:
        return "A simple predictor which tries the k most common tactic stems."

def train(arg_list : List[str]) -> None:
    predictor = TryCommonPredictor()
    predictor.train(arg_list)
