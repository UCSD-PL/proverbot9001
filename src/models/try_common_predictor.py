#!/usr/bin/env python3.7

import argparse
from argparse import Namespace
import re

import torch

from typing import Dict, Any, List, Tuple, NamedTuple, Union

from tokenizer import Tokenizer
from models.tactic_predictor import TokenizingPredictor, Prediction, TacticContext, TokenizerEmbeddingState
from models.components import Embedding, SimpleEmbedding
from format import read_tuple, ScrapedTactic
from util import *
from serapi_instance import get_stem
from data import get_text_data, Dataset, TokenizedDataset

from dataclasses import dataclass

class TryCommonSample(NamedTuple):
    tactic : int
@dataclass(init=True, repr=True)
class TryCommonDataset(Dataset):
    data : List[TryCommonSample]
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i : Any):
        return self.data[i]

class TryCommonPredictor(TokenizingPredictor[TryCommonDataset, List[float]]):
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
        return list(vars(self.training_args).items())
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, correct : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        return [self.predictKTactics(TacticContext([], [], ""), k)] * len(in_data), 0.
    def _encode_tokenized_data(self, data : TokenizedDataset, arg_values : Namespace,
                               t : Tokenizer, e : Embedding)\
                               -> TryCommonDataset:
        return TryCommonDataset([TryCommonSample(tactic)
                                 for prev_tactics, goal, tactic in
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
            torch.save((arg_values, encdec_state, stem_probs), f)
    def load_saved_state(self,
                         args : Namespace,
                         metadata : TokenizerEmbeddingState,
                         state : List[float]) -> None:
        self._tokenizer = metadata.tokenizer
        self._embedding = metadata.embedding
        self.training_args = args
        self.context_filter = args.context_filter
        self.probabilities = state
        pass
    def _description(self) -> str:
        return "A simple predictor which tries the k most common tactic stems."

def train(arg_list : List[str]) -> None:
    predictor = TryCommonPredictor()
    predictor.train(arg_list)
