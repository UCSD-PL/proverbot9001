#!/usr/bin/env python3

import argparse
import re

import torch

from typing import Dict, Any, List, Tuple, Union

from models.tactic_predictor import TacticPredictor, Prediction, ContextInfo
from models.components import SimpleEmbedding
from format import read_tuple, ScrapedTactic
from util import *
from serapi_instance import get_stem
from data import get_text_data

class TryCommonPredictor(TacticPredictor):
    def load_saved_state(self, filename : str) -> None:
        checkpoint = torch.load(filename)

        self.probabilities = checkpoint['probabilities']
        self.embedding = checkpoint['stem-embeddings']
        self.context_filter = checkpoint['context-filter']

    def __init__(self, options : Dict[str, Any]) -> None:
        assert options["filename"]
        self.load_saved_state(options["filename"])

    def predictKTactics(self, in_data : Dict[str, Union[str, List[str]]], k : int) \
        -> List[Prediction]:
        return [Prediction(self.embedding.decode_token(idx) + ".", prob) for idx, prob
                in zip(*list_topk(self.probabilities, k))]
    def predictKTacticsWithLoss(self, in_data : Dict[str, Union[str, List[str]]],
                                k : int, correct : str) -> \
        Tuple[List[Prediction], float]:
        # Try common doesn't calculate a meaningful loss
        return self.predictKTactics(in_data, k), 0
    def getOptions(self) -> List[Tuple[str, str]]:
        return [("context filter", self.context_filter)]
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[ContextInfo],
                                      k : int, correct : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        return [self.predictKTactics({}, k)] * len(in_data), 0.

def train(arg_list : List[str]) -> None:
    parser = argparse.ArgumentParser(description=
                                     "A simple predictor which tries "
                                     "the k most common tactic stems.")
    parser.add_argument("scrape_file")
    parser.add_argument("save_file")
    parser.add_argument("--context-filter", dest="context_filter", type=str,
                        default="goal-changes%no-args")
    args = parser.parse_args(arg_list)
    text_dataset = get_text_data(args.scrape_file, args.context_filter, verbose=True)
    substitutions = {"auto": "eauto.",
                     "intros until": "intros.",
                     "intro": "intros.",
                     "constructor": "econstructor."}
    preprocessed_dataset = [ScrapedTactic(prev_tactics, hyps, goal, tactic
                                          if get_stem(tactic) not in substitutions
                                          else substitutions[get_stem(tactic)])
                            for prev_tactics, hyps, goal, tactic in text_dataset]
    embedding = SimpleEmbedding()
    dataset = [embedding.encode_token(get_stem(tactic))
               for prev_tactics, hyps, context, tactic
               in preprocessed_dataset]
    stem_counts = [0] * embedding.num_tokens()
    for stem in dataset:
        stem_counts[stem] += 1

    total_count = sum(stem_counts)
    stem_probs = [count / total_count for count in stem_counts]

    with open(args.save_file, 'wb') as f:
        torch.save({'probabilities' : stem_probs,
                    'stem-embeddings' : embedding,
                    'context-filter': args.context_filter}, f)
