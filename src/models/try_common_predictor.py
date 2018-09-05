#!/usr/bin/env python3

import argparse
import re

import torch

from typing import Dict, Any, List, Tuple, TypeVar

from models.tactic_predictor import TacticPredictor
from models.components import SimpleEmbedding
from format import read_tuple
from util import *

class TryCommonPredictor(TacticPredictor):
    def load_saved_state(self, filename : str) -> None:
        checkpoint = torch.load(filename)
        assert checkpoint['probabilities']
        assert checkpoint['stem-embeddings']

        self.probabilities = checkpoint['probabilities']
        self.embedding = checkpoint['stem-embeddings']

    def __init__(self, options : Dict[str, Any]) -> None:
        assert options["filename"]
        self.load_saved_state(options["filename"])

    def predictKTactics(self, in_data : Dict[str, str], k : int) -> List[str]:
        probs, indices = list_topk(self.probabilities, k)
        return [self.embedding.decode_token(idx) + "." for idx in indices]

def read_scrapefile(filename, embedding):
    dataset = []
    with open(filename, 'r') as scrapefile:
        t = read_tuple(scrapefile)
        while pair:
            hyps, context, tactic = t
            dataset.append(embedding.encode_token(get_stem(tactic)))
            t = read_tuple(scrapefile)
    return dataset

def train(arg_list : List[str]) -> None:
    parser = argparse.ArgumentParser(description=
                                     "A simple predictor which tries "
                                     "the k most common tactic stems.")
    parser.add_argument("scrape_file")
    parser.add_argument("save_file")
    args = parser.parse_args(arg_list)
    embedding = SimpleEmbedding()
    dataset = read_scrapefile(args.scrape_file, embedding)

    stem_counts = [0] * embedding.num_tokens()
    for stem in dataset:
        stem_counts[stem] += 1

    total_count = sum(stem_counts)
    stem_probs = [count / total_count for count in stem_counts]

    with open(args.save_file, 'wb') as f:
        torch.save({'probabilities' : stem_probs,
                    'stem-embeddings' : embedding}, f)

T = TypeVar('T')
def list_topk(lst : List[T], k : int) -> Tuple[List[int], List[T]]:
    l = sorted(enumerate(lst), key=lambda x:x[1], reverse=True)
    lk = l[:k]
    return reversed(list(zip(*lk)))
