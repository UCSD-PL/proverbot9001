#!/usr/bin/env python3

import json
import argparse
import time
import sys
import pickle
import heapq
import math
from typing import Dict, List, Tuple, Optional, IO, NamedTuple, cast, Set, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import torch
import coq2vec
from completed_proof import completed_proof
#from rl import ReplayBuffer
from coq_serapy.contexts import (FullContext, truncate_tactic_context,
                                         Obligation, TacticContext, ProofContext)

Transition = Tuple[str, Sequence[Obligation]]
FullTransition = Tuple[Obligation, str, List[Obligation]]

class ReplayBuffer:
    _contents: Dict[Obligation, Tuple[int, Set[Transition]]]
    window_size: int
    window_end_position: int
    allow_partial_batches: int
    def __init__(self, window_size: int,
                 allow_partial_batches: bool) -> None:
        self.window_size = window_size
        self.window_end_position = 0
        self.allow_partial_batches = allow_partial_batches
        self._contents = {}

    def sample(self, batch_size) -> Optional[List[Tuple[Obligation, Set[Transition]]]]:
        sample_pool: List[Tuple[Obligation, List[Transition]]] = []
        for obl, (last_updated, transitions) in self._contents.copy().items():
            if last_updated <= self.window_end_position - self.window_size:
                del self._contents[obl]
            else:
                sample_pool.append((obl, transitions))
        if len(sample_pool) >= batch_size:
            return random.sample(sample_pool, batch_size)
        if self.allow_partial_batches:
            return sample_pool
        return None

    def add_transition(self, transition: FullTransition) -> None:
        self._contents[transition[0]] = (self.window_end_position,
                                         {(transition[1], tuple(transition[2]))} |
                                         self._contents.get(transition[0], (0, set()))[1])
        self.window_end_position += 1
