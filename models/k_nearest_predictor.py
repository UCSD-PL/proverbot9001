#!/usr/bin/env python3

import argparse
import time
import shutil
import math
import threading
import itertools
import statistics
from queue import PriorityQueue

import torch
from models.tactic_predictor import TacticPredictor
from models.components import SimpleEmbedding

from typing import Tuple, Dict, TypeVar, Generic, Optional, Callable

from format import read_pair
import tokenizer
from tokenizer import KeywordTokenizer, get_topk_keywords

from util import *

T = TypeVar('T')
V = TypeVar('V')

class SPTreeNode(Generic[V]):
    def __init__(self, value : float , axis : int,
                 left_child : Optional['SPTreeNode[V]'],
                 right_child : Optional['SPTreeNode[V]'],
                 item : Optional[Tuple[List[float], V]] = None) -> None:
        assert (not left_child is None and not right_child is None and item is None) or \
            (left_child is None and right_child is None and not item is None)
        self.left = left_child
        self.right = right_child
        self.value = value
        self.axis = axis
        self.item = item
        pass
    def __str__(self):
        if self.left is None:
            assert self.right is None
            assert not self.item is None
            return "Leaf: {}".format(self.item)
        else:
            assert not self.right is None
            assert self.item is None
            return "Branch on dim {} at {}".format(self.axis, self.value)
    def getSamples(self) -> List[Tuple[List[float], V]]:
        if self.left is None:
            assert self.right is None
            assert not self.item is None
            return [self.item]
        else:
            assert not self.right is None
            return self.left.getSamples() + self.right.getSamples()

class NearnessTree(Generic[T]):
    def __init__(self, items : List[Tuple[List[int], T]]) -> None:
        num_dimensions = len(items[0][0])
        if len(items) == 0:
            self.tree = None
        else:
            start = time.time()
            floatItems = [(floatVector(vec), o) for vec, o in items]
            dim_values = zip(*[vec for vec, o in floatItems])
            self.dim_maxs = [max(values + (1,)) for values in dim_values]
            normalizedFloatItems = [(self.normalizeVector(vec), o)
                                    for vec, o in floatItems]
            self.tree = self.buildTree(normalizedFloatItems, num_dimensions, 0)
            timeTaken = time.time() - start
            print("Built tree in {:.2f}".format(timeTaken))
        pass
    def normalizeVector(self, vec : List[float]) -> List[float]:
        return [floatItem / maxItem for floatItem, maxItem in zip(vec, self.dim_maxs)]
    def buildTree(self, items : List[Tuple[List[float], T]],
                  num_dimensions : int, cur_dimension : int) -> Optional[SPTreeNode[T]]:
        assert len(items) > 0
        if len(items) == 1:
            return SPTreeNode(items[0][0][0], 0, None, None, item=items[0])
        else:
            feature_vectors = [item[0] for item in items]
            dim_value_list = list(zip(*feature_vectors))
            best_split_dimension = max(enumerate(statistics.pvariance(dim_values) for
                                                 dim_values in dim_value_list),
                                       key=lambda x: x[1])[0]
            dim_sorted = sorted(items, key=lambda x: x[0][best_split_dimension])
            num_items = len(items)
            left_list = dim_sorted[:num_items//2]
            right_list = dim_sorted[num_items//2:]
            split_value = dim_sorted[num_items//2][0][best_split_dimension]
            return SPTreeNode(split_value, best_split_dimension,
                              self.buildTree(left_list),
                              self.buildTree(right_list))
    def findNearest(self, item : List[int]) -> Optional[Tuple[List[float], T]]:
        normalizedItem = self.normalizeVector(floatVector(item))
        def nearestNeighbor(curTree : SPTreeNode[T], best_distance : float) \
            -> Tuple[Tuple[List[float], T], float]:
            if curTree.left is None and curTree.right is None:
                assert not curTree.item is None
                return curTree.item, vectorDistanceSquared(normalizedItem,
                                                           curTree.item[0])
            else:
                if normalizedItem[curTree.axis] <= curTree.value:
                    assert not curTree.left is None
                    left_nearest, left_best_distance = nearestNeighbor(
                        curTree.left, best_distance)
                    new_best_distance = min(left_best_distance, best_distance)
                    if (normalizedItem[curTree.axis] + new_best_distance > curTree.value):
                        assert not curTree.right is None
                        right_nearest, right_best_distance = nearestNeighbor(
                            curTree.right, new_best_distance)
                        if right_best_distance < new_best_distance:
                            return right_nearest, right_best_distance
                    return left_nearest, left_best_distance
                else:
                    assert not curTree.right is None
                    right_nearest, right_best_distance = nearestNeighbor(
                        curTree.right, best_distance)
                    new_best_distance = min(right_best_distance, best_distance)
                    if (normalizedItem[curTree.axis] + new_best_distance > curTree.value):
                        assert not curTree.left is None
                        left_nearest, left_best_distance = nearestNeighbor(
                            curTree.left, new_best_distance)
                        if left_best_distance < new_best_distance:
                            return left_nearest, left_best_distance
                    return right_nearest, right_best_distance
        if self.tree is None:
            return None
        start = time.time()
        answer = nearestNeighbor(self.tree, float("Inf"))[0]
        timeTaken = time.time() - start
        # print("Found nearest neighbor in {:.2f} seconds".format(timeTaken))
        return answer

class KNNPredictor(TacticPredictor):
    def load_saved_state(self, filename : str) -> None:
        checkpoint = torch.load(filename)
        assert checkpoint["embedding"]
        self.embedding = checkpoint["embedding"]
        assert checkpoint["tokenizer"]
        self.tokenizer = checkpoint["tokenizer"]
        assert checkpoint["tree"]
        self.bst = checkpoint["tree"]
        pass

    def getOptions(self) -> List[Tuple[str, str]]:
        return [("num tokens", str(self.tokenizer.numTokens()))]

    def __init__(self, options : Dict[str, Any]) -> None:
        assert options["filename"]
        self.load_saved_state(options["filename"])

    def predictKTactics(self, in_data : Dict[str, str], k : int) -> List[str]:
        input_vector = getWordbagVector(self.tokenizer.toTokenList(in_data["goal"]),
                                        self.tokenizer.numTokens())

        nearest = self.bst.findNearest(input_vector)
        assert not nearest is None
        return [self.embedding.decode_token(nearest[1])] * k

    def predictKTacticsWithLoss(self, in_data : Dict[str, str], k : int,
                                correct : str) -> Tuple[List[str], float]:
        return self.predictKTactics(in_data, k), 0

def vectorDistanceSquared(vec1 : List[float], vec2 : List[float]):
    return sum([(item1 - item2) ** 2 for item1, item2 in zip(vec1, vec2)])

def floatVector(vec : List[int]) -> List[float]:
    return [float(dim) for dim in vec]

def getWordbagVector(goal : List[int], vocab_size : int) -> List[int]:
    wordbag = [0] * vocab_size
    for t in goal:
        assert t < vocab_size, \
            "t: {}, context_vocab_size(): {}".format(t, vocab_size)
        wordbag[t] += 1
    return wordbag

def read_scrapefile(filename : str, num_pairs : float = float("Inf")) -> List[Tuple[str, str]]:
    dataset : List[Tuple[str,str]] = []
    with open(filename, 'r') as scrapefile:
        pair = read_pair(scrapefile)
        while pair and len(dataset) < num_pairs:
            dataset.append(pair)
            pair = read_pair(scrapefile)

    return dataset

def main(args_list : List[str]) -> None:
    parser = argparse.ArgumentParser(description=
                                     "A k-nearest neighbors predictor")
    parser.add_argument("scrape_file")
    parser.add_argument("save_file")
    args = parser.parse_args(args_list)

    embedding = SimpleEmbedding()
    print("Reading data...")
    untokenized_samples = read_scrapefile(args.scrape_file, 10000)
    print("Read {} data pairs".format(len(untokenized_samples)))
    print("Getting keywords...")
    keywords = get_topk_keywords([sample[0] for sample in untokenized_samples], 100)
    tokenizer = KeywordTokenizer(keywords, 2)
    print("Encoding data...")
    samples = [(getWordbagVector(tokenizer.toTokenList(context),
                                 tokenizer.numTokens()),
                embedding.encode_token(get_stem(tactic)))
               for context, tactic in untokenized_samples
               if not re.match("[\{\}\+\-\*].*", tactic)]
    print("Building BST...")
    bst = NearnessTree(samples)
    print("Loaded.")
    with open(args.save_file, 'wb') as f:
        torch.save({'embedding': embedding,
                    'tokenizer': tokenizer,
                    'tree': bst}, f)
    print("Saved.")
