#!/usr/bin/env python3

import argparse
import time
import shutil
import math
import threading
import itertools
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
                 item : Optional[Tuple[List[int], V]] = None) -> None:
        assert (not left_child is None and not right_child is None and item is None) or \
            (left_child is None and right_child is None and not item is None)
        self.left = left_child
        self.right = right_child
        self.value = value
        self.axis = axis
        self.item = item
        pass

class NearnessTree(Generic[T]):
    def __init__(self, items : List[Tuple[List[int], T]]) -> None:
        num_dimensions = len(items[0][0])
        print("Building tree with {} dimensions from {}".format(num_dimensions, items))
        if len(items) == 0:
            self.tree = None
        else:
            self.tree = self.buildTree(items, num_dimensions, 0)
        pass
    def buildTree(self, items : List[Tuple[List[int], T]],
                  num_dimensions : int, cur_dimension : int) -> Optional[SPTreeNode[T]]:
        assert len(items) > 0
        if len(items) == 1:
            return SPTreeNode(items[0][0][cur_dimension],
                              cur_dimension, None, None,
                              item=items[0])
        dim_values = [item[0][cur_dimension] for item in items]
        partition_value = median(dim_values)
        if partition_value == items[0][0][cur_dimension]:
            for item in items:
                assert item[0][cur_dimension] == partition_value,\
                    "Not a true median! List {}, median {}"\
                    .format(dim_values, partition_value)
            # print("All items have the same value on dimension {}".format(cur_dimension))
            return self.buildTree(items, num_dimensions,
                                  (cur_dimension + 1) % num_dimensions)

        left_items = list(filter(lambda x: x[0][cur_dimension] <= partition_value,
                                 items))
        assert len(left_items) < len(items)
        right_items = list(filter(lambda x: x[0][cur_dimension] > partition_value,
                                  items))
        assert len(right_items) < len(items)
        newNode = SPTreeNode(partition_value, cur_dimension,
                             self.buildTree(left_items, num_dimensions,
                                            (cur_dimension + 1) % num_dimensions),
                             self.buildTree(right_items, num_dimensions,
                                            (cur_dimension + 1) % num_dimensions))
        return newNode
    def findNearest(self, item : List[int]) -> Optional[Tuple[List[int], T]]:
        def nearestNeighbor(curTree : SPTreeNode[T], best_distance : float) \
            -> Tuple[Tuple[List[int], T], float]:
            if curTree.left is None and curTree.right is None:
                assert not curTree.item is None
                return curTree.item, vectorDistanceSquared(item, curTree.item[0])
            else:
                if item[curTree.axis] <= curTree.value:
                    assert not curTree.left is None
                    left_nearest, left_best_distance = nearestNeighbor(
                        curTree.left, best_distance)
                    new_best_distance = min(left_best_distance, best_distance)
                    if (item[curTree.axis] + new_best_distance > curTree.value):
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
                    if (item[curTree.axis] + new_best_distance > curTree.value):
                        assert not curTree.left is None
                        left_nearest, left_best_distance = nearestNeighbor(
                            curTree.left, new_best_distance)
                        if left_best_distance < new_best_distance:
                            return left_nearest, left_best_distance
                    return right_nearest, right_best_distance
        if self.tree is None:
            return None
        return nearestNeighbor(self.tree, float("Inf"))[0]

def median(lst : List[float]) -> float:
    def approx_median(lst : List[float]) -> float:
        assert len(lst) > 0
        if len(lst) == 1:
            return lst[0]
        k = 5
        chunks = [lst[i:i+k] for i in range(0, len(lst), k)]
        assert len(chunks) < len(lst), \
            "Input list is no longer than chunks! len(lst): {}, len(chunks): {}"\
            .format(len(lst), len(chunks))
        medians = [sorted(chunk)[len(chunk) // 2] for chunk in chunks]
        if len(medians) == 1:
            return medians[0]
        else:
            assert len(medians) < len(lst), \
                "Input list had length {}, but medians list had length {}"\
                .format(len(lst), len(medians))
            return approx_median(medians)

    def partition(items : List[T], predicate : Callable[[T], bool]):
        a, b = itertools.tee((predicate(item), item) for item in items)
        return ((item for pred, item in a if not pred),
                (item for pred, item in b if pred))

    def kselect(lst : List[float], k) -> float:
        assert k < len(lst)
        assert k > 0
        if k == 1:
            return lst[0]
        amedian = approx_median(lst)
        if amedian == lst[0]:
            assert False
            return lst[0]
        smaller_half, greater_half = partition(lst, lambda x: x >= amedian)
        greater_list = list(greater_half)
        assert len(greater_list) > 0
        assert len(greater_list) < len(lst)
        if len(greater_list) >= k:
            return kselect(greater_list, k)
        else:
            smaller_list = list(smaller_half)
            assert len(smaller_list) < len(lst)
            return kselect(smaller_list, k - len(greater_list))

    assert len(lst) > 0
    if len(lst) % 2 == 1:
        return kselect(lst, len(lst) // 2)
    else:
        return (kselect(lst, len(lst) // 2) + kselect(lst, len(lst) // 2 + 1)) / 2

class KNNPredictor(TacticPredictor):
    def load_saved_state(self, filename : str) -> None:
        self.embedding = SimpleEmbedding()
        print("Reading data...")
        untokenized_samples = read_scrapefile(filename, 10)
        print("Getting keywords...")
        keywords = get_topk_keywords([sample[0] for sample in untokenized_samples], 100)
        self.tokenizer = KeywordTokenizer(keywords, 2)
        print("Encoding data...")
        samples = [(getWordbagVector(self.tokenizer.toTokenList(context),
                                     self.tokenizer.numTokens()),
                    self.embedding.encode_token(get_stem(tactic)))
                   for context, tactic in untokenized_samples
                   if not re.match("[\{\}\+\-\*].*", tactic)]
        print("Building BST...")
        self.bst = NearnessTree(samples)
        print("Loaded.")
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

def vectorDistanceSquared(vec1 : List[int], vec2 : List[int]):
    return sum([(item1 - item2) ** 2 for item1, item2 in zip(vec1, vec2)])

def getWordbagVector(goal : List[int], vocab_size : int) -> List[int]:
    wordbag = [0] * vocab_size
    for t in goal:
        assert t < vocab_size, \
            "t: {}, context_vocab_size(): {}".format(t, vocab_size)
        wordbag[t] += 1
    return wordbag

def read_scrapefile(filename : str, num_pairs : float = float("Inf")) -> List[Tuple[str, str]]:
    dataset = []
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
    shutil.copy2(args.scrape_file, args.save_file)
