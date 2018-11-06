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

from typing import Tuple, Dict, TypeVar, Generic, Optional, Callable, Union, cast

from tokenizer import tokenizers
from data import read_text_data, filter_data, \
    encode_bag_classify_data, encode_bag_classify_input
from context_filter import get_context_filter

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
        def buildTree(items : List[Tuple[List[float], T]]) \
            -> Optional[SPTreeNode[T]]:
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
                                  buildTree(left_list),
                                  buildTree(right_list))
        num_dimensions = len(items[0][0])
        for item in items:
            assert len(item[0]) == num_dimensions,\
                "This item has {} dimensions, "\
                "even though the first item has {} dimensions"\
                .format(len(item[0]), len(items[0][0]))
        if len(items) == 0:
            self.tree = None
        else:
            start = time.time()
            dim_values = zip(*[vec for vec, o in items])
            self.dim_maxs = [max(values + (1,)) for values in dim_values]
            normalizedFloatItems = [(self.normalizeVector(vec), o)
                                    for vec, o in items]
            self.tree = buildTree(normalizedFloatItems)
            timeTaken = time.time() - start
            print("Built tree in {:.2f}".format(timeTaken))
        pass
    def normalizeVector(self, vec : List[int]) -> List[float]:
        return normalizeVector(self.dim_maxs, vec)
    def unnormalizeVector(self, vec : List[float]) -> List[int]:
        return [int(floatItem * maxItem)
                for floatItem, maxItem in zip(vec, self.dim_maxs)]
    def getSamples(self, tree = None) -> List[Tuple[List[int], T]]:
        if tree is None:
            tree = self.tree
        if tree is None:
            return []
        else:
            return [(self.unnormalizeVector(vec), output)
                    for vec, output in tree.getSamples()]
    def findKNearest(self, item : List[int], k : int) -> \
        Optional[List[Tuple[List[int], T]]]:
        normalizedItem = self.normalizeVector(item)
        def kNearestNeighbors(curTree : SPTreeNode[T], k_best_distances : List[float]) \
            -> List[Tuple[Optional[Tuple[List[float], T]], float]]:
            if curTree.left is None and curTree.right is None:
                assert not curTree.item is None
                single_item = (curTree.item,
                               vectorDistanceSquared(normalizedItem,
                                                     curTree.item[0]))
                nearest_neighbors = [single_item,
                                     (None, float("Inf")),
                                     (None, float("Inf"))]
                for neighbor in nearest_neighbors:
                    if not neighbor[0] is None:
                        assert len(neighbor[0][0]) == len(item), \
                            "Item has {} dimensions, "\
                            "but the neighbor only has {} dimensions!"\
                            .format(len(item), len(neighbor[0][0]))
                return nearest_neighbors
            else:
                if normalizedItem[curTree.axis] <= curTree.value:
                    firstSubtree = curTree.left
                    secondSubtree = curTree.right
                else:
                    firstSubtree = curTree.right
                    secondSubtree = curTree.left
                assert not firstSubtree is None
                nearest_neighbors = kNearestNeighbors(firstSubtree, k_best_distances)
                first_k_best_distances = sorted([distance for item, distance in
                                                 nearest_neighbors] +
                                                k_best_distances)[:k]
                if (abs(normalizedItem[curTree.axis] - curTree.value) <
                    first_k_best_distances[-1]):
                    assert not secondSubtree is None
                    second_nearest_neighbors = kNearestNeighbors(secondSubtree,
                                                                 first_k_best_distances)
                    nearest_neighbors = sorted(nearest_neighbors +
                                               second_nearest_neighbors,
                                               key=lambda x: x[1])[:k]
                for neighbor in nearest_neighbors:
                    if not neighbor[0] is None:
                        assert len(neighbor[0][0]) == len(item), \
                            "Item has {} dimensions, "\
                            "but the neighbor only has {} dimensions!"\
                            .format(len(item), len(neighbor[0][0]))
                return nearest_neighbors
        def getNeighbor(pair : Tuple[Optional[Tuple[List[float], T]], float]) \
            -> Tuple[List[int], T]:

            neighbor, distance = pair
            assert not neighbor is None
            return self.unnormalizeVector(neighbor[0]), neighbor[1]

        if self.tree is None:
            return None
        # start = time.time()
        answer = [getNeighbor(pair) for pair in
                  kNearestNeighbors(self.tree, [float("Inf")] * k)]
        # timeTaken = time.time() - start
        # print("Found {} nearest neighbors in {:.2f} seconds".format(k, timeTaken))
        return answer


class KNNPredictor(TacticPredictor):
    def load_saved_state(self, filename : str) -> None:
        checkpoint = torch.load(filename)
        assert checkpoint["embedding"]
        self.embedding = checkpoint["embedding"]
        assert checkpoint["tokenizer"]
        self.tokenizer = checkpoint["tokenizer"]
        assert checkpoint["tokenizer-name"]
        self.tokenizer_name = checkpoint["tokenizer-name"]
        assert checkpoint["tree"]
        self.bst = checkpoint["tree"]
        assert checkpoint["num-samples"]
        self.num_samples = checkpoint["num-samples"]
        assert checkpoint["context-filter"]
        self.context_filter = checkpoint["context-filter"]
        pass

    def getOptions(self) -> List[Tuple[str, str]]:
        return [("# tokens", str(self.tokenizer.numTokens())),
                ("# tactics (stems)", self.embedding.num_tokens()),
                ("# samples used", self.num_samples),
                ("tokenizer", self.tokenizer_name),
                ("context filter", self.context_filter),
        ]

    def __init__(self, options : Dict[str, Any]) -> None:
        assert options["filename"]
        self.load_saved_state(options["filename"])

    def predictKTactics(self, in_data : Dict[str, Union[str, List[str]]], k : int) -> \
        List[Tuple[str, float]]:
        input_vector = encode_bag_classify_input(cast(str, in_data["goal"]), self.tokenizer)

        nearest = self.bst.findKNearest(input_vector, k)
        assert not nearest is None
        for pair in nearest:
            assert not pair is None
        predictions = [self.embedding.decode_token(output) + "."
                       for neighbor, output in nearest]
        return list(zip(predictions, (.5**i for i in itertools.count())))

    def predictKTacticsWithLoss(self, in_data : Dict[str, Union[str, List[str]]], k : int,
                                correct : str) -> Tuple[List[Tuple[str, float]], float]:
        # k-nearest doesn't calculate a meaningful loss
        return self.predictKTactics(in_data, k), 0

def normalizeVector(dim_maxs : List[int], vec : List[int]) -> List[float]:
    return [floatItem / maxItem for floatItem, maxItem in zip(vec, dim_maxs)]

def vectorDistanceSquared(vec1 : List[float], vec2 : List[float]):
    return sum([(item1 - item2) ** 2 for item1, item2 in zip(vec1, vec2)])

def floatVector(vec : List[int]) -> List[float]:
    return [float(dim) for dim in vec]

def main(args_list : List[str]) -> None:
    parser = argparse.ArgumentParser(description=
                                     "A k-nearest neighbors predictor")
    parser.add_argument("scrape_file")
    parser.add_argument("save_file")
    parser.add_argument("--num-keywords", dest="num_keywords",
                        default=250, type=int)
    parser.add_argument("--num-samples", dest="num_samples",
                        default=float("Inf"), type=float)
    parser.add_argument("--tokenizer",
                        choices=list(tokenizers.keys()), type=str,
                        default=list(tokenizers.keys())[0])
    parser.add_argument("--context-filter", dest="context_filter",
                        type=str,
                        default="default")
    args = parser.parse_args(args_list)

    print("Reading data...")
    raw_samples = list(read_text_data(args.scrape_file, args.num_samples))
    print("Read {} input-output pairs".format(len(raw_samples)))
    print("Filtering/Encoding data...")
    start = time.time()
    filtered_samples = filter_data(raw_samples, get_context_filter(args.context_filter))
    samples, tokenizer, embedding = encode_bag_classify_data(filtered_samples,
                                                             tokenizers[args.tokenizer],
                                                             args.num_keywords,
                                                             2)
    timeTaken = time.time() - start
    print("Encoded data in in {:.2f}".format(timeTaken))
    print("Building BST...")
    bst = NearnessTree(samples)
    print("Loaded.")
    with open(args.save_file, 'wb') as f:
        torch.save({'embedding': embedding,
                    'tokenizer': tokenizer,
                    'tokenizer-name': args.tokenizer,
                    'tree': bst,
                    'num-samples': len(samples),
                    'context-filter': args.context_fitler}, f)
    print("Saved.")

def filterNones(lst : List[Optional[T]]) -> List[T]:
    return [item for item in lst if
            not item is None]

def assertKNearestCorrect(neighbors : List[Tuple[List[int], T]],
                          samples   : List[Tuple[List[int], T]],
                          in_vec    : List[int],
                          k         : int,
                          dim_maxs  : List[int]) \
    -> None:
    samples_with_distance = [
        ((sample_vec, output),
         vectorDistanceSquared(normalizeVector(dim_maxs, sample_vec),
                               normalizeVector(dim_maxs, in_vec)))
        for sample_vec, output in samples]
    correct_k_nearest = [
        sample for sample, distance in
        sorted(samples_with_distance, key=lambda x: x[1])][:k]
    for correct, found in zip(correct_k_nearest, neighbors[:len(correct_k_nearest)]):
        assert correct[0] == found[0], "input:\n {} (len {})\ncorrect:\n{} (len {}),\nfound:\n{} (len {})"\
            .format(in_vec, len(in_vec), correct[0], len(correct[0]), found[0], len(found[0]))
    pass
