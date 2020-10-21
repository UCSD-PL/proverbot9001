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
import time
import shutil
import math
import threading
import itertools
import statistics

import torch
from models.tactic_predictor import (TacticPredictor, Prediction,
                                     TrainablePredictor,
                                     add_tokenizer_args)
from models.args import take_std_args

from typing import Tuple, Dict, TypeVar, Generic, Optional, Callable, Union, cast, NamedTuple

from tokenizer import tokenizers, Tokenizer
from data import get_text_data, filter_data, \
    encode_bag_classify_data, encode_bag_classify_input, ScrapedTactic, RawDataset, ClassifyBagDataset
from context_filter import get_context_filter
from serapi_instance import get_stem
from models.components import Embedding, PredictorState

from util import *
from format import TacticContext
from dataclasses import dataclass

V = TypeVar('V')

@dataclass
class KNearestPredictorState(PredictorState, Generic[V]):
    inner: "NearnessTree[V]"


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


class KNNMetadata(NamedTuple):
    embedding : Embedding
    tokenizer : Tokenizer
    tokenizer_name : str
    num_samples : int
    context_filter : str
class KNNPredictor(TrainablePredictor[ClassifyBagDataset, KNNMetadata, KNearestPredictorState]):
    def _encode_data(self, data : RawDataset, arg_values : argparse.Namespace) \
        -> Tuple[ClassifyBagDataset, KNNMetadata]:
        samples, tokenizer, embedding = \
            encode_bag_classify_data(data,
                                     tokenizers[arg_values.tokenizer],
                                     arg_values.num_keywords, 2)
        return samples, KNNMetadata(embedding, tokenizer, arg_values.tokenizer,
                                    len(samples), arg_values.context_filter)
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) -> None:
        super().add_args_to_parser(parser, default_values)
        add_tokenizer_args(parser, default_values)
        parser.add_argument("--max-length", dest="max_length", type=int,
                            default=default_values.get("max-length", 30))
    def _optimize_model_to_disc(self,
                                encoded_data : ClassifyBagDataset,
                                encdec_state : KNNMetadata,
                                arg_values : argparse.Namespace) \
        -> None:
        bst = NearnessTree(encoded_data)
        with open(arg_values.save_file, 'wb') as f:
            torch.save(("k-nearest", (arg_values, encdec_state, KNearestPredictorState(1, bst))), f)
    def load_saved_state(self,
                         args : argparse.Namespace,
                         unparsed_args : List[str],
                         metadata : KNNMetadata,
                         state : KNearestPredictorState) -> None:
        self.embedding, self.tokenizer, self.tokenizer_name, \
            self.num_samples, self.context_filter = metadata
        self.bst = state.inner
        self.training_args = args
        self.unparsed_args = unparsed_args

    def getOptions(self) -> List[Tuple[str, str]]:
        return [("# tokens", str(self.tokenizer.numTokens())),
                ("# tactics (stems)", str(self.embedding.num_tokens())),
                ("# samples used", str(self.num_samples)),
                ("tokenizer", self.tokenizer_name),
                ("context_filter", self.context_filter),
        ]

    def _description(self) -> str:
        return "A k-nearest neighbors predictor"

    def predictKTactics(self, in_data : TacticContext, k : int) -> \
        List[Prediction]:
        input_vector = encode_bag_classify_input(in_data.goal, self.tokenizer)

        nearest = self.bst.findKNearest(input_vector, k)
        assert not nearest is None
        for pair in nearest:
            assert not pair is None
        predictions = [Prediction(self.embedding.decode_token(output) + ".", .5**i)
                       for i, (neighbor, output) in enumerate(nearest)]
        return predictions

    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int,
                                correct : str) -> Tuple[List[Prediction], float]:
        # k-nearest doesn't calculate a meaningful loss
        return self.predictKTactics(in_data, k), 0
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, correct : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        return [self.predictKTactics(in_data_point, k) for in_data_point in in_data], 0

def normalizeVector(dim_maxs : List[int], vec : List[int]) -> List[float]:
    return [floatItem / maxItem for floatItem, maxItem in zip(vec, dim_maxs)]

def vectorDistanceSquared(vec1 : List[float], vec2 : List[float]):
    return sum([(item1 - item2) ** 2 for item1, item2 in zip(vec1, vec2)])

def floatVector(vec : List[int]) -> List[float]:
    return [float(dim) for dim in vec]

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

def main(args_list : List[str]) -> None:
    predictor = KNNPredictor()
    predictor.train(args_list)
