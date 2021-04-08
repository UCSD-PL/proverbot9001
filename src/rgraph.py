#!/usr/bin/env python3
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

import json
import argparse
from pathlib_revised import Path2
from dataclasses import dataclass
from typing import (List, Optional, Dict, Any, Tuple, cast)

import pygraphviz as pgv
import torch

from util import unwrap
from coq_serapy.contexts import (TacticContext, ProofContext,
                                 truncate_tactic_context)
import predict_tactic

from models import (features_polyarg_predictor, features_q_estimator,
                    polyarg_q_estimator, tactic_predictor)
from models.q_estimator import QEstimator


@dataclass
class LabeledTransition:
    relevant_lemmas: List[str]
    prev_tactics: List[str]
    before: ProofContext
    after: ProofContext
    action: str
    original_certainty: float
    reward: float
    graph_node: Optional['LabeledNode']

    @property
    def after_context(self) -> TacticContext:
        return TacticContext(self.relevant_lemmas,
                             self.prev_tactics,
                             self.after.focused_hyps,
                             self.after.focused_goal)

    @property
    def before_context(self) -> TacticContext:
        return TacticContext(self.relevant_lemmas,
                             self.prev_tactics,
                             self.before.focused_hyps,
                             self.before.focused_goal)

    def to_dict(self) -> Dict[str, Any]:
        return {"relevant_lemmas": self.relevant_lemmas,
                "prev_tactics": self.prev_tactics,
                "before": self.before.to_dict(),
                "after": self.after.to_dict(),
                "action": self.action,
                "original_certainty": self.original_certainty,
                "reward": self.reward}

    @classmethod
    def from_dict(cls, data) -> 'LabeledTransition':
        return LabeledTransition(data["relevant_lemmas"],
                                 data["prev_tactics"],
                                 ProofContext.from_dict(data["before"]),
                                 ProofContext.from_dict(data["after"]),
                                 data["action"],
                                 data["original_certainty"],
                                 data["reward"],
                                 None)


@dataclass(init=True)
class LabeledNode:
    node_id: int
    transition: Optional[LabeledTransition]
    parent: Optional["LabeledNode"]
    children: List["LabeledNode"]

    @property
    def action(self) -> str:
        return unwrap(self.transition).action

    @property
    def reward(self) -> float:
        return unwrap(self.transition).reward


class ReinforceGraph:
    __next_node_id: int
    start_node: LabeledNode
    lemma_name: str
    graph_nodes: List[Tuple[int, Dict[str, str]]]
    graph_edges: List[Tuple[int, int, Dict[str, str]]]

    def __init__(self, lemma_name: str) -> None:
        self.__next_node_id = 0
        self.lemma_name = lemma_name
        self.graph_nodes = []
        self.graph_edges = []
        self.start_node = self.mkNode(None, None)
        pass

    def addTransition(self, src: LabeledNode, transition: LabeledTransition,
                      **kwargs) -> LabeledNode:
        for child in src.children:
            if child.action == transition.action:
                assert child.reward == transition.reward
                child.transition = transition
                return child
        return self.mkNode(transition, src, **kwargs)

    def addGhostTransition(self, src: LabeledNode,
                           transition: LabeledTransition,
                           **kwargs) -> LabeledNode:
        for child in src.children:
            if child.action == transition.action:
                return child
        return self.mkNode(transition, src, fillcolor="grey", **kwargs)

    def mkNode(self, transition: Optional[LabeledTransition],
               previous_node: Optional[LabeledNode],
               **kwargs) -> LabeledNode:
        if 'fillcolor' not in kwargs and transition:
            if transition.reward > 0:
                color = "palegreen"
            elif transition.reward < 0:
                color = "indianred1"
            else:
                color = "white"
            self.graph_nodes.append((self.__next_node_id,
                                     {"label": transition.action,
                                      "fillcolor": color,
                                      "style": "filled",
                                      **kwargs}))
        elif transition:
            self.graph_nodes.append((self.__next_node_id,
                                     {"label": transition.action,
                                      **kwargs}))
        else:
            self.graph_nodes.append((self.__next_node_id,
                                     {"label": self.lemma_name,
                                      **kwargs}))
        self.__next_node_id += 1
        newNode = LabeledNode(self.__next_node_id-1,
                              transition,
                              previous_node, [])
        if previous_node:
            assert transition
            self.graph_edges.append((previous_node.node_id, newNode.node_id,
                                     {"label": f"{transition.reward:.2f}"}))
            previous_node.children.append(newNode)
        return newNode

    def mkQED(self, src: LabeledNode):
        for existing_node in src.children:
            if existing_node.transition is None:
                return
        self.graph_nodes.append((self.__next_node_id,
                                 {"label": "QED"}))
        self.graph_edges.append((src.node_id, self.__next_node_id, {}))
        newNode = LabeledNode(self.__next_node_id,
                              None, src, [])
        src.children.append(newNode)
        self.__next_node_id += 1
        cur_node = src
        while cur_node != self.start_node:
            self.setNodeOutlineColor(cur_node, "palegreen1")
            assert cur_node.parent
            cur_node = cur_node.parent
        pass

    def setNodeColor(self, node: LabeledNode, color: str) -> None:
        for (nidx, props) in self.graph_nodes:
            if nidx == node.node_id:
                props["fillcolor"] = color
                props["style"] = "filled"
                continue

    def setNodeOutlineColor(self, node: LabeledNode, color: str) -> None:
        for (nidx, props) in self.graph_nodes:
            if nidx == node.node_id:
                props["color"] = color
                continue

    def setNodeApproxQScore(self, node: LabeledNode, score: float) -> None:
        for (nidx, props) in self.graph_nodes:
            if nidx == node.node_id:
                props["label"] = f"{node.action} (~{score:.2f})"

    def save(self, filename: str) -> None:
        def node_to_dict(node: LabeledNode):
            return {"id": node.node_id,
                    "transition": node.transition.to_dict()
                    if node.transition else None,
                    "children": [node_to_dict(child) for child in
                                 node.children]}

        with open(filename, 'w') as f:
            json.dump({"nodes": self.graph_nodes,
                       "edges": self.graph_edges,
                       "lemma": self.lemma_name,
                       "data": node_to_dict(self.start_node)},
                      f)

    @classmethod
    def load(cls, filename: str) -> 'ReinforceGraph':
        def node_from_dict(d: Dict[str, Any],
                           parent: Optional[LabeledNode] = None):
            node = LabeledNode(d["id"],
                               LabeledTransition.from_dict(d["transition"])
                               if d["transition"] else None,
                               parent,
                               [])
            node.children = [node_from_dict(child, parent=node)
                             for child in d["children"]]
            return node

        with open(filename, 'r') as f:
            d = json.load(f)
        graph = ReinforceGraph(d["lemma"])
        graph.graph_nodes = d["nodes"]
        graph.graph_edges = d["edges"]
        graph.start_node = node_from_dict(d["data"])
        return graph

    def draw(self, filename: str) -> None:
        graph = pgv.AGraph(directed=True)
        for (nidx, props) in self.graph_nodes:
            graph.add_node(nidx, **props)
        for (a, b, props) in self.graph_edges:
            graph.add_edge(a, b, **props)
        # with nostderr():
        graph.draw(filename, prog="dot")


def assignApproximateQScores(graph: ReinforceGraph,
                             max_term_length: int,
                             predictor: tactic_predictor.TacticPredictor,
                             estimator: QEstimator,
                             node: Optional[LabeledNode] = None) -> None:
    if node is None:
        node = graph.start_node
    elif node.transition:
        ctxt = truncate_tactic_context(
            node.transition.before_context,
            max_term_length)
        score = estimator([(ctxt,
                            node.transition.action,
                            node.transition.original_certainty)])[0]
        graph.setNodeApproxQScore(
            node, score)
    for child in node.children:
        assignApproximateQScores(
            graph, max_term_length,
            predictor, estimator, child)


def main():
    parser = \
      argparse.ArgumentParser(
          description="A module for drawing and re-drawing reinforcement "
          "learning graphs")

    parser.add_argument("predictor_weights")
    parser.add_argument("estimator_weights")
    parser.add_argument("graph_json")
    parser.add_argument("--max-term-length", default=512, type=int)

    args = parser.parse_args()

    predictor = predict_tactic.loadPredictorByFile(args.predictor_weights)
    q_estimator_name, *saved = torch.load(str(args.estimator_weights))
    if q_estimator_name == "features evaluator":
        q_estimator = features_q_estimator.FeaturesQEstimator(0, 0, 0)
    elif q_estimator_name == "polyarg evaluator":
        q_estimator = polyarg_q_estimator.PolyargQEstimator(
            0, 0, 0,
            cast(features_polyarg_predictor.FeaturesPolyargPredictor,
                 predictor))

    graph = ReinforceGraph.load(args.graph_json)
    assignApproximateQScores(graph, args.max_term_length,
                             predictor, q_estimator)
    graph.draw(Path2(args.graph_json).stem)


if __name__ == "__main__":
    main()
