#!/usr/bin/env python3

import json
import argparse
import time
import sys
import pickle
import heapq
import math
from typing import Dict, List, Tuple, Optional, IO, NamedTuple, cast
from dataclasses import dataclass, field
from pathlib import Path

import pygraphviz as pgv
from tqdm import tqdm, trange

if sys.version_info >= (3, 10):
    from lemma_models import Lemma, UnhandledExpr

import coq_serapy
from coq_serapy.contexts import TacticContext, FullContext, ProofContext, truncate_tactic_context
import tokenizer
from models.tactic_predictor import Prediction, TacticPredictor
from search_results import TacticInteraction, SearchResult, SearchStatus
from util import nostderr, unwrap, eprint, mybarfmt

from value_estimator import Estimator

unnamed_goal_number: int = 0

class FeaturesExtractor:
    tactic_map: Dict[str, int]
    token_map: Dict[str, int]
    _num_tactics: int
    _num_tokens: int

    def __init__(self, tacticsfile: str, tokensfile: str) -> None:
        self.tactic_map = {}
        self.token_map = {}
        with open(tacticsfile, 'r') as f:
            for idx, line in enumerate(f, start=2):
                self.tactic_map[line.strip()] = idx
        with open(tokensfile, 'r') as f:
            for idx, line in enumerate(f, start=2):
                self.token_map[line.strip()] = idx
        self._num_tactics = len(self.tactic_map)
        self._num_tokens = len(self.token_map)

    def state_features(self, context: TacticContext) -> \
            Tuple[List[int], List[float]]:
        if len(context.prev_tactics) > 1:
            prev_tactic = coq_serapy.get_stem(context.prev_tactics[-1])
            prev_tactic_index = self.tactic_map.get(prev_tactic, 1)
        else:
            prev_tactic_index = 0

        if context.goal != "":
            goal_head_index = self.token_map.get(tokenizer.get_words(context.goal)[0], 1)
        else:
            goal_head_index = 0

        goal_length_feature = min(len(tokenizer.get_words(context.goal)),
                                  100) / 100
        num_hyps_feature = min(len(context.hypotheses), 30) / 30
        return [prev_tactic_index, goal_head_index], \
               [goal_length_feature, num_hyps_feature]

    def state_features_bounds(self) -> Tuple[List[int], List[float]]:
        return [self._num_tactics, self._num_tokens], [1.0, 1.0]

    def action_features(self, context: TacticContext,
                        action: str, certainty: float) \
            -> Tuple[List[int], List[float]]:
        stem, argument = coq_serapy.split_tactic(action)
        stem_idx = self.tactic_map.get(stem, 1)
        all_premises = context.hypotheses + context.relevant_lemmas
        stripped_arg = argument.strip(".").strip()
        if stripped_arg == "":
            arg_idx = 0
        else:
            index_hyp_vars = dict(coq_serapy.get_indexed_vars_in_hyps(
                all_premises))
            if stripped_arg in index_hyp_vars:
                hyp_varw, _, rest = all_premises[index_hyp_vars[stripped_arg]
                                                 ].partition(":")
                arg_idx = self.token_map.get(tokenizer.get_words(rest)[0], 1) + 2
            else:
                goal_symbols = tokenizer.get_symbols(context.goal)
                if stripped_arg in goal_symbols:
                    arg_idx = self.token_map.get(stripped_arg, 1) \
                                         + self._num_tokens + 2
                else:
                    arg_idx = 1
        return [stem_idx, arg_idx], [certainty]

    def action_features_bounds(self) -> Tuple[List[int], List[float]]:
        return [self._num_tactics, self._num_tokens * 2 + 2], [1.0]


@dataclass(init=True)
class LabeledNode:
    prediction: str
    certainty: float
    time_taken: Optional[float]
    node_id: int
    context_before: FullContext
    previous: Optional["LabeledNode"]
    children: List["LabeledNode"]


class SearchGraph:
    __graph: pgv.AGraph
    __next_node_id: int
    feature_extractor: Optional[FeaturesExtractor]
    start_node: LabeledNode

    def __init__(self, tactics_file: Path, tokens_file: Path, lemma_name: str,
                 features_json: bool) -> None:
        self.__graph = pgv.AGraph(directed=True)
        self.__next_node_id = 0
        self.start_node = self.mkNode(Prediction(lemma_name, 1.0),
                                      FullContext(
                                          [], [], ProofContext([], [], [], [])),
                                      None)
        self.start_node.time_taken = 0.0
        if features_json:
            self.feature_extractor = FeaturesExtractor(str(tactics_file),
                                                       str(tokens_file))
        pass

    def mkNode(self, prediction: Prediction, context_before: FullContext,
               previous_node: Optional[LabeledNode],
               **kwargs) -> LabeledNode:

        tooltip = ""
        for hyp in context_before.obligations.focused_hyps:
            tooltip += hyp[:64] + "&#10;"
        tooltip += "-" * 64 + "&#10;"
        tooltip += context_before.obligations.focused_goal[:64]

        self.__graph.add_node(self.__next_node_id,
                              label="{}\n({:.2f})".format(
                                  prediction.prediction,
                                  prediction.certainty),
                              tooltip=tooltip,
                              **kwargs)
        self.__next_node_id += 1
        newNode = LabeledNode(prediction.prediction, prediction.certainty,
                              None, self.__next_node_id-1,
                              context_before, previous_node, [])
        if previous_node:
            self.__graph.add_edge(previous_node.node_id,
                                  newNode.node_id, **kwargs)
            previous_node.children.append(newNode)
        return newNode

    def mkQED(self, predictionNode: LabeledNode):
        self.mkNode(Prediction("QED", 1.0), FullContext(
            [], [], ProofContext([], [], [], [])),
                    predictionNode,
                    fillcolor="green", style="filled")
        cur_node = predictionNode
        cur_path = []
        while cur_node != self.start_node:
            self.setNodeColor(cur_node, "palegreen1")
            cur_path.append(cur_node)
            assert cur_node.previous
            cur_node = cur_node.previous
        return [TacticInteraction(n.prediction, n.context_before.obligations)
                for n in reversed(cur_path)]
        pass

    def setNodeColor(self, node: LabeledNode, color: str) -> None:
        node_handle = self.__graph.get_node(node.node_id)
        if node_handle.attr["fillcolor"] != None and node_handle.attr["fillcolor"] != "":
            node_handle.attr["fillcolor"] += (":" + color)
        else:
            node_handle.attr["fillcolor"] = color
            node_handle.attr["style"] = "filled"

    def draw(self, filename: str) -> None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with nostderr():
            self.__graph.draw(filename, prog="dot")

    def write_feat_json(self, filename: str) -> None:
        assert self.feature_extractor
        def write_node(node: LabeledNode, f: IO[str]) -> None:
            assert self.feature_extractor
            if len(node.children) == 0:
                return
            state_feats = self.feature_extractor.state_features(
                node.children[0].context_before.as_tcontext())
            action_feats = [self.feature_extractor.action_features(
                child.context_before.as_tcontext(),
                child.prediction, child.certainty)
                            for child in node.children]
            action_rewards = [50 if child.prediction == "QED" else 0
                              for child in node.children]
            json.dump({"node_id": node.node_id,
                       "state_features": state_feats,
                       "actions": [{"features": feats,
                                    "reward": reward,
                                    "result_node": child.node_id}
                                   for feats, reward, child
                                   in zip(action_feats,
                                          action_rewards,
                                          node.children)]},
                      f)
            f.write("\n")
            for child in node.children:
                write_node(child, f)
        with Path(filename).open('w') as f:
            json.dump({"state_features_max_values":
                       self.feature_extractor.state_features_bounds(),
                       "action_features_max_values":
                       self.feature_extractor.action_features_bounds()},
                      f)
            f.write("\n")
            write_node(self.start_node, f)


class SubSearchResult (NamedTuple):
    solution: Optional[List[TacticInteraction]]
    solved_subgoals: int


def contextInPath(full_context: ProofContext, path: List[LabeledNode]):
    return any([coq_serapy.contextSurjective(full_context,
                                             n.context_before.obligations)
                for n in path])


def numNodesInTree(branching_factor: int, depth: int):
    assert depth > 0, f"depth is {depth}"
    result = int((branching_factor ** depth - 1) /
                 (branching_factor - 1))
    assert result >= 1, f"result is {result}"
    return result


def time_on_path(node: LabeledNode) -> float:
    if node.previous is None:
        return unwrap(node.time_taken)
    else:
        return time_on_path(unwrap(node.previous)) + unwrap(node.time_taken)


def tryPrediction(args: argparse.Namespace,
                  coq: coq_serapy.SerapiInstance,
                  prediction: str,
                  previousTime: float) \
                  -> Tuple[ProofContext, int, int, int,
                           Optional[Exception], float, bool]:
    coq.quiet = True
    time_left = max(args.max_proof_time - previousTime, 0)
    start_time = time.time()
    time_per_command = (coq.hammer_timeout + args.max_tactic_time
                        if coq.use_hammer else args.max_tactic_time)
    try:
        coq.run_stmt(prediction, timeout=min(time_left, time_per_command))
        error = None
    except (coq_serapy.TimeoutError, coq_serapy.ParseError,
            coq_serapy.CoqExn, coq_serapy.OverflowError,
            coq_serapy.ParseError,
            RecursionError,
            coq_serapy.UnrecognizedError) as e:
        return (unwrap(coq.proof_context), 0, 0, 0, e,
                time.time() - start_time, False)

    time_taken = time.time() - start_time
    num_stmts = 1
    subgoals_closed = 0
    unshelved = False
    if len(unwrap(coq.proof_context).fg_goals) == 0 and \
       len(unwrap(coq.proof_context).shelved_goals) > 0: # type: ignore
        unshelved = True
        coq.run_stmt("Unshelve.")
        num_stmts += 1
    while len(unwrap(coq.proof_context).fg_goals) == 0 \
            and not completed_proof(coq):
        coq.run_stmt("}")
        subgoals_closed += 1
        num_stmts += 1
    if coq.count_fg_goals() > 1 or \
       (coq.count_fg_goals() > 0 and subgoals_closed > 0):
        subgoals_opened = 1
        coq.run_stmt("{")
        num_stmts += 1
    else:
        subgoals_opened = 0
    context_after = coq.proof_context
    assert context_after
    return (context_after, num_stmts, subgoals_closed,
            subgoals_opened, error, time_taken, unshelved)


goalBignessLimit = 3000
maxHyps = 32

def contextIsBig(context: ProofContext):
    for obligation in context.all_goals:
        for hypothesis in obligation.hypotheses:
            if len(hypothesis) > goalBignessLimit:
                return True
        if len(obligation.goal) > goalBignessLimit:
            return True
        if len(obligation.hypotheses) > maxHyps:
            return True
    return False


class TqdmSpy(tqdm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._time = time.time

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        self.__n = value

    def update(self, value):
        self.n = self.n + value
        super().update(value)


def dfs_proof_search_with_graph(lemma_name: str,
                                module_prefix: str,
                                relevant_lemmas: List[str],
                                coq: coq_serapy.SerapiInstance,
                                output_dir: Path,
                                args: argparse.Namespace,
                                bar_idx: int,
                                predictor: TacticPredictor) \
                                -> SearchResult:
    g = SearchGraph(args.tactics_file, args.tokens_file, lemma_name,
                    args.features_json)

    def cleanupSearch(num_stmts: int, msg: Optional[str] = None):
        if msg:
            eprint(f"Cancelling {num_stmts} statements "
                   f"because {msg}.", guard=args.verbose >= 2)
        for _ in range(num_stmts):
            coq.cancel_last()
    hasUnexploredNode = False

    def search(pbar: tqdm, current_path: List[LabeledNode],
               subgoal_distance_stack: List[int],
               extra_depth: int) -> SubSearchResult:
        nonlocal hasUnexploredNode
        nonlocal relevant_lemmas
        global unnamed_goal_number
        full_context_before = FullContext(relevant_lemmas,
                                          coq.prev_tactics,
                                          unwrap(coq.proof_context))
        predictions = predictor.predictKTactics(
            truncate_tactic_context(full_context_before.as_tcontext(),
                                    args.max_term_length),
                    args.max_attempts)
        assert len(predictions) == args.max_attempts
        if coq.use_hammer:
            predictions = [Prediction(prediction.prediction[:-1] + "; try hammer.",
                                      prediction.certainty)
                           for prediction in predictions]
        num_successful_predictions = 0
        for _prediction_idx, prediction in enumerate(predictions):
            if num_successful_predictions >= args.search_width:
                break
            try:
                context_after, num_stmts, \
                    subgoals_closed, subgoals_opened, \
                    error, time_taken, unshelved = \
                    tryPrediction(args, coq, prediction.prediction,
                                  time_on_path(current_path[-1]))
                if error:
                    if args.count_failing_predictions:
                        num_successful_predictions += 1
                    if args.show_failing_predictions:
                        predictionNode = g.mkNode(prediction,
                                                  full_context_before,
                                                  current_path[-1])
                        predictionNode.time_taken = time_taken
                        if isinstance(error, RecursionError):
                            g.setNodeColor(predictionNode, "grey75")
                        else:
                            g.setNodeColor(predictionNode, "red")
                    continue
                num_successful_predictions += 1
                pbar.update(1)
                assert cast(TqdmSpy, pbar).n > 0

                predictionNode = g.mkNode(prediction,
                                          full_context_before,
                                          current_path[-1])
                predictionNode.time_taken = time_taken
                if unshelved:
                    predictionNode = g.mkNode(Prediction("Unshelve.", 1.0),
                                              full_context_before,
                                              predictionNode)
                    predictionNode.time_taken = 0

                # ### 1.
                if subgoal_distance_stack:
                    new_distance_stack = (subgoal_distance_stack[:-1] +
                                          [subgoal_distance_stack[-1]+1])
                else:
                    new_distance_stack = []

                # ### 2.
                new_extra_depth = extra_depth
                for _ in range(subgoals_closed):
                    closed_goal_distance = new_distance_stack.pop()
                    new_extra_depth += closed_goal_distance

                # ### 3.
                new_distance_stack += [0] * subgoals_opened

                #############
                if completed_proof(coq):
                    solution = g.mkQED(predictionNode)
                    return SubSearchResult(solution, subgoals_closed)
                elif contextInPath(context_after,
                                   current_path[1:] + [predictionNode]):
                    if not args.count_softfail_predictions:
                        num_successful_predictions -= 1
                    g.setNodeColor(predictionNode, "orange")
                    cleanupSearch(num_stmts,
                                  "resulting context is in current path")
                elif contextIsBig(context_after):
                    g.setNodeColor(predictionNode, "orange4")
                    cleanupSearch(num_stmts,
                                  "resulting context has too big a goal")
                elif len(current_path) < args.search_depth + new_extra_depth \
                        and len(current_path) < args.hard_depth_limit:
                    if subgoals_closed > 0:
                        g.setNodeColor(predictionNode, "blue")
                    sub_search_result = search(pbar,
                                               current_path + [predictionNode],
                                               new_distance_stack,
                                               new_extra_depth)
                    cleanupSearch(num_stmts, "we finished subsearch")
                    if sub_search_result.solution or \
                       sub_search_result.solved_subgoals > subgoals_opened:
                        new_subgoals_closed = \
                            subgoals_closed + \
                            sub_search_result.solved_subgoals - \
                            subgoals_opened
                        return SubSearchResult(sub_search_result.solution,
                                               new_subgoals_closed)
                    if subgoals_closed > 0:
                        return SubSearchResult(None, subgoals_closed)
                else:
                    hasUnexploredNode = True
                    cleanupSearch(num_stmts, "we hit the depth limit")
                    if subgoals_closed > 0:
                        # depth = (args.search_depth + new_extra_depth + 1) \
                        #     - len(current_path)
                        return SubSearchResult(None, subgoals_closed)
            except coq_serapy.CoqAnomaly:
                predictionNode = g.mkNode(prediction,
                                          full_context_before,
                                          current_path[-1])
                g.setNodeColor(predictionNode, "grey25")
                if lemma_name == "":
                    unnamed_goal_number += 1
                    g.draw(f"{output_dir}/{module_prefix}"
                           f"{unnamed_goal_number}.svg")
                else:
                    if args.features_json:
                        g.write_feat_json(f"{output_dir}/{module_prefix}"
                                          f"{lemma_name}.json")
                    g.draw(f"{output_dir}/{module_prefix}"
                           f"{lemma_name}.svg")

                raise
        return SubSearchResult(None, 0)
    total_nodes = numNodesInTree(args.search_width,
                                 args.search_depth + 2) - 1
    desc_name = lemma_name
    if len(desc_name) > 25:
        desc_name = desc_name[:22] + "..."
    if coq.count_fg_goals() > 1:
        coq.run_stmt("{")
        subgoals_stack_start = [0]
    else:
        subgoals_stack_start = []

    with TqdmSpy(total=total_nodes, unit="pred", file=sys.stdout,
                 desc=desc_name, disable=(not args.progress),
                 leave=False,
                 position=bar_idx + 1,
                 dynamic_ncols=True, bar_format=mybarfmt) as pbar:
        if args.search_prefix is None:
            command_list, _ = search(pbar, [g.start_node], subgoals_stack_start, 0)
        else:
            next_node = g.start_node
            for command in coq_serapy.read_commands(args.search_prefix):
                full_context_before = FullContext(relevant_lemmas,
                                                  coq.prev_tactics,
                                                  unwrap(coq.proof_context))
                next_node = g.mkNode(Prediction(command, 1.0),
                                     full_context_before,
                                     next_node)
                next_node.time_taken = 0.0
                coq.run_stmt(command)
            command_list, _ = search(pbar, [next_node], subgoals_stack_start, 0)
        pbar.clear()
    g.draw(f"{output_dir}/{module_prefix}{lemma_name}.svg")
    if args.features_json:
        g.write_feat_json(f"{output_dir}/{module_prefix}"
                          f"{lemma_name}.json")
    if command_list:
        return SearchResult(SearchStatus.SUCCESS, command_list)
    if hasUnexploredNode:
        return SearchResult(SearchStatus.INCOMPLETE, None)
    return SearchResult(SearchStatus.FAILURE, None)


def completed_proof(coq: coq_serapy.SerapiInstance) -> bool:
    if coq.proof_context:
        return len(coq.proof_context.all_goals) == 0 and \
            coq.tactic_history.curDepth() == 0
    return False


@dataclass
class BFSNode:
    prediction: Prediction
    postfix: List[str]
    score: float
    time_taken: float
    context_before: FullContext
    previous: Optional["BFSNode"]
    children: List["BFSNode"]
    color: Optional[str]

    def __init__(self, prediction: Prediction, score: float, time_taken: float,
                 postfix: List[str], context_before: FullContext, previous: Optional["BFSNode"],
                 color: Optional[str] = None) -> None:
        self.prediction = prediction
        self.score = score
        self.time_taken = time_taken
        self.postfix = postfix
        self.context_before = context_before
        self.previous = previous
        self.children = []
        if self.previous:
            self.previous.children.append(self)
        self.color = color
        pass

    def setNodeColor(self, color: str) -> None:
        assert color
        if self.color != None and self.color != "":
            self.color = (unwrap(self.color) + ":" + color)
        else:
            self.color = color

    def mkQED(self) -> None:
        qed_node = BFSNode(Prediction("QED", 1.0), 100, 0, [],
                           FullContext(
                            [], [], ProofContext([], [], [], [])),
                           self, "green")
        cur_node = self
        while cur_node.previous != None:
            cur_node.setNodeColor("palegreen1")
            cur_node = unwrap(cur_node.previous)

    def draw_graph(self, path: str) -> None:
        graph = pgv.AGraph(directed=True)
        next_node_id = 0
        def add_subgraph(root: "BFSNode") -> int:
            nonlocal graph
            nonlocal next_node_id
            label=f"{root.prediction.prediction}\n{root.score:.2e}"
            if root.color:
                fillcolor = root.color
                style="filled"
            else:
                fillcolor = "lightgrey"
                style=""

            tooltip = ""
            for hyp in root.context_before.obligations.focused_hyps:
                tooltip += hyp[:64] + "&#10;"
            tooltip += "-" * 64 + "&#10;"
            tooltip += root.context_before.obligations.focused_goal[:64]


            graph.add_node(next_node_id, label=label, fillcolor=fillcolor, style=style,
                           tooltip=tooltip)

            root_node_id = next_node_id
            next_node_id += 1
            for child in root.children:
                child_id = add_subgraph(child)
                graph.add_edge(root_node_id, child_id)
            return root_node_id
        add_subgraph(self)
        with nostderr():
            graph.draw(path, prog="dot")

    def pp(self) -> str:
        if not self.previous:
            return f" -> {self.prediction.prediction}"
        else:
            return f"{self.previous.prediction.prediction} => {self.prediction.prediction}"

    def commands(self) -> List[str]:
        return [node.prediction.prediction for node in
                self.path()]
    def interactions(self) -> List[TacticInteraction]:
        return [TacticInteraction(n.prediction.prediction,
                                  n.context_before.obligations)
                for n in self.path()]
    def total_time(self) -> float:
        return sum(node.time_taken for node in
                   self.path())

    def path(self) -> List['BFSNode']:
        if self.previous is None:
            return [self]
        else:
            return self.previous.path() + [self]

    def traverse_to(self, coq: coq_serapy.SerapiInstance, initial_history_len: int) -> None:
        # Get both the current and target histories
        full_cur_history = coq.tactic_history.getFullHistory()[initial_history_len:]
        full_node_history = [item for replay_node in self.path()[1:]
                             for item in [replay_node.prediction.prediction] + replay_node.postfix]
        # Get the number of commands common to the beginning of the current
        # history and the history of the target node
        common_prefix_len = 0
        for item1, item2, in zip(full_node_history, full_cur_history):
            if item1 != item2:
                break
            common_prefix_len += 1
        # Return to the place where the current history and the history of
        # the target node diverged.
        while len(coq.tactic_history.getFullHistory()) > initial_history_len + common_prefix_len:
            coq.cancel_last()
        # Run the next nodes history from that point.
        for cmd in full_node_history[common_prefix_len:]:
            coq.run_stmt(cmd)
        return


def contextInHistory(full_context: ProofContext, node: BFSNode):
    return any([coq_serapy.contextSurjective(full_context,
                                                  n.context_before.obligations)
                for n in node.path()[1:]])

def get_leaf_descendents(node: BFSNode) -> List[BFSNode]:
    if len(node.children) == 0:
        return [node]
    return [node for nodelist in [get_leaf_descendents(node) for node in node.children]
            for node in nodelist]

def get_prunable_nodes(node: BFSNode) -> List[BFSNode]:
    num_closes = len([cmd for cmd in node.postfix if cmd == "}"])
    if num_closes == 0:
        return []
    num_opens = len([cmd for cmd in node.postfix if cmd == "{"])
    significant_parent = node
    while num_opens < num_closes and significant_parent.previous is not None:
        num_opens += len([cmd for cmd in significant_parent.previous.postfix if cmd == "{"])
        num_closes += len([cmd for cmd in significant_parent.previous.postfix if cmd == "}"])
        significant_parent = significant_parent.previous

    return [leaf for leaf in get_leaf_descendents(significant_parent) if leaf != node]

def bfs_beam_proof_search(lemma_name: str,
                          module_prefix: str,
                          relevant_lemmas: List[str],
                          coq: coq_serapy.SerapiInstance,
                          args: argparse.Namespace,
                          bar_idx: int,
                          predictor: TacticPredictor) \
                          -> SearchResult:
    hasUnexploredNode = False
    graph_file = f"{args.output_dir}/{module_prefix}{lemma_name}.svg"

    features_extractor = FeaturesExtractor(args.tactics_file, args.tokens_file)
    if args.scoring_function == "lstd":
        state_estimator = Estimator(args.beta_file)
    elif args.scoring_function == "pickled":
        with args.pickled_estimator.open('rb') as f:
            john_model = pickle.load(f)

    initial_history_len = len(coq.tactic_history.getFullHistory())
    start_node = BFSNode(Prediction(lemma_name, 1.0), 1.0, 0.0, [],
                         FullContext([], [],
                                     ProofContext([], [], [], [])), None)
    search_start_node = start_node
    if args.search_prefix:
        for command in coq_serapy.read_commands(args.search_prefix):
            full_context_before = FullContext(relevant_lemmas,
                                              coq.prev_tactics,
                                              unwrap(coq.proof_context))
            search_start_node = BFSNode(Prediction(command, 1.0), 1.0, 0.0, [],
                                 full_context_before, search_start_node)
    if coq.count_fg_goals() > 1:
        coq.run_stmt("{")
        subgoals_stack_start = [0]
    else:
        subgoals_stack_start = []
    nodes_todo: List[Tuple[BFSNode, List[int], int]] = \
        [(search_start_node, subgoals_stack_start, 0)]

    total_nodes = numNodesInTree(args.search_width,
                                 args.search_depth + 2) - 1
    with tqdm(total=total_nodes, unit="pred", file=sys.stdout,
              desc=lemma_name, disable=(not args.progress),
              leave=False,
              position=bar_idx + 1,
              dynamic_ncols=True, bar_format=mybarfmt) as pbar:
        while len(nodes_todo) > 0:
            next_nodes_todo: List[Tuple[BFSNode, List[int], int]] = []
            while len(nodes_todo) > 0:
                next_node, subgoal_distance_stack, extra_depth = nodes_todo.pop()
                pbar.update()
                next_node.traverse_to(coq, initial_history_len)

                full_context_before = FullContext(relevant_lemmas,
                                                  coq.prev_tactics,
                                                  unwrap(coq.proof_context))
                num_successful_predictions = 0
                predictions = predictor.predictKTactics(
                    truncate_tactic_context(full_context_before.as_tcontext(),
                                            args.max_term_length),
                            args.max_attempts)
                for prediction in predictions:
                    if num_successful_predictions >= args.search_width:
                        break
                    context_after, num_stmts, \
                        subgoals_closed, subgoals_opened, \
                        error, time_taken, unshelved = \
                        tryPrediction(args, coq, prediction.prediction,
                                      next_node.total_time())

                    postfix = []
                    if unshelved:
                        postfix.append("Unshelve.")
                    postfix += ["}"] * subgoals_closed
                    postfix += ["{"] * subgoals_opened


                    prediction_node = BFSNode(
                        prediction,
                        0,
                        time_taken, postfix, full_context_before, next_node)
                    if error:
                        if args.count_failing_predictions:
                            num_successful_predictions += 1
                        prediction_node.setNodeColor("red")
                        continue
                    if contextIsBig(context_after) or \
                            contextInHistory(context_after, prediction_node):
                        if args.count_softfail_predictions:
                            num_successful_predictions += 1
                        eprint(f"Prediction in history or too big", guard=args.verbose >= 2)
                        prediction_node.setNodeColor("orange")
                        for _ in range(num_stmts):
                            coq.cancel_last()
                        continue
                    if len(coq.proof_context.all_goals) > args.max_subgoals:
                        if args.count_softfail_predictions:
                            num_successful_predictions += 1
                        prediction_node.setNodeColor("orange")
                        for _ in range(num_stmts):
                            coq.cancel_last()
                        continue
                    if completed_proof(coq):
                        prediction_node.mkQED()
                        start_node.draw_graph(graph_file)
                        return SearchResult(SearchStatus.SUCCESS,
                                            prediction_node.interactions()[1:])

                    if args.scoring_function == "certainty":
                        prediction_node.score = next_node.score * prediction.certainty
                    elif args.scoring_function == "pickled":
                        assert sys.version_info >= (3, 10), "Pickled estimators only supported in python 3.10 or newer"
                        score = 0.
                        for idx, goal in enumerate(coq.get_all_sexp_goals()):
                            try:
                                score += -float(john_model.predict(Lemma("", goal)))
                            except UnhandledExpr:
                                print(f"Couldn't handle goal {coq.proof_context.all_goals[idx]}")
                                raise
                        prediction_node.score = score
                    elif args.scoring_function == "const":
                        prediction_node.score = 1.0
                    else:
                        assert args.scoring_function == "lstd"
                        prediction_node.score = state_estimator.estimateVal(
                                          features_extractor.state_features(
                                              TacticContext(full_context_before.relevant_lemmas,
                                                            full_context_before.prev_tactics,
                                                            context_after.focused_hyps,
                                                            context_after.focused_goal)))

                    num_successful_predictions += 1

                    if subgoals_closed > 0:
                        prediction_node.setNodeColor("blue")
                        # Prune unexplored nodes from the tree that are trying to
                        # solve the subgoal(s) we just solved.
                        prunable_nodes = get_prunable_nodes(prediction_node)
                        # Prune them from nodes_todo, which are nodes at the
                        # current level which we haven't explored yet.
                        nodes_todo = [node for node in nodes_todo if node[0] not in prunable_nodes]
                        # Prune them from next_nodes_todo, which are new children
                        # of nodes at the current level which we already explored.
                        next_nodes_todo = [node for node in next_nodes_todo if node[0] not in prunable_nodes]

                    # ### 1.
                    if subgoal_distance_stack:
                        new_distance_stack = (subgoal_distance_stack[:-1] +
                                              [subgoal_distance_stack[-1]+1])
                    else:
                        new_distance_stack = []

                    # ### 2.
                    new_extra_depth = extra_depth
                    for _ in range(subgoals_closed):
                        closed_goal_distance = new_distance_stack.pop()
                        new_extra_depth += closed_goal_distance

                    # ### 3.
                    new_distance_stack += [0] * subgoals_opened

                    next_nodes_todo.append((prediction_node, new_distance_stack,
                                            new_extra_depth))

                    for _ in range(num_stmts):
                        coq.cancel_last()
                    if subgoals_closed > 0:
                        break
            next_nodes_todo.sort(key=lambda n: n[0].score, reverse=True)
            while len(nodes_todo) < args.beam_width and len(next_nodes_todo) > 0:
                next_node, subgoal_distance_stack, extra_depth = next_nodes_todo.pop(0)
                if len(next_node.path()) <= args.search_depth + extra_depth:
                    nodes_todo.append((next_node, subgoal_distance_stack, extra_depth))
                else:
                    hasUnexploredNode = True

    start_node.draw_graph(graph_file)
    if hasUnexploredNode:
        return SearchResult(SearchStatus.INCOMPLETE, None)
    else:
        return SearchResult(SearchStatus.FAILURE, None)

@dataclass(order=True)
class AStarTask:
    f_score: float
    node: BFSNode=field(compare=False)


def best_first_proof_search(lemma_name: str,
                       module_prefix: Optional[str],
                       relevant_lemmas: List[str],
                       coq: coq_serapy.SerapiInstance,
                       args: argparse.Namespace,
                       bar_idx: int,
                       predictor: TacticPredictor) \
                       -> SearchResult:
    assert args.scoring_function in ["pickled", "const"] or args.search_type != "astar", "only pickled and const scorers are currently compatible with A* search"
    if args.scoring_function == "pickled":
        with args.pickled_estimator.open('rb') as f:
            john_model = pickle.load(f)
    graph_file = f"{args.output_dir}/{module_prefix}{lemma_name}.svg"
    initial_history_len = len(coq.tactic_history.getFullHistory())
    start_node = BFSNode(Prediction(lemma_name, 1.0), 1.0, 0.0, [],
                         FullContext([], [],
                                     ProofContext([], [], [], [])), None)
    search_start_node = start_node
    if args.search_prefix:
        for command in coq_serapy.read_commands(args.search_prefix):
            full_context_before = FullContext(relevant_lemmas,
                                              coq.prev_tactics,
                                              unwrap(coq.proof_context))
            search_start_node = BFSNode(Prediction(command, 1.0), 1.0, 0.0, [],
                                        full_context_before, search_start_node)
    nodes_todo: List[AStarTask] = [AStarTask(1.0, search_start_node)]

    desc_name = lemma_name
    if len(desc_name) > 25:
        desc_name = desc_name[:22] + "..."
    for _step in trange(args.astar_steps, unit="pred", file=sys.stdout,
                       desc=desc_name, disable=(not args.progress),
                       leave=False, position=bar_idx + 1,
                       dynamic_ncols=True, bar_format=mybarfmt):
        if len(nodes_todo) == 0:
            break
        next_node = heapq.heappop(nodes_todo)
        next_node.node.traverse_to(coq, initial_history_len)

        full_context_before = FullContext(relevant_lemmas,
                                          coq.prev_tactics,
                                          unwrap(coq.proof_context))
        num_successful_predictions = 0
        predictions = predictor.predictKTactics(
            truncate_tactic_context(full_context_before.as_tcontext(),
                                    args.max_term_length),
            args.max_attempts)

        for prediction in predictions:
            if num_successful_predictions >= args.search_width:
                break
            context_after, num_stmts, \
                subgoals_closed, subgoals_opened, \
                error, time_taken, unshelved = \
                tryPrediction(args, coq, prediction.prediction,
                             next_node.node.total_time())

            postfix = []
            if unshelved:
                postfix.append("Unshelve.")
            postfix += ["}"] * subgoals_closed
            postfix += ["{"] * subgoals_opened

            prediction_node = BFSNode(
                prediction,
                0,
                time_taken, postfix, full_context_before, next_node.node)
            if error:
                if args.count_failing_predictions:
                    num_successful_predictions += 1
                prediction_node.setNodeColor("red")
                continue
            else:
                num_successful_predictions += 1
            # Check if we've gone in circles
            if contextInHistory(context_after, prediction_node):
                if args.count_softfail_predictions:
                    num_successful_predictions += 1
                eprint(f"Prediction in history", guard=args.verbose >= 2)
                prediction_node.setNodeColor("orange")
                for _ in range(num_stmts):
                    coq.cancel_last()
                continue
            # Check if the resulting context is too big
            if len(coq.proof_context.all_goals) > args.max_subgoals or \
              contextIsBig(context_after):
                if args.count_softfail_predictions:
                    num_successful_predictions += 1
                prediction_node.setNodeColor("orange")
                for _ in range(num_stmts):
                    coq.cancel_last()
                continue
            # Check if the proof is done
            if completed_proof(coq):
                prediction_node.mkQED()
                start_node.draw_graph(graph_file)
                return SearchResult(SearchStatus.SUCCESS,
                                    prediction_node.interactions()[1:])
            if args.scoring_function == "const":
                h_score = 1.
            elif args.scoring_function == "certainty":
                h_score = -abs(next_node.f_score * prediction.certainty)
            elif args.scoring_function == "norm-certainty":
                h_score = -math.sqrt(abs(next_node.f_score * prediction.certainty))
            else:
                assert args.scoring_function == "pickled"
                h_score = 0.
                for idx, goal in enumerate(coq.get_all_sexp_goals()):
                    try:
                        h_score += john_model.predict(Lemma("", goal))
                    except UnhandledExpr:
                        print(f"Goal failed to be handled: {coq.proof_context.all_goals[idx]}")
                        raise
            if args.search_type == "astar":
                # Calculate the A* f_score
                g_score = len(prediction_node.path())
                score = g_score + h_score
            else:
                score = h_score

            prediction_node.score = score

            # Put our new prediction node in our priority queue
            heapq.heappush(nodes_todo, AStarTask(score, prediction_node))
            # Return us to before running the prediction, so we're ready for
            # the next one.
            for _ in range(num_stmts):
                coq.cancel_last()
            # If we solved the subgoal...
            if subgoals_closed > 0:
                prediction_node.setNodeColor("blue")
                # Get unexplored nodes from the tree that are trying to
                # solve the subgoal(s) we just solved.
                prunable_nodes = get_prunable_nodes(prediction_node)
                # Prune them from the frontier nodes
                nodes_todo = [node for node in nodes_todo
                              if node.node not in prunable_nodes]
                heapq.heapify(nodes_todo)
                # Don't run the rest of the predictions at this state
                break

    hasUnexploredNode = len(nodes_todo) > 0
    start_node.draw_graph(graph_file)
    if hasUnexploredNode:
        return SearchResult(SearchStatus.INCOMPLETE, None)
    else:
        return SearchResult(SearchStatus.FAILURE, None)
