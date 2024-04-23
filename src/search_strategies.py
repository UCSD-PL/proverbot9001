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
import random
from random import shuffle
from collections import Counter
import copy

import pygraphviz as pgv
from tqdm import tqdm, trange

if sys.version_info >= (3, 10):
    from lemma_models import Lemma, UnhandledExpr

import coq_serapy
from coq_serapy.contexts import TacticContext, FullContext, ProofContext, truncate_tactic_context
import tokenizer
from models.tactic_predictor import Prediction, TacticPredictor
from models.features_polyarg_predictor import FeaturesPolyargPredictor
from search_results import TacticInteraction, SearchResult, SearchStatus
from util import nostderr, unwrap, eprint, mybarfmt, copyArgs, print_time

from value_estimator import Estimator
import dataloader
from train_my_rnn_model import zhannRNN
import coq2vec

unnamed_goal_number: int = 0


class FeaturesExtractor:
    tactic_map: Dict[str, int]
    token_map: Dict[str, int]
    _num_tactics: int
    _num_tokens: int

    def __init__(self, common_tactic_stems: List[str], common_tokens: List[str]) -> None:
        self.tactic_map = {}
        self.token_map = {}
        for idx, tactic_stem in enumerate(common_tactic_stems, start=2):
            self.tactic_map[tactic_stem] = idx
        for idx, token in enumerate(common_tokens, start=2):
            self.token_map[token] = idx
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
    subgoals_opened: int
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
        self.start_node = self.mkNode(Prediction(lemma_name, 1.0, 1.0),
                                      FullContext(
                                          [], [], ProofContext([], [], [], [])),
                                      0,
                                      None)
        self.start_node.time_taken = 0.0
        if features_json:
            self.feature_extractor = FeaturesExtractor(str(tactics_file),
                                                       str(tokens_file))
        pass

    def mkNode(self, prediction: Prediction, context_before: FullContext, subgoals_opened: int,
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
                              tooltip=tooltip.replace("\\", "\\\\"),
                              **kwargs)
        self.__next_node_id += 1
        newNode = LabeledNode(prediction.prediction, prediction.certainty,
                              None, self.__next_node_id-1,
                              context_before, subgoals_opened, previous_node, [])
        if previous_node:
            self.__graph.add_edge(previous_node.node_id,
                                  newNode.node_id, **kwargs)
            previous_node.children.append(newNode)
        return newNode

    def mkQED(self, predictionNode: LabeledNode):
        self.mkNode(Prediction("QED", 1.0, 1.0), FullContext(
            [], [], ProofContext([], [], [], [])), 
                    0,
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
    steps_explored: int


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
    except (coq_serapy.CoqTimeoutError, coq_serapy.ParseError,
            coq_serapy.CoqExn, coq_serapy.CoqOverflowError,
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

def dfs_subgoal_sharing(lemma_name: str,
                                module_prefix: str,
                                relevant_lemmas: List[str],
                                coq: coq_serapy.SerapiInstance,
                                output_dir: Path,
                                args: argparse.Namespace,
                                bar_idx: int,
                                predictor_list: [TacticPredictor]) \
                                -> SearchResult:

    # two lines below for fast finish
    g = SearchGraph(args.tactics_file, args.tokens_file, lemma_name,
                    args.features_json)
    g.draw(f"{output_dir}/{module_prefix}{lemma_name}.svg")
    return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, 0)

    def cleanupSearch(num_stmts: int, msg: Optional[str] = None):
        if msg:
            eprint(f"Cancelling {num_stmts} statements "
                   f"because {msg}.", guard=args.verbose >= 2)
        for _ in range(num_stmts):
            coq.cancel_last()
    hasUnexploredNode = False

    def search_vote(pbar: tqdm, current_path: List[LabeledNode],
               subgoal_distance_stack: List[int],
               extra_depth: int, steps_explored: int, curr_predictor_list: [TacticPredictor], init_subgoals_seen: {}) -> (SubSearchResult, dict, SearchGraph):


        eprint("the predictor list")
        eprint(curr_predictor_list)
        subgoals_seen = init_subgoals_seen

        total_subgoals_opened = 1
        times_subgoals_seen = 0
        subgoals_opened = 0
        nonlocal hasUnexploredNode
        nonlocal relevant_lemmas
        global unnamed_goal_number
        full_context_before = FullContext(relevant_lemmas,
                                          coq.prev_tactics,
                                          unwrap(coq.proof_context))
        shuffle(curr_predictor_list)
        predictions_lists = []
        for curr_predictor in range(len(curr_predictor_list)):
            apredictor = curr_predictor_list[curr_predictor]
            predictions_temp = apredictor.predictKTactics(args,
                truncate_tactic_context(full_context_before.as_tcontext(), args.max_term_length),
                args.max_attempts,
                blacklist=args.blacklisted_tactics)
            predictions_lists.append(predictions_temp)
        predictions = []
        only_tactics = tuple()
        full_tup = tuple()
        for tup in zip(*predictions_lists):
            full_tup = full_tup + tup
            only_tactics = only_tactics + tuple([full_pred[0] for full_pred in tup])
            mostcommon = Counter(only_tactics).most_common()[0]
            if mostcommon[1] > 1:
                best_tactic = mostcommon[0]
            else:
                highest_sureness = 0
                best_tactic = None
                for tactic_and_sureness in full_tup:
                    if tactic_and_sureness[2] > highest_sureness:
                        highest_sureness = tactic_and_sureness[2]
                        best_tactic = tactic_and_sureness[0]
            predictions.append(Prediction(best_tactic, 1.0, 1.0))
            only_tactics = tuple([x for x in only_tactics if not x==best_tactic])
            full_tup = tuple([x for x in full_tup if not x[0]==best_tactic])


        #assert len(predictions) == args.max_attempts
        if coq.use_hammer:
            predictions = [Prediction(prediction.prediction[:-1] + "; try hammer.",
                                      prediction.certainty, prediction.no_softmax_certainty)
                           for prediction in predictions]

        assert len(predictions) == args.max_attempts
        if coq.use_hammer:
            predictions = [Prediction(prediction.prediction[:-1] + "; try hammer.",
                                      prediction.certainty, prediction.no_softmax_certainty)
                           for prediction in predictions]
        num_successful_predictions = 0
        substeps_explored = 1
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
                                                  subgoals_opened,
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
                                          subgoals_opened,
                                          current_path[-1])
                predictionNode.time_taken = time_taken
                if unshelved:
                    predictionNode = g.mkNode(Prediction("Unshelve.", 1.0, 1.0),
                                              full_context_before,
                                              subgoals_opened,
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
                ####
                if subgoals_opened > 0:
                    total_subgoals_opened += 1
                    current_goal = predictionNode.context_before.as_tcontext().goal
                    hashed_goal = hash(current_goal)
                    solved_already = subgoals_seen.get(hashed_goal)
                    if solved_already is not None:
                        g.setNodeColor(predictionNode, "purple")
                        times_subgoals_seen += 1
                        for step in solved_already:
                            try:
                                context_after, tmp_num_stmts, \
                                    tmp_subgoals_closed, tmp_subgoals_opened, \
                                    error, time_taken, unshelved = \
                                    tryPrediction(args, coq, step,
                                                  time_on_path(current_path[-1]))
                                num_stmts += tmp_num_stmts
                                if error:
                                    continue
                                postfix = []
                                if unshelved:
                                    postfix.append("Unshelve.")
                                postfix += ["}"] * tmp_subgoals_closed
                                postfix += ["{"] * tmp_subgoals_opened
                                full_context_after = FullContext(relevant_lemmas, coq.prev_tactics,unwrap(coq.proof_context))
                                # ### 1.
                                #if subgoal_distance_stack:
                                #    new_distance_stack = (subgoal_distance_stack[:-1] +
                                #                          [subgoal_distance_stack[-1]+1])
                                #else:
                                #    new_distance_stack = []

                                # ### 2.
                                for _ in range(tmp_subgoals_closed):
                                    closed_goal_distance = new_distance_stack.pop()
                                    new_extra_depth += closed_goal_distance

                                # ### 3.
                                new_distance_stack += [0] * tmp_subgoals_opened

                                current_path  = current_path + [predictionNode]
                                predictionNode_tmp = g.mkNode(Prediction(step, 1.0, 1.0),
                                                          full_context_after,
                                                          subgoals_opened,
                                                          predictionNode)
                                predictionNode = predictionNode_tmp
                                predictionNode.time_taken = time_taken
                            except coq_serapy.CoqAnomaly:
                                if lemma_name == "":
                                    eprint("encountered unnamed goal!")
                                else:
                                    eprint("coqanomaly without unnamed goal!")
                                raise
                #############
                if completed_proof(coq):
                    solution = g.mkQED(predictionNode)
                    return (SubSearchResult(solution, subgoals_closed, steps_explored + substeps_explored), subgoals_seen, g)
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
                        and len(current_path) < args.hard_depth_limit \
                        and (args.max_steps is None or
                             substeps_explored < args.max_steps):
                    if subgoals_closed > 0:
                        g.setNodeColor(predictionNode, "blue")
                        tmp_subgoals_closed = subgoals_closed
                        subgoal_solution = [predictionNode.prediction]
                        subgoal_node = predictionNode
                        while tmp_subgoals_closed > 0: #TODO: add protections
                            closed_goal = subgoal_node.context_before.as_tcontext().goal
                            subgoal_node = subgoal_node.previous
                            while not subgoal_node.subgoals_opened > 0:
                                subgoal_solution.append(subgoal_node.prediction)
                                #for postfixbit in subgoal_node.postfix:
                                #    subgoal_solution.append(postfixbit)
                                subgoal_node = subgoal_node.previous
                            hashed_goal = hash(closed_goal)
                            subgoal_solution.reverse()
                            subgoals_seen[hashed_goal] = subgoal_solution
                            tmp_subgoals_closed = tmp_subgoals_closed - 1
                            subgoal_solution.reverse()
                    
                    (sub_search_result, tmp_subgoals_seen, _) = search_vote(pbar,
                                               current_path + [predictionNode],
                                               new_distance_stack,
                                               new_extra_depth, steps_explored + substeps_explored, curr_predictor_list, subgoals_seen)
                    subgoals_seen.update(tmp_subgoals_seen)
                    substeps_explored += sub_search_result.steps_explored
                    cleanupSearch(num_stmts, "we finished subsearch")
                    if sub_search_result.solution or \
                       sub_search_result.solved_subgoals > subgoals_opened:
                        new_subgoals_closed = \
                            subgoals_closed + \
                            sub_search_result.solved_subgoals - \
                            subgoals_opened
                        if times_subgoals_seen > 0:
                            eprint("times subgoals seen")
                            eprint(times_subgoals_seen)
                            eprint("times subgoals opened")
                            eprint(total_subgoals_opened)
                        return (SubSearchResult(sub_search_result.solution,
                                               new_subgoals_closed, substeps_explored), subgoals_seen, g)
                    if subgoals_closed > 0:
                        if times_subgoals_seen > 0:
                            eprint("times subgoals seen")
                            eprint(times_subgoals_seen)
                            eprint("times subgoals opened")
                            eprint(total_subgoals_opened)
                        return (SubSearchResult(None, subgoals_closed, substeps_explored), subgoals_seen, g)
                else:
                    hasUnexploredNode = True
                    cleanupSearch(num_stmts, "we hit the depth limit")
                    if subgoals_closed > 0:
                        # depth = (args.search_depth + new_extra_depth + 1) \
                        #     - len(current_path)
                        if times_subgoals_seen > 0:
                            eprint("times subgoals seen")
                            eprint(times_subgoals_seen)
                            eprint("times subgoals opened")
                            eprint(total_subgoals_opened)
                        return (SubSearchResult(None, subgoals_closed, substeps_explored), subgoals_seen, g)
            except coq_serapy.CoqAnomaly:
                predictionNode = g.mkNode(prediction,
                                          full_context_before,
                                          subgoals_opened,
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
        if times_subgoals_seen > 0:
            eprint("times subgoals seen")
            eprint(times_subgoals_seen)
            eprint("times subgoals opened")
            eprint(total_subgoals_opened)
        return (SubSearchResult(None, 0, substeps_explored), subgoals_seen, g)

    def search(pbar: tqdm, current_path: List[LabeledNode],
               subgoal_distance_stack: List[int],
               extra_depth: int, steps_explored: int, curr_predictor: TacticPredictor, init_subgoals_seen: {}) -> (SubSearchResult, dict, SearchGraph):



        subgoals_seen = init_subgoals_seen

        total_subgoals_opened = 1
        times_subgoals_seen = 0
        subgoals_opened = 0
        nonlocal hasUnexploredNode
        nonlocal relevant_lemmas
        global unnamed_goal_number
        full_context_before = FullContext(relevant_lemmas,
                                          coq.prev_tactics,
                                          unwrap(coq.proof_context))
        

        #round_robin = 0
        #shuffle(predictor_list)
        #predictions_lists = []
        #for curr_predictor in range(len(predictor_list)):
        #    apredictor = predictor_list[curr_predictor]
        #    temp_predictions = apredictor.predictKTactics(args,
        #        truncate_tactic_context(full_context_before.as_tcontext(), args.max_term_length),
        #        args.max_attempts,
        #        blacklist=args.blacklisted_tactics)
        #    predictions_lists.append(temp_predictions)
        #all_predictions = [eachlist[i] for eachlist in predictions_lists for i in range(round(args.max_attempts/len(predictions_lists))+1)]
        #predictions = all_predictions[:args.max_attempts]
        predictions = curr_predictor.predictKTactics(args,
           truncate_tactic_context(full_context_before.as_tcontext(),
                                    args.max_term_length),
            args.max_attempts,
            blacklist=args.blacklisted_tactics)
        assert len(predictions) == args.max_attempts
        if coq.use_hammer:
            predictions = [Prediction(prediction.prediction[:-1] + "; try hammer.",
                                      prediction.certainty, prediction.no_softmax_certainty)
                           for prediction in predictions]
        num_successful_predictions = 0
        substeps_explored = 1
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
                                                  subgoals_opened,
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
                                          subgoals_opened,
                                          current_path[-1])
                predictionNode.time_taken = time_taken
                if unshelved:
                    predictionNode = g.mkNode(Prediction("Unshelve.", 1.0, 1.0),
                                              full_context_before,
                                              subgoals_opened,
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
                ####
                if subgoals_opened > 0:
                    total_subgoals_opened += 1
                    current_goal = predictionNode.context_before.as_tcontext().goal
                    hashed_goal = hash(current_goal)
                    solved_already = subgoals_seen.get(hashed_goal)
                    if solved_already is not None:
                        g.setNodeColor(predictionNode, "purple")
                        times_subgoals_seen += 1
                        for step in solved_already:
                            try:
                                context_after, tmp_num_stmts, \
                                    tmp_subgoals_closed, tmp_subgoals_opened, \
                                    error, time_taken, unshelved = \
                                    tryPrediction(args, coq, step,
                                                  time_on_path(current_path[-1]))
                                num_stmts += tmp_num_stmts
                                if error:
                                    continue
                                postfix = []
                                if unshelved:
                                    postfix.append("Unshelve.")
                                postfix += ["}"] * tmp_subgoals_closed
                                postfix += ["{"] * tmp_subgoals_opened
                                full_context_after = FullContext(relevant_lemmas, coq.prev_tactics,unwrap(coq.proof_context))
                                # ### 1.
                                #if subgoal_distance_stack:
                                #    new_distance_stack = (subgoal_distance_stack[:-1] +
                                #                          [subgoal_distance_stack[-1]+1])
                                #else:
                                #    new_distance_stack = []

                                # ### 2.
                                for _ in range(tmp_subgoals_closed):
                                    closed_goal_distance = new_distance_stack.pop()
                                    new_extra_depth += closed_goal_distance

                                # ### 3.
                                new_distance_stack += [0] * tmp_subgoals_opened

                                current_path  = current_path + [predictionNode]
                                predictionNode_tmp = g.mkNode(Prediction(step, 1.0, 1.0),
                                                          full_context_after,
                                                          subgoals_opened,
                                                          predictionNode)
                                predictionNode = predictionNode_tmp
                                predictionNode.time_taken = time_taken
                            except coq_serapy.CoqAnomaly:
                                if lemma_name == "":
                                    eprint("encountered unnamed goal!")
                                else:
                                    eprint("coqanomaly without unnamed goal!")
                                raise
                #############
                if completed_proof(coq):
                    solution = g.mkQED(predictionNode)
                    return (SubSearchResult(solution, subgoals_closed, steps_explored + substeps_explored), subgoals_seen, g)
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
                        and len(current_path) < args.hard_depth_limit \
                        and (args.max_steps is None or
                             substeps_explored < args.max_steps):
                    if subgoals_closed > 0:
                        g.setNodeColor(predictionNode, "blue")
                        tmp_subgoals_closed = subgoals_closed
                        subgoal_solution = [predictionNode.prediction]
                        subgoal_node = predictionNode
                        while tmp_subgoals_closed > 0: #TODO: add protections
                            closed_goal = subgoal_node.context_before.as_tcontext().goal
                            subgoal_node = subgoal_node.previous
                            while not subgoal_node.subgoals_opened > 0:
                                subgoal_solution.append(subgoal_node.prediction)
                                #for postfixbit in subgoal_node.postfix:
                                #    subgoal_solution.append(postfixbit)
                                subgoal_node = subgoal_node.previous
                            hashed_goal = hash(closed_goal)
                            subgoal_solution.reverse()
                            subgoals_seen[hashed_goal] = subgoal_solution
                            tmp_subgoals_closed = tmp_subgoals_closed - 1
                            subgoal_solution.reverse()
                    
                    (sub_search_result, tmp_subgoals_seen, _) = search(pbar,
                                               current_path + [predictionNode],
                                               new_distance_stack,
                                               new_extra_depth, steps_explored + substeps_explored, curr_predictor, subgoals_seen)
                    subgoals_seen.update(tmp_subgoals_seen)
                    substeps_explored += sub_search_result.steps_explored
                    cleanupSearch(num_stmts, "we finished subsearch")
                    if sub_search_result.solution or \
                       sub_search_result.solved_subgoals > subgoals_opened:
                        new_subgoals_closed = \
                            subgoals_closed + \
                            sub_search_result.solved_subgoals - \
                            subgoals_opened
                        if times_subgoals_seen > 0:
                            eprint("times subgoals seen")
                            eprint(times_subgoals_seen)
                            eprint("times subgoals opened")
                            eprint(total_subgoals_opened)
                        return (SubSearchResult(sub_search_result.solution,
                                               new_subgoals_closed, substeps_explored), subgoals_seen, g)
                    if subgoals_closed > 0:
                        if times_subgoals_seen > 0:
                            eprint("times subgoals seen")
                            eprint(times_subgoals_seen)
                            eprint("times subgoals opened")
                            eprint(total_subgoals_opened)
                        return (SubSearchResult(None, subgoals_closed, substeps_explored), subgoals_seen, g)
                else:
                    hasUnexploredNode = True
                    cleanupSearch(num_stmts, "we hit the depth limit")
                    if subgoals_closed > 0:
                        # depth = (args.search_depth + new_extra_depth + 1) \
                        #     - len(current_path)
                        if times_subgoals_seen > 0:
                            eprint("times subgoals seen")
                            eprint(times_subgoals_seen)
                            eprint("times subgoals opened")
                            eprint(total_subgoals_opened)
                        return (SubSearchResult(None, subgoals_closed, substeps_explored), subgoals_seen, g)
            except coq_serapy.CoqAnomaly:
                predictionNode = g.mkNode(prediction,
                                          full_context_before,
                                          subgoals_opened,
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
        if times_subgoals_seen > 0:
            eprint("times subgoals seen")
            eprint(times_subgoals_seen)
            eprint("times subgoals opened")
            eprint(total_subgoals_opened)
        return (SubSearchResult(None, 0, substeps_explored), subgoals_seen, g)

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
        prep_subgoals_seen = {}
        command_list_lists = []
        total_steps_lists = []
        graph_lists = []
        tmp_pred_list = copy.deepcopy(predictor_list)
        predictor_list.append(tmp_pred_list)
        for curr_predictor in predictor_list:
            eprint("new predictor")
            g = SearchGraph(args.tactics_file, args.tokens_file, lemma_name,
                            args.features_json)
            next_node = g.start_node
            if args.search_prefix is not None:
                for command in coq_serapy.read_commands(args.search_prefix):
                    full_context_before = FullContext(relevant_lemmas,
                                                  coq.prev_tactics,
                                                  unwrap(coq.proof_context))
                    next_node = g.mkNode(Prediction(command, 1.0, 1.0),
                                     full_context_before,
                                     subgoals_opened,
                                     next_node)
                next_node.time_taken = 0.0
                coq.run_stmt(command)
            stack_start = copy.deepcopy(subgoals_stack_start)
            if isinstance(curr_predictor, list):
                (sub_search_result, temp_subgoals_seen, graph) = search_vote(pbar, [next_node], stack_start, 0, 0, curr_predictor, prep_subgoals_seen)
            else:
                (sub_search_result, temp_subgoals_seen, graph) = search(pbar, [next_node], stack_start, 0, 0, curr_predictor, prep_subgoals_seen)
            command_list_one = sub_search_result.solution
            total_steps_one = sub_search_result.steps_explored
            if command_list_one:
                graph.draw(f"{output_dir}/{module_prefix}{lemma_name}.svg")
                if args.features_json:
                    graph.write_feat_json(f"{output_dir}/{module_prefix}"
                                      f"{lemma_name}.json")
                return SearchResult(SearchStatus.SUCCESS, relevant_lemmas, command_list_one, total_steps_one)
            graph.draw(f"{output_dir}/{module_prefix}{lemma_name}.svg")
            prep_subgoals_seen.update(temp_subgoals_seen)
            eprint("subgoals seen")
            eprint(prep_subgoals_seen)
            if args.search_prefix is not None:
                cleanup_stmts = 0
                for command in coq_serapy.read_commands(args.search_prefix):
                    cleanup_stmts += 1
                cleanupSearch(cleanup_stmts)
        pbar.clear()
    graph.draw(f"{output_dir}/{module_prefix}{lemma_name}.svg")
    if args.features_json:
        graph.write_feat_json(f"{output_dir}/{module_prefix}"
                          f"{lemma_name}.json")
    if hasUnexploredNode:
        return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, total_steps_one)
    return SearchResult(SearchStatus.FAILURE, relevant_lemmas, None, total_steps_one)

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

    # two lines below for fast finish
    g.draw(f"{output_dir}/{module_prefix}{lemma_name}.svg")
    return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, 0)

    def cleanupSearch(num_stmts: int, msg: Optional[str] = None):
        if msg:
            eprint(f"Cancelling {num_stmts} statements "
                   f"because {msg}.", guard=args.verbose >= 2)
        for _ in range(num_stmts):
            coq.cancel_last()

    def search(pbar: tqdm, current_path: List[LabeledNode],
               subgoal_distance_stack: List[int],
               extra_depth: int, steps_explored: int) -> SubSearchResult:
        nonlocal hasUnexploredNode
        nonlocal relevant_lemmas
        global unnamed_goal_number
        full_context_before = FullContext(relevant_lemmas,
                                          coq.prev_tactics,
                                          unwrap(coq.proof_context))

        #round_robin = 0
        #shuffle(predictor_list)
        #predictions_lists = []
        #for curr_predictor in range(len(predictor_list)):
        #    apredictor = predictor_list[curr_predictor]
        #    temp_predictions = apredictor.predictKTactics(args,
        #        truncate_tactic_context(full_context_before.as_tcontext(), args.max_term_length),
        #        args.max_attempts,
        #        blacklist=args.blacklisted_tactics)
        #    predictions_lists.append(temp_predictions)
        #all_predictions = [eachlist[i] for eachlist in predictions_lists for i in range(round(args.max_attempts/len(predictions_lists))+1)]
        #predictions = all_predictions[:args.max_attempts]
        predictions = predictor.predictKTactics(args,
           truncate_tactic_context(full_context_before.as_tcontext(),
                                    args.max_term_length),
            args.max_attempts,
            blacklist=args.blacklisted_tactics)
        assert len(predictions) == args.max_attempts
        if coq.use_hammer:
            predictions = [Prediction(prediction.prediction[:-1] + "; try hammer.",
                                      prediction.certainty, prediction.no_softmax_certainty)
                           for prediction in predictions]
        num_successful_predictions = 0
        substeps_explored = 1
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
                                          subgoals_opened,
                                          current_path[-1])
                predictionNode.time_taken = time_taken
                if unshelved:
                    predictionNode = g.mkNode(Prediction("Unshelve.", 1.0, 1.0),
                                              full_context_before,
                                              subgoals_opened,
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
                    return SubSearchResult(solution, subgoals_closed, steps_explored + substeps_explored)
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
                        and len(current_path) < args.hard_depth_limit \
                        and (args.max_steps is None or
                             substeps_explored < args.max_steps):
                    if subgoals_closed > 0:
                        g.setNodeColor(predictionNode, "blue")
                    sub_search_result = search(pbar,
                                               current_path + [predictionNode],
                                               new_distance_stack,
                                               new_extra_depth, steps_explored + substeps_explored)
                    substeps_explored += sub_search_result.steps_explored
                    cleanupSearch(num_stmts, "we finished subsearch")
                    if sub_search_result.solution or \
                       sub_search_result.solved_subgoals > subgoals_opened:
                        new_subgoals_closed = \
                            subgoals_closed + \
                            sub_search_result.solved_subgoals - \
                            subgoals_opened
                        return SubSearchResult(sub_search_result.solution,
                                               new_subgoals_closed, substeps_explored)
                    if subgoals_closed > 0:
                        return SubSearchResult(None, subgoals_closed, substeps_explored)
                else:
                    hasUnexploredNode = True
                    cleanupSearch(num_stmts, "we hit the depth limit")
                    if subgoals_closed > 0:
                        # depth = (args.search_depth + new_extra_depth + 1) \
                        #     - len(current_path)
                        return SubSearchResult(None, subgoals_closed, substeps_explored)
            except coq_serapy.CoqAnomaly:
                predictionNode = g.mkNode(prediction,
                                          full_context_before,
                                          subgoals_opened,
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
        return SubSearchResult(None, 0, substeps_explored)
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
        next_node = g.start_node
        if args.search_prefix is not None:
            for command in coq_serapy.read_commands(args.search_prefix):
                full_context_before = FullContext(relevant_lemmas,
                                                  coq.prev_tactics,
                                                  unwrap(coq.proof_context))
                next_node = g.mkNode(Prediction(command, 1.0, 1.0),
                                     full_context_before,
                                     subgoals_opened,
                                     next_node)
                next_node.time_taken = 0.0
                coq.run_stmt(command)
        hasUnexploredNode = False
        command_list, _, total_steps = search(pbar, [next_node], subgoals_stack_start, 0, 0)
        pbar.clear()
    g.draw(f"{output_dir}/{module_prefix}{lemma_name}.svg")
    if args.features_json:
        g.write_feat_json(f"{output_dir}/{module_prefix}"
                          f"{lemma_name}.json")
    if command_list:
        return SearchResult(SearchStatus.SUCCESS, relevant_lemmas, command_list, total_steps)
    if hasUnexploredNode:
        return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, total_steps)
    return SearchResult(SearchStatus.FAILURE, relevant_lemmas, None, total_steps)

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
    subgoals_opened: int

    def __init__(self, prediction: Prediction, score: float, time_taken: float,
                 postfix: List[str], context_before: FullContext, previous: Optional["BFSNode"],
                 color: Optional[str] = None, subgoals_opened: Optional[int] = 0) -> None:
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
        self.subgoals_opened = subgoals_opened
        pass

    def setSubgoalsOpened(self, opened: int) -> None:
        self.subgoals_opened = opened

    def setNodeColor(self, color: str) -> None:
        assert color
        if self.color != None and self.color != "":
            self.color = (unwrap(self.color) + ":" + color)
        else:
            self.color = color

    def mkQED(self) -> None:
        qed_node = BFSNode(Prediction("QED", 1.0, 1.0), 100, 0, [],
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
            label=f"{root.prediction.prediction}\nS:{root.score:.2e};C:{root.prediction.certainty:.2e}"
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
                           tooltip=tooltip.replace("\\", "\\\\"))

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
        curNode = self
        curPath = [self]
        while curNode.previous is not None:
            curNode = curNode.previous
            curPath.append(curNode)
        return list(reversed(curPath))

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
        for i in range(len(coq.tactic_history.getFullHistory()) -
                       (initial_history_len + common_prefix_len)):
            coq.cancel_last_noupdate()
        # Run the next nodes history from that point.
        for cmd in full_node_history[common_prefix_len:]:
            coq.run_stmt_noupdate(cmd)
        coq.update_state()
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

def combo_subgoal_search(lemma_name: str,
                          module_prefix: str,
                          relevant_lemmas: List[str],
                          coq: coq_serapy.SerapiInstance,
                          output_dir: Path,
                          args: argparse.Namespace,
                          bar_idx: int,
                          predictor_list: [TacticPredictor]) \
                          -> SearchResult:

    initial_history_len = len(coq.tactic_history.getFullHistory())
    full_context_before = FullContext(relevant_lemmas,
                                          coq.prev_tactics,
                                          unwrap(coq.proof_context))
    steps_taken = 0
    # Empty proof scripts set 
    proof_scripts = {tuple([])}
    steps_taken = 1 
    search_status = SearchStatus.INCOMPLETE
    max_steps = args.max_steps
    # context_history = []
    script_to_continue = tuple([])

    graph_file = f"{output_dir}/{module_prefix}{lemma_name}.svg"

    start_node = BFSNode(Prediction(lemma_name, 1.0, 1.0), 1.0, 0.0, [], full_context_before, None, subgoals_opened = 1)
    search_node = start_node
    nodes_to_continue = []
    # only to end fast
    #start_node.draw_graph(graph_file)
    #return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, 0)

    subgoals_seen = {}

    assert (args.max_steps is not None)

    total_subgoals_opened = 1
    times_subgoals_seen = 0

    while steps_taken < max_steps:

        shuffle(predictor_list)
        #round_robin = 0
        for curr_predictor in range(len(predictor_list)):
            if (args.search_width % len(predictor_list)) > curr_predictor:
                num_to_predict = int(args.search_width/len(predictor_list)) + (args.search_width % len(predictor_list))
            else:
                num_to_predict = int(args.search_width/len(predictor_list))
            if num_to_predict == 0:
                break
            num_predicted = 0
            apredictor = predictor_list[curr_predictor]
            #round_robin = round_robin + 1
            predictions = apredictor.predictKTactics(args,
                truncate_tactic_context(search_node.context_before.as_tcontext(), args.max_term_length),
                args.max_attempts,
                blacklist=args.blacklisted_tactics)
            #assert len(predictions) == args.max_attempts
            if coq.use_hammer:
                predictions = [Prediction(prediction.prediction[:-1] + "; try hammer.",
                                          prediction.certainty, prediction.no_softmax_certainty)
                               for prediction in predictions]
            for _prediction_idx, prediction in enumerate(predictions):
                if prediction.prediction in [childnode.prediction.prediction for childnode in search_node.children]:
                    continue
                try:
                    context_after, num_stmts, \
                        subgoals_closed, subgoals_opened, \
                        error, time_taken, unshelved = \
                            tryPrediction(args, coq, prediction.prediction, time.time())
                    if error:
                        continue
                    if contextIsBig(context_after) or \
                            contextInHistory(context_after, search_node):
                        eprint(f"Prediction in history or too big", guard=args.verbose >= 2)
                        for _ in range(num_stmts):
                            coq.cancel_last()
                        continue
                    if len(unwrap(coq.proof_context).all_goals) > args.max_subgoals:
                        for _ in range(num_stmts):
                            coq.cancel_last()
                        continue
                    # Check if the proof is done
                    postfix = []
                    if unshelved:
                        postfix.append("Unshelve.")
                    postfix += ["}"] * subgoals_closed
                    postfix += ["{"] * subgoals_opened
                    full_context_after = FullContext(relevant_lemmas, coq.prev_tactics,unwrap(coq.proof_context))
                    current_node = BFSNode(prediction, 0, time_taken, postfix, full_context_after, search_node, subgoals_opened = subgoals_opened)
                    subgoal_node = current_node
                    subgoal_solution = [subgoal_node.prediction.prediction]
                    if subgoals_opened > 0:
                        total_subgoals_opened += 1
                        current_goal = current_node.context_before.as_tcontext().goal
                        hashed_goal = hash(current_goal)
                        solved_already = subgoals_seen.get(hashed_goal)
                        if solved_already is not None:
                            times_subgoals_seen += 1
                            for step in solved_already:
                                try:
                                    context_after, temp_num_stmts, temp_subgoals_closed, temp_subgoals_opened, \
                                            error, time_taken, unshelved = tryPrediction(args, coq, step, time.time())
                                    if error:
                                        continue
                                    postfix = []
                                    if unshelved:
                                        postfix.append("Unshelve.")
                                    postfix += ["}"] * temp_subgoals_closed
                                    postfix += ["{"] * temp_subgoals_opened
                                    full_context_after = FullContext(relevant_lemmas, coq.prev_tactics,unwrap(coq.proof_context))
                                    current_node = BFSNode(Prediction(step,1,1), 0, time_taken, postfix, \
                                            full_context_after, current_node, subgoals_opened = temp_subgoals_opened)
                                    num_stmts = num_stmts + temp_num_stmts
                                    if completed_proof(coq):
                                        print("completed proof")
                                        current_node.mkQED()
                                        start_node.draw_graph(graph_file)
                                        return SearchResult(SearchStatus.SUCCESS, relevant_lemmas,current_node.interactions()[1:], 0) 
                                except coq_serapy.CoqAnomaly:
                                    if lemma_name == "":
                                        eprint("encountered unnamed goal!")
                                    else:
                                        eprint("coqanomaly without unnamed goal!")
                                    raise

                    if subgoals_closed > 0:
                        while subgoals_closed > 0: #TODO: add protections
                            closed_goal = subgoal_node.context_before.as_tcontext().goal
                            subgoal_node = subgoal_node.previous
                            while not subgoal_node.subgoals_opened > 0:
                                subgoal_solution.append(subgoal_node.prediction.prediction)
                                #for postfixbit in subgoal_node.postfix:
                                #    subgoal_solution.append(postfixbit)
                                subgoal_node = subgoal_node.previous
                            hashed_goal = hash(closed_goal)
                            subgoal_solution.reverse()
                            subgoals_seen[hashed_goal] = subgoal_solution
                            subgoals_closed = subgoals_closed - 1
                            subgoal_solution.reverse()

                    nodes_to_continue.append(current_node)
                    if completed_proof(coq):
                        eprint("completed proof")
                        current_node.mkQED()
                        start_node.draw_graph(graph_file)
                        return SearchResult(SearchStatus.SUCCESS, relevant_lemmas,
                                            current_node.interactions()[1:], 0)
                    else: 
                        for _ in range(num_stmts):
                            coq.cancel_last()
                            assert coq.proof_context
                        num_predicted += 1
                        if num_predicted >= num_to_predict:
                            break
                        else:
                            continue
                except coq_serapy.CoqAnomaly:
                    if lemma_name == "":
                        eprint("encountered unnamed goal!")
                    else:
                        eprint("coqanomaly without unnamed goal!")

                    raise
        steps_taken = steps_taken + 1

        with open("opened_subgoals.csv", "a") as afile:
            afile.write(lemma_name + "," + module_prefix + "," + str(total_subgoals_opened) + "," + str(times_subgoals_seen) + "," + "/n")
        
        if len(nodes_to_continue) == 0:
            steps_taken = max_steps
            next_search_node = start_node
            eprint("no nodes to conitnue!")
        else:
            myint = random.randint(0, len(nodes_to_continue) - 1)
            next_search_node = nodes_to_continue[myint]
            nodes_to_continue = [i for i in nodes_to_continue if not (i == myint)]
        
        next_search_node.traverse_to(coq, initial_history_len) 

        search_node = next_search_node
    # not sure what to do with the normally drawn graph
    #start_node.draw_graph(graph_file)
    #if hasUnexploredNode:
    start_node.draw_graph(graph_file)
    return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, 0)

def combo_b_search(lemma_name: str,
                          module_prefix: str,
                          relevant_lemmas: List[str],
                          coq: coq_serapy.SerapiInstance,
                          output_dir: Path,
                          args: argparse.Namespace,
                          bar_idx: int,
                          predictor_list: [TacticPredictor]) \
                          -> SearchResult:

    initial_history_len = len(coq.tactic_history.getFullHistory())
    full_context_before = FullContext(relevant_lemmas,
                                          coq.prev_tactics,
                                          unwrap(coq.proof_context))
    steps_taken = 0
    # Empty proof scripts set 
    proof_scripts = {tuple([])}
    steps_taken = 1 
    search_status = SearchStatus.INCOMPLETE
    max_steps = args.max_steps
    # context_history = []
    script_to_continue = tuple([])

    graph_file = f"{output_dir}/{module_prefix}{lemma_name}.svg"

    start_node = BFSNode(Prediction(lemma_name, 1.0, 1.0), 1.0, 0.0, [], full_context_before, None)
    search_node = start_node
    nodes_to_continue = []
    # only to end fast
    start_node.draw_graph(graph_file)
    #return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, 0)

    while steps_taken < max_steps:

        shuffle(predictor_list)
        #round_robin = 0
        for curr_predictor in range(len(predictor_list)):
            if (args.search_width % len(predictor_list)) > curr_predictor:
                num_to_predict = int(args.search_width/len(predictor_list)) + (args.search_width % len(predictor_list))
            else:
                num_to_predict = int(args.search_width/len(predictor_list))
            if num_to_predict == 0:
                break
            num_predicted = 0
            apredictor = predictor_list[curr_predictor]
            #round_robin = round_robin + 1
            predictions = apredictor.predictKTactics(args,
                truncate_tactic_context(search_node.context_before.as_tcontext(), args.max_term_length),
                args.max_attempts,
                blacklist=args.blacklisted_tactics)
            #assert len(predictions) == args.max_attempts
            if coq.use_hammer:
                predictions = [Prediction(prediction.prediction[:-1] + "; try hammer.",
                                          prediction.certainty, prediction.no_softmax_certainty)
                               for prediction in predictions]
            for _prediction_idx, prediction in enumerate(predictions):
                if prediction.prediction in [childnode.prediction.prediction for childnode in search_node.children]:
                    continue
                try:
                    context_after, num_stmts, \
                        subgoals_closed, subgoals_opened, \
                        error, time_taken, unshelved = \
                            tryPrediction(args, coq, prediction.prediction, time.time())
                    if error:
                        continue
                    if contextIsBig(context_after) or \
                            contextInHistory(context_after, search_node):
                        eprint(f"Prediction in history or too big", guard=args.verbose >= 2)
                        for _ in range(num_stmts):
                            coq.cancel_last()
                        continue
                    if len(unwrap(coq.proof_context).all_goals) > args.max_subgoals:
                        for _ in range(num_stmts):
                            coq.cancel_last()
                        continue
                    # Check if the proof is done
                    postfix = []
                    if unshelved:
                        postfix.append("Unshelve.")
                    postfix += ["}"] * subgoals_closed
                    postfix += ["{"] * subgoals_opened
                    full_context_after = FullContext(relevant_lemmas, coq.prev_tactics,unwrap(coq.proof_context))
                    current_node = BFSNode(prediction, 0, time_taken, postfix, full_context_after, search_node)
                    nodes_to_continue.append(current_node)
                    if completed_proof(coq):
                        eprint("completed proof")
                        current_node.mkQED()
                        start_node.draw_graph(graph_file)
                        return SearchResult(SearchStatus.SUCCESS, relevant_lemmas,
                                            search_node.interactions()[1:], 0)
                    else: 
                        for _ in range(num_stmts):
                            coq.cancel_last()
                            assert coq.proof_context
                        num_predicted += 1
                        if num_predicted >= num_to_predict:
                            break
                        else:
                            continue
                except coq_serapy.CoqAnomaly:
                    if lemma_name == "":
                        eprint("encountered unnamed goal!")
                    else:
                        eprint("coqanomaly without unnamed goal!")

                    raise
        steps_taken = steps_taken + 1
        
        if len(nodes_to_continue) == 0:
            steps_taken = max_steps
            next_search_node = start_node
            eprint("no nodes to conitnue!")
        else:
            myint = random.randint(0, len(nodes_to_continue) - 1)
            next_search_node = nodes_to_continue[myint]
            nodes_to_continue = [i for i in nodes_to_continue if not (i == myint)]
        
        next_search_node.traverse_to(coq, initial_history_len) 

        search_node = next_search_node
    # not sure what to do with the normally drawn graph
    #start_node.draw_graph(graph_file)
    #if hasUnexploredNode:
    start_node.draw_graph(graph_file)
    return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, 0)

def build_training_data(lemma_name: str,
                          module_prefix: str,
                          relevant_lemmas: List[str],
                          coq: coq_serapy.SerapiInstance,
                          output_dir: Path,
                          args: argparse.Namespace,
                          bar_idx: int,
                          predictor_list: [TacticPredictor]) \
                          -> SearchResult:

    initial_history_len = len(coq.tactic_history.getFullHistory())
    full_context_before = FullContext(relevant_lemmas,
                                          coq.prev_tactics,
                                          unwrap(coq.proof_context))
    steps_taken = 0
    # Empty proof scripts set 
    proof_scripts = {tuple([])}
    steps_taken = 1 
    search_status = SearchStatus.INCOMPLETE
    max_steps = args.max_steps
    # context_history = []
    script_to_continue = tuple([])

    graph_file = f"{output_dir}/{module_prefix}{lemma_name}.svg"

    start_node = BFSNode(Prediction(lemma_name, 1.0, 1.0), 1.0, 0.0, [], full_context_before, None)
    search_node = start_node
    nodes_to_continue = []
    # only to end fast
    start_node.draw_graph(graph_file)
    #return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, 0)

    while steps_taken < max_steps:
        
        shuffle(predictor_list)
        #round_robin = 0
        predictions_lists = []
        for curr_predictor in range(len(predictor_list)):
            apredictor = predictor_list[curr_predictor]
            predictions_temp = apredictor.predictKTactics(args,
                truncate_tactic_context(search_node.context_before.as_tcontext(), args.max_term_length),
                args.max_attempts,
                blacklist=args.blacklisted_tactics)
            #if coq.use_hammer:
            #    predictions_temp = [Prediction(prediction.prediction[:-1] + "; try hammer.",
            #                              prediction.certainty)
            #                   for prediction in predictions_temp]
            predictions_lists.append(predictions_temp)
        predictions = []
        only_tactics = tuple()
        full_tup = tuple()
        predictions_added = 0
        for tup in zip(*predictions_lists):
            full_tup = full_tup + tup
            only_tactics = only_tactics + tuple([full_pred[0] for full_pred in tup])
            mostcommon = Counter(only_tactics).most_common()[0]
            if mostcommon[1] > 1:
                best_tactic = mostcommon[0]
            else:
                highest_sureness = 0
                best_tactic = None
                for tactic_and_sureness in full_tup:
                    if tactic_and_sureness[2] > highest_sureness:
                        highest_sureness = tactic_and_sureness[2]
                        best_tactic = tactic_and_sureness[0]
            predictions.append(Prediction(best_tactic, 1.0, 1.0))
            only_tactics = tuple([x for x in only_tactics if not x==best_tactic])
            full_tup = tuple([x for x in full_tup if not x[0]==best_tactic])
        predictions_added = 0
        for _prediction_idx, prediction in enumerate(predictions):
            if predictions_added >=5:
                break
            if prediction.prediction in [childnode.prediction.prediction for childnode in search_node.children]:
                continue
            try:
                context_after, num_stmts, \
                    subgoals_closed, subgoals_opened, \
                    error, time_taken, unshelved = \
                        tryPrediction(args, coq, prediction.prediction, time.time())
                time_after_try = time.time()
                if error:
                    continue
                if contextIsBig(context_after) or \
                        contextInHistory(context_after, search_node):
                    eprint(f"Prediction in history or too big", guard=args.verbose >= 2)
                    for _ in range(num_stmts):
                        coq.cancel_last()
                    continue
                if len(unwrap(coq.proof_context).all_goals) > args.max_subgoals:
                    for _ in range(num_stmts):
                        coq.cancel_last()
                    continue
                postfix = []
                if unshelved:
                    postfix.append("Unshelve.")
                postfix += ["}"] * subgoals_closed
                postfix += ["{"] * subgoals_opened
                full_context_after = FullContext(relevant_lemmas, coq.prev_tactics,unwrap(coq.proof_context))
                current_node = BFSNode(prediction, 0, time_taken, postfix, full_context_after, search_node)
                nodes_to_continue.append(current_node)
                if completed_proof(coq):
                    eprint("completed proof")
                    current_node.mkQED()
                    start_node.draw_graph(graph_file)
                    return SearchResult(SearchStatus.SUCCESS, relevant_lemmas,
                                        search_node.interactions()[1:], 0)
                else: 
                    for _ in range(num_stmts):
                        coq.cancel_last()
                        assert coq.proof_context
                    predictions_added += 1
                    continue
                time_after_remove = time.time()
                eprint("time for append") 
                eprint(time_after_remove - time_after_check)
            except coq_serapy.CoqAnomaly:
                if lemma_name == "":
                    eprint("encountered unnamed goal!")
                else:
                    eprint("coqanomaly without unnamed goal!")

                raise
        steps_taken = steps_taken + 1
        
        if len(nodes_to_continue) == 0:
            steps_taken = max_steps
            next_search_node = start_node
            eprint("no nodes to conitnue!")
        else:
            myint = random.randint(0, len(nodes_to_continue) - 1)
            next_search_node = nodes_to_continue[myint]
            nodes_to_continue = [i for i in nodes_to_continue if not (i == myint)]
        
        next_search_node.traverse_to(coq, initial_history_len) 

        search_node = next_search_node

    start_node.draw_graph(graph_file)
    return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, 0)

def combo_b_two_search(lemma_name: str,
                          module_prefix: str,
                          relevant_lemmas: List[str],
                          coq: coq_serapy.SerapiInstance,
                          output_dir: Path,
                          args: argparse.Namespace,
                          bar_idx: int,
                          predictor_list: [TacticPredictor]) \
                          -> SearchResult:

    initial_history_len = len(coq.tactic_history.getFullHistory())
    full_context_before = FullContext(relevant_lemmas,
                                          coq.prev_tactics,
                                          unwrap(coq.proof_context))
    steps_taken = 0
    # Empty proof scripts set 
    proof_scripts = {tuple([])}
    steps_taken = 1 
    search_status = SearchStatus.INCOMPLETE
    max_steps =150
    # context_history = []
    script_to_continue = tuple([])

    graph_file = f"{output_dir}/{module_prefix}{lemma_name}.svg"

    start_node = BFSNode(Prediction(lemma_name, 1.0, 1.0), 1.0, 0.0, [], full_context_before, None)
    search_node = start_node
    nodes_to_continue = []
    # only to end fast
    start_node.draw_graph(graph_file)
    #return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, 0)

    while steps_taken < max_steps:
        

        shuffle(predictor_list)
        #round_robin = 0
        predictions_lists = []
        for curr_predictor in range(len(predictor_list)):
            apredictor = predictor_list[curr_predictor]
            predictions_temp = apredictor.predictKTactics(args,
                truncate_tactic_context(search_node.context_before.as_tcontext(), args.max_term_length),
                args.max_attempts,
                blacklist=args.blacklisted_tactics)
            if coq.use_hammer:
                predictions_temp = [Prediction(prediction.prediction[:-1] + "; try hammer.",
                                          prediction.certainty, prediction.no_softmax_certainty)
                               for prediction in predictions_temp]
            predictions_lists.append(predictions_temp)
        predictions = []
        for tup in zip(*predictions_lists):
            predictions.append(Counter(tup).most_common(1)[0][0])
        predictions_added = 0
        for _prediction_idx, prediction in enumerate(predictions):
            if predictions_added >=5:
                break
            if prediction.prediction in [childnode.prediction.prediction for childnode in search_node.children]:
                continue
            try:
                context_after, num_stmts, \
                    subgoals_closed, subgoals_opened, \
                    error, time_taken, unshelved = \
                        tryPrediction(args, coq, prediction.prediction, time.time())
                if error:
                    continue
                if contextIsBig(context_after) or \
                        contextInHistory(context_after, search_node):
                    eprint(f"Prediction in history or too big", guard=args.verbose >= 2)
                    for _ in range(num_stmts):
                        coq.cancel_last()
                    continue
                if len(unwrap(coq.proof_context).all_goals) > args.max_subgoals:
                    for _ in range(num_stmts):
                        coq.cancel_last()
                    continue
                postfix = []
                if unshelved:
                    postfix.append("Unshelve.")
                postfix += ["}"] * subgoals_closed
                postfix += ["{"] * subgoals_opened
                full_context_after = FullContext(relevant_lemmas, coq.prev_tactics,unwrap(coq.proof_context))
                current_node = BFSNode(prediction, 0, time_taken, postfix, full_context_after, search_node)
                nodes_to_continue.append(current_node)
                if completed_proof(coq):
                    eprint("completed proof")
                    current_node.mkQED()
                    start_node.draw_graph(graph_file)
                    return SearchResult(SearchStatus.SUCCESS, relevant_lemmas,
                                        search_node.interactions()[1:], 0)
                else: 
                    for _ in range(num_stmts):
                        coq.cancel_last()
                        assert coq.proof_context
                    predictions_added += 1
                    continue
            except coq_serapy.CoqAnomaly:
                if lemma_name == "":
                    eprint("encountered unnamed goal!")
                else:
                    eprint("coqanomaly without unnamed goal!")

                raise
        steps_taken = steps_taken + 1
        
        if len(nodes_to_continue) == 0:
            steps_taken = max_steps
            next_search_node = start_node
            eprint("no nodes to conitnue!")
        else:
            myint = random.randint(0, len(nodes_to_continue) - 1)
            next_search_node = nodes_to_continue[myint]
            nodes_to_continue = [i for i in nodes_to_continue if not (i == myint)]
        
        next_search_node.traverse_to(coq, initial_history_len) 

        search_node = next_search_node

    start_node.draw_graph(graph_file)
    return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, 0)

def combo_b_vote_search(lemma_name: str,
                          module_prefix: str,
                          relevant_lemmas: List[str],
                          coq: coq_serapy.SerapiInstance,
                          output_dir: Path,
                          args: argparse.Namespace,
                          bar_idx: int,
                          predictor_list: [TacticPredictor]) \
                          -> SearchResult:

    initial_history_len = len(coq.tactic_history.getFullHistory())
    full_context_before = FullContext(relevant_lemmas,
                                          coq.prev_tactics,
                                          unwrap(coq.proof_context))
    steps_taken = 0
    # Empty proof scripts set 
    proof_scripts = {tuple([])}
    steps_taken = 1 
    search_status = SearchStatus.INCOMPLETE
    max_steps = args.max_steps
    # context_history = []
    script_to_continue = tuple([])

    graph_file = f"{output_dir}/{module_prefix}{lemma_name}.svg"

    start_node = BFSNode(Prediction(lemma_name, 1.0, 1.0), 1.0, 0.0, [], full_context_before, None)
    search_node = start_node
    nodes_to_continue = []
    # only to end fast
    start_node.draw_graph(graph_file)
    #return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, 0)

    while steps_taken < max_steps:
        
        shuffle(predictor_list)
        #round_robin = 0
        predictions_lists = []
        for curr_predictor in range(len(predictor_list)):
            apredictor = predictor_list[curr_predictor]
            predictions_temp = apredictor.predictKTactics(args,
                truncate_tactic_context(search_node.context_before.as_tcontext(), args.max_term_length),
                args.max_attempts,
                blacklist=args.blacklisted_tactics)
            #if coq.use_hammer:
            #    predictions_temp = [Prediction(prediction.prediction[:-1] + "; try hammer.",
            #                              prediction.certainty)
            #                   for prediction in predictions_temp]
            predictions_lists.append(predictions_temp)
        predictions = []
        only_tactics = tuple()
        full_tup = tuple()
        predictions_added = 0
        for tup in zip(*predictions_lists):
            full_tup = full_tup + tup
            only_tactics = only_tactics + tuple([full_pred[0] for full_pred in tup])
            mostcommon = Counter(only_tactics).most_common()[0]
            if mostcommon[1] > 1:
                best_tactic = mostcommon[0]
            else:
                highest_sureness = 0
                best_tactic = None
                for tactic_and_sureness in full_tup:
                    if tactic_and_sureness[2] > highest_sureness:
                        highest_sureness = tactic_and_sureness[2]
                        best_tactic = tactic_and_sureness[0]
            predictions.append(Prediction(best_tactic, 1.0, 1.0))
            only_tactics = tuple([x for x in only_tactics if not x==best_tactic])
            full_tup = tuple([x for x in full_tup if not x[0]==best_tactic])
        predictions_added = 0
        for _prediction_idx, prediction in enumerate(predictions):
            if predictions_added >=5:
                break
            if prediction.prediction in [childnode.prediction.prediction for childnode in search_node.children]:
                continue
            try:
                context_after, num_stmts, \
                    subgoals_closed, subgoals_opened, \
                    error, time_taken, unshelved = \
                        tryPrediction(args, coq, prediction.prediction, time.time())
                time_after_try = time.time()
                if error:
                    continue
                if contextIsBig(context_after) or \
                        contextInHistory(context_after, search_node):
                    eprint(f"Prediction in history or too big", guard=args.verbose >= 2)
                    for _ in range(num_stmts):
                        coq.cancel_last()
                    continue
                if len(unwrap(coq.proof_context).all_goals) > args.max_subgoals:
                    for _ in range(num_stmts):
                        coq.cancel_last()
                    continue
                postfix = []
                if unshelved:
                    postfix.append("Unshelve.")
                postfix += ["}"] * subgoals_closed
                postfix += ["{"] * subgoals_opened
                full_context_after = FullContext(relevant_lemmas, coq.prev_tactics,unwrap(coq.proof_context))
                current_node = BFSNode(prediction, 0, time_taken, postfix, full_context_after, search_node)
                nodes_to_continue.append(current_node)
                if completed_proof(coq):
                    eprint("completed proof")
                    current_node.mkQED()
                    start_node.draw_graph(graph_file)
                    return SearchResult(SearchStatus.SUCCESS, relevant_lemmas,
                                        search_node.interactions()[1:], 0)
                else: 
                    for _ in range(num_stmts):
                        coq.cancel_last()
                        assert coq.proof_context
                    predictions_added += 1
                    continue
                time_after_remove = time.time()
                eprint("time for append") 
                eprint(time_after_remove - time_after_check)
            except coq_serapy.CoqAnomaly:
                if lemma_name == "":
                    eprint("encountered unnamed goal!")
                else:
                    eprint("coqanomaly without unnamed goal!")

                raise
        steps_taken = steps_taken + 1
        
        if len(nodes_to_continue) == 0:
            steps_taken = max_steps
            next_search_node = start_node
            eprint("no nodes to conitnue!")
        else:
            myint = random.randint(0, len(nodes_to_continue) - 1)
            next_search_node = nodes_to_continue[myint]
            nodes_to_continue = [i for i in nodes_to_continue if not (i == myint)]
        
        next_search_node.traverse_to(coq, initial_history_len) 

        search_node = next_search_node

    start_node.draw_graph(graph_file)
    return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, 0)

def bfs_beam_proof_search(lemma_name: str,
                          module_prefix: str,
                          relevant_lemmas: List[str],
                          coq: coq_serapy.SerapiInstance,
                          output_dir: Path,
                          args: argparse.Namespace,
                          bar_idx: int,
                          predictor: TacticPredictor) \
                          -> SearchResult:
    hasUnexploredNode = False
    graph_file = f"{output_dir}/{module_prefix}{lemma_name}.svg"

    features_extractor = FeaturesExtractor(str(args.tactics_file), str(args.tokens_file))
    if args.scoring_function == "lstd":
        state_estimator = Estimator(args.beta_file)
    elif args.scoring_function == "pickled":
        with args.pickled_estimator.open('rb') as f:
            john_model = pickle.load(f)

    initial_history_len = len(coq.tactic_history.getFullHistory())
    start_node = BFSNode(Prediction(lemma_name, 1.0, 1.0), 1.0, 0.0, [],
                         FullContext([], [],
                                     ProofContext([], [], [], [])), None)
    search_start_node = start_node
    if args.search_prefix:
        for command in coq_serapy.read_commands(args.search_prefix):
            full_context_before = FullContext(relevant_lemmas,
                                              coq.prev_tactics,
                                              unwrap(coq.proof_context))
            search_start_node = BFSNode(Prediction(command, 1.0, 1.0), 1.0, 0.0, [],
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
                            args.max_attempts,
                    blacklist=args.blacklisted_tactics)
                prednum = 0
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
                    if len(unwrap(coq.proof_context).all_goals) > args.max_subgoals:
                        if args.count_softfail_predictions:
                            num_successful_predictions += 1
                        prediction_node.setNodeColor("orange")
                        for _ in range(num_stmts):
                            coq.cancel_last()
                        continue
                    if completed_proof(coq):
                        prediction_node.mkQED()
                        start_node.draw_graph(graph_file)
                        return SearchResult(SearchStatus.SUCCESS, relevant_lemmas,
                                            prediction_node.interactions()[1:], 0)

                    if args.scoring_function == "certainty":
                        prediction_node.score = next_node.score * prediction.certainty
                    elif args.scoring_function == "pickled":
                        assert sys.version_info >= (3, 10), "Pickled estimators only supported in python 3.10 or newer"
                        score = 0.
                        for obl in coq.proof_context.fg_goals + coq.proof_context.bg_goals:
                            try:
                                score += -float(john_model.predict_obl(obl))
                            except UnhandledExpr:
                                print(f"Couldn't handle goal {unwrap(coq.proof_context).all_goals[idx]}")
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
        return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, 0)
    else:
        return SearchResult(SearchStatus.FAILURE, relevant_lemmas, None, 0)

@dataclass(order=True)
class AStarTask:
    f_score: float
    node: BFSNode=field(compare=False)


def best_first_proof_search(lemma_name: str,
                       module_prefix: Optional[str],
                       relevant_lemmas: List[str],
                       coq: coq_serapy.SerapiInstance,
                       output_dir: Path,
                       args: argparse.Namespace,
                       bar_idx: int,
                       predictor: TacticPredictor) \
                       -> SearchResult:
    assert args.scoring_function in ["pickled", "const", "pickled-normcert"] or args.search_type != "astar", "only pickled and const scorers are currently compatible with A* search"
    if args.scoring_function in ["pickled", "pickled-normcert"]:
        with args.pickled_estimator.open('rb') as f:
            john_model = pickle.load(f)
    graph_file = f"{output_dir}/{module_prefix}{lemma_name}.svg"
    initial_history_len = len(coq.tactic_history.getFullHistory())
    start_node = BFSNode(Prediction(lemma_name, 1.0, 1.0), 1.0, 0.0, [],
                         FullContext([], [],
                                     ProofContext([], [], [], [])), None)
    search_start_node = start_node
    if args.search_prefix:
        for command in coq_serapy.read_commands(args.search_prefix):
            full_context_before = FullContext(relevant_lemmas,
                                              coq.prev_tactics,
                                              unwrap(coq.proof_context))
            search_start_node = BFSNode(Prediction(command, 1.0, 1.0), 1.0, 0.0, [],
                                        full_context_before, search_start_node)
    nodes_todo: List[AStarTask] = [AStarTask(1.0, search_start_node)]

    desc_name = lemma_name
    if len(desc_name) > 25:
        desc_name = desc_name[:22] + "..."
    assert args.max_steps != None, "When using astar search, you need a step limit. Please specify one with --max-steps"
    for step in trange(args.max_steps, unit="pred", file=sys.stdout,
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
        predictions = predictor.predictKTactics(args,
            truncate_tactic_context(full_context_before.as_tcontext(),
                                    args.max_term_length),
            args.max_attempts,
            blacklist=args.blacklisted_tactics)

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
            # Check if we've gone in circles
            if contextInHistory(context_after, prediction_node):
                if args.count_softfail_predictions:
                    num_successful_predictions += 1
                eprint(f"Prediction in history", guard=args.verbose >= 2)
                prediction_node.setNodeColor("orange")
                for _ in range(num_stmts):
                    coq.cancel_last()
                continue
            num_successful_predictions += 1
            # Check if the resulting context is too big
            if len(unwrap(coq.proof_context).all_goals) > args.max_subgoals or \
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
                return SearchResult(SearchStatus.SUCCESS, relevant_lemmas,
                                    prediction_node.interactions()[1:], step+1)
            if args.scoring_function == "const":
                h_score = 1.
            elif args.scoring_function == "certainty":
                h_score = -abs(next_node.f_score * prediction.certainty)
            elif args.scoring_function == "norm-certainty":
                h_score = -math.sqrt(abs(next_node.f_score * prediction.certainty))
            else:
                assert args.scoring_function in ["pickled", "pickled-normcert"]
                assert sys.version_info >= (3, 10), "Pickled estimators only supported in python 3.10 or newer"
                h_score = 0.
                for obl in coq.proof_context.fg_goals + coq.proof_context.bg_goals:
                    try:
                        h_score += float(john_model.predict_obl(obl))
                    except UnhandledExpr:
                        print(f"Couldn't handle goal {unwrap(coq.proof_context).all_goals[idx]}")
                        raise
                if args.scoring_function == "pickled-normcert":
                    normcert_score = prediction.certainty
                    path_length = 1
                    for node in next_node.node.path():
                        normcert_score *= node.prediction.certainty
                        path_length += 1
                    normcert_score = max(normcert_score ** (1 / path_length), 0.001)
                    assert normcert_score <= 1 and normcert_score > 0, normcert_score
                    h_score /= normcert_score
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
        return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, step)
    return SearchResult(SearchStatus.FAILURE, relevant_lemmas, None, step)

def rnn_dfs_proof_search(lemma_name: str,
                         module_prefix: str,
                         relevant_lemmas: List[str],
                         coq: coq_serapy.SerapiInstance,
                         output_dir: Path,
                         args: argparse.Namespace,
                         bar_idx: int,
                         predictor_list: [TacticPredictor],
                         model_list: [zhannRNN],
                         vectorizer: coq2vec.CoqTermRNNVectorizer) \
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
               extra_depth: int, steps_explored: int) -> SubSearchResult:
        nonlocal hasUnexploredNode
        nonlocal relevant_lemmas
        global unnamed_goal_number
        full_context_before = FullContext(relevant_lemmas,
                                          coq.prev_tactics,
                                          unwrap(coq.proof_context))

        rnn_current_goal = unwrap(coq.proof_context).fg_goals[0].goal
        encoded_tensor = vectorizer.term_to_vector(' '.join(rnn_current_goal.split()))
        #certainties = []
        #bestmod = 0
        #bestnum = 0
        #t = 0
        #for rnnmod in model_list:
        #    the_cert = rnnmod(encoded_tensor).item()
        #    certainties.append(the_cert)
        #    if the_cert > bestnum:
        #        bestnum = the_cert
        #        bestmod = t
        #    t += 1
        rnnmod = model_list[0]
        modout = rnnmod(encoded_tensor)
        listmod = modout.tolist()
        bestmod = listmod.index(max(listmod))

        predictor = predictor_list[bestmod]

        # PUT HERE
        predictions = predictor.predictKTactics(args,
                                                truncate_tactic_context(full_context_before.as_tcontext(),
                                                args.max_term_length),
                                                args.max_attempts,
                                                blacklist=args.blacklisted_tactics)
        assert len(predictions) == args.max_attempts

        #assert len(predictions) == args.max_attempts
        if coq.use_hammer:
            predictions = [Prediction(prediction.prediction[:-1] + "; try hammer.",
                                      prediction.certainty, prediction.no_softmax_certainty)
                           for prediction in predictions]
        num_successful_predictions = 0
        substeps_explored = 1
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
                                                  subgoals_opened,
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
                                          subgoals_opened,
                                          current_path[-1])
                predictionNode.time_taken = time_taken
                if unshelved:
                    predictionNode = g.mkNode(Prediction("Unshelve.", 1.0, 1.0),
                                              full_context_before,
                                              subgoals_opened,
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
                    return SubSearchResult(solution, subgoals_closed, steps_explored + substeps_explored)
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
                        and len(current_path) < args.hard_depth_limit \
                        and (args.max_steps is None or
                             substeps_explored < args.max_steps):
                    if subgoals_closed > 0:
                        g.setNodeColor(predictionNode, "blue")
                    sub_search_result = search(pbar,
                                               current_path + [predictionNode],
                                               new_distance_stack,
                                               new_extra_depth, steps_explored + substeps_explored)
                    substeps_explored += sub_search_result.steps_explored
                    cleanupSearch(num_stmts, "we finished subsearch")
                    if sub_search_result.solution or \
                       sub_search_result.solved_subgoals > subgoals_opened:
                        new_subgoals_closed = \
                            subgoals_closed + \
                            sub_search_result.solved_subgoals - \
                            subgoals_opened
                        return SubSearchResult(sub_search_result.solution,
                                               new_subgoals_closed, substeps_explored)
                    if subgoals_closed > 0:
                        return SubSearchResult(None, subgoals_closed, substeps_explored)
                else:
                    hasUnexploredNode = True
                    cleanupSearch(num_stmts, "we hit the depth limit")
                    if subgoals_closed > 0:
                        # depth = (args.search_depth + new_extra_depth + 1) \
                        #     - len(current_path)
                        return SubSearchResult(None, subgoals_closed, substeps_explored)
            except coq_serapy.CoqAnomaly:
                predictionNode = g.mkNode(prediction,
                                          full_context_before,
                                          subgoals_opened,
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
        return SubSearchResult(None, 0, substeps_explored)
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
        next_node = g.start_node
        if args.search_prefix is not None:
            for command in coq_serapy.read_commands(args.search_prefix):
                full_context_before = FullContext(relevant_lemmas,
                                                  coq.prev_tactics,
                                                  unwrap(coq.proof_context))
                next_node = g.mkNode(Prediction(command, 1.0, 1.0),
                                     full_context_before,
                                     0,
                                     next_node)
                next_node.time_taken = 0.0
                coq.run_stmt(command)
        command_list, _, total_steps = search(pbar, [next_node], subgoals_stack_start, 0, 0)
        pbar.clear()
    g.draw(f"{output_dir}/{module_prefix}{lemma_name}.svg")
    if args.features_json:
        g.write_feat_json(f"{output_dir}/{module_prefix}"
                          f"{lemma_name}.json")
    if command_list:
        return SearchResult(SearchStatus.SUCCESS, relevant_lemmas, command_list, total_steps)
    if hasUnexploredNode:
        return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, total_steps)
    return SearchResult(SearchStatus.FAILURE, relevant_lemmas, None, total_steps)


def dfs_proof_search_with_vote(lemma_name: str,
                                module_prefix: str,
                                relevant_lemmas: List[str],
                                coq: coq_serapy.SerapiInstance,
                                output_dir: Path,
                                args: argparse.Namespace,
                                bar_idx: int,
                                predictor_list: [TacticPredictor]) \
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
               extra_depth: int, steps_explored: int) -> SubSearchResult:
        nonlocal hasUnexploredNode
        nonlocal relevant_lemmas
        global unnamed_goal_number
        full_context_before = FullContext(relevant_lemmas,
                                          coq.prev_tactics,
                                          unwrap(coq.proof_context))

        shuffle(predictor_list)
        predictions_lists = []
        for curr_predictor in range(len(predictor_list)):
            apredictor = predictor_list[curr_predictor]
            predictions_temp = apredictor.predictKTactics(args,
                truncate_tactic_context(full_context_before.as_tcontext(), args.max_term_length),
                args.max_attempts,
                blacklist=args.blacklisted_tactics)
            predictions_lists.append(predictions_temp)
        predictions = []
        only_tactics = tuple()
        full_tup = tuple()
        for tup in zip(*predictions_lists):
            full_tup = full_tup + tup
            only_tactics = only_tactics + tuple([full_pred[0] for full_pred in tup])
            mostcommon = Counter(only_tactics).most_common()[0]
            if mostcommon[1] > 1:
                best_tactic = mostcommon[0]
            else:
                highest_sureness = 0
                best_tactic = None
                for tactic_and_sureness in full_tup:
                    if tactic_and_sureness[2] > highest_sureness:
                        highest_sureness = tactic_and_sureness[2]
                        best_tactic = tactic_and_sureness[0]
            predictions.append(Prediction(best_tactic, 1.0, 1.0))
            only_tactics = tuple([x for x in only_tactics if not x==best_tactic])
            full_tup = tuple([x for x in full_tup if not x[0]==best_tactic])


        #assert len(predictions) == args.max_attempts
        if coq.use_hammer:
            predictions = [Prediction(prediction.prediction[:-1] + "; try hammer.",
                                      prediction.certainty, prediction.no_softmax_certainty)
                           for prediction in predictions]
        num_successful_predictions = 0
        substeps_explored = 1
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
                                                  subgoals_opened,
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
                                          subgoals_opened,
                                          current_path[-1])
                predictionNode.time_taken = time_taken
                if unshelved:
                    predictionNode = g.mkNode(Prediction("Unshelve.", 1.0, 1.0),
                                              full_context_before,
                                              subgoals_opened,
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
                    return SubSearchResult(solution, subgoals_closed, steps_explored + substeps_explored)
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
                        and len(current_path) < args.hard_depth_limit \
                        and (args.max_steps is None or
                             substeps_explored < args.max_steps):
                    if subgoals_closed > 0:
                        g.setNodeColor(predictionNode, "blue")
                    sub_search_result = search(pbar,
                                               current_path + [predictionNode],
                                               new_distance_stack,
                                               new_extra_depth, steps_explored + substeps_explored)
                    substeps_explored += sub_search_result.steps_explored
                    cleanupSearch(num_stmts, "we finished subsearch")
                    if sub_search_result.solution or \
                       sub_search_result.solved_subgoals > subgoals_opened:
                        new_subgoals_closed = \
                            subgoals_closed + \
                            sub_search_result.solved_subgoals - \
                            subgoals_opened
                        return SubSearchResult(sub_search_result.solution,
                                               new_subgoals_closed, substeps_explored)
                    if subgoals_closed > 0:
                        return SubSearchResult(None, subgoals_closed, substeps_explored)
                else:
                    hasUnexploredNode = True
                    cleanupSearch(num_stmts, "we hit the depth limit")
                    if subgoals_closed > 0:
                        # depth = (args.search_depth + new_extra_depth + 1) \
                        #     - len(current_path)
                        return SubSearchResult(None, subgoals_closed, substeps_explored)
            except coq_serapy.CoqAnomaly:
                predictionNode = g.mkNode(prediction,
                                          full_context_before,
                                          subgoals_opened,
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
        return SubSearchResult(None, 0, substeps_explored)
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
        next_node = g.start_node
        if args.search_prefix is not None:
            for command in coq_serapy.read_commands(args.search_prefix):
                full_context_before = FullContext(relevant_lemmas,
                                                  coq.prev_tactics,
                                                  unwrap(coq.proof_context))
                next_node = g.mkNode(Prediction(command, 1.0, 1.0),
                                     full_context_before,
                                     0,
                                     next_node)
                next_node.time_taken = 0.0
                coq.run_stmt(command)
        command_list, _, total_steps = search(pbar, [next_node], subgoals_stack_start, 0, 0)
        pbar.clear()
    g.draw(f"{output_dir}/{module_prefix}{lemma_name}.svg")
    if args.features_json:
        g.write_feat_json(f"{output_dir}/{module_prefix}"
                          f"{lemma_name}.json")
    if command_list:
        return SearchResult(SearchStatus.SUCCESS, relevant_lemmas, command_list, total_steps)
    if hasUnexploredNode:
        return SearchResult(SearchStatus.INCOMPLETE, relevant_lemmas, None, total_steps)
    return SearchResult(SearchStatus.FAILURE, relevant_lemmas, None, total_steps)

def dfs_estimated(lemma_name: str,
                  module_prefix: str,
                  relevant_lemmas: List[str],
                  coq: coq_serapy.SerapiInstance,
                  output_dir: Path,
                  args: argparse.Namespace,
                  bar_idx: int,
                  predictor: TacticPredictor) \
                  -> SearchResult:
    with args.pickled_estimator.open('rb') as f:
        with nostderr():
            john_model = pickle.load(f)

    est_sol_length = 0.
    for obl in coq.proof_context.fg_goals:
        est_sol_length += max(1, john_model.predict_obl(obl))
    temp_args = copyArgs(args)
    caution_factor = 1.4
    if est_sol_length * caution_factor < args.search_depth:
        eprint(f"Estimated solution length is {est_sol_length}, "
               f"giving proof only {int(est_sol_length * caution_factor)} depth budget")
        temp_args.search_depth = math.ceil(est_sol_length * caution_factor)
    return dfs_proof_search_with_graph(
        lemma_name, module_prefix,
        relevant_lemmas, coq, output_dir,
        temp_args, bar_idx, predictor)
