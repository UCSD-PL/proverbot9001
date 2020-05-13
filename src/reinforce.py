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

import argparse
import random
import torch

import serapi_instance
import dataloader
import tokenizer
from models import tactic_predictor
from models.q_estimator import QEstimator
from models.features_q_estimator import FeaturesQEstimator
import predict_tactic
from util import eprint, print_time, nostderr

from dataclasses import dataclass
from typing import List, Tuple, Iterator, Optional
from format import TacticContext
from pathlib_revised import Path2

import pygraphviz as pgv
from tqdm import trange


def main() -> None:
    parser = \
        argparse.ArgumentParser(
            description="A module for exploring deep Q learning "
            "with proverbot9001")

    parser.add_argument("scrape_file")

    parser.add_argument("environment_file", type=Path2)
    parser.add_argument("out_weights", type=Path2)
    parser.add_argument("--proof", default=None)

    parser.add_argument("--prelude", default=".", type=Path2)

    parser.add_argument("--predictor-weights",
                        default=Path2("data/polyarg-weights.dat"),
                        type=Path2)
    parser.add_argument("--start-from", default=None, type=Path2)
    parser.add_argument("--num-predictions", default=16, type=int)

    parser.add_argument("--buffer-size", default=256, type=int)
    parser.add_argument("--batch-size", default=32, type=int)

    parser.add_argument("--num-episodes", default=256, type=int)
    parser.add_argument("--episode-length", default=16, type=int)

    parser.add_argument("--learning-rate", default=0.0001, type=float)

    parser.add_argument("--progress", "-P", action='store_true')
    parser.add_argument("--verbose", "-v", action='count', default=0)

    parser.add_argument("--ghosts", action='store_true')

    args = parser.parse_args()

    reinforce(args)


@dataclass(init=True)
class LabeledNode:
    action: str
    reward: float
    node_id: int
    parent: Optional["LabeledNode"]
    children: List["LabeledNode"]


class ReinforceGraph:
    __graph: pgv.AGraph
    __next_node_id: int
    start_node: LabeledNode

    def __init__(self, lemma_name: str) -> None:
        self.__graph = pgv.AGraph(directed=True)
        self.__next_node_id = 0
        self.start_node = self.mkNode(lemma_name, 0, None)
        pass

    def addTransition(self, src: LabeledNode, action: str, reward: float,
                      **kwargs) -> LabeledNode:
        for child in src.children:
            if child.action == action:
                assert child.reward == reward
                return child
        return self.mkNode(action, reward, src, **kwargs)

    def addGhostTransition(self, src: LabeledNode, action: str,
                           **kwargs) -> LabeledNode:
        for child in src.children:
            if child.action == action:
                return child
        return self.mkNode(action, 0, src, fillcolor="grey", **kwargs)

    def mkNode(self, action: str, reward: float,
               previous_node: Optional[LabeledNode],
               **kwargs) -> LabeledNode:
        if 'fillcolor' not in kwargs:
            if reward > 0:
                color = "palegreen"
            elif reward < 0:
                color = "indianred1"
            else:
                color = "white"
            self.__graph.add_node(self.__next_node_id, label=action,
                                  fillcolor=color, style="filled",
                                  **kwargs)
        else:
            self.__graph.add_node(self.__next_node_id, label=action)
        self.__next_node_id += 1
        newNode = LabeledNode(action, reward, self.__next_node_id-1,
                              previous_node, [])
        if previous_node:
            self.__graph.add_edge(previous_node.node_id, newNode.node_id,
                                  label=str(reward), **kwargs)
            previous_node.children.append(newNode)
        return newNode

    def mkQED(self, src: LabeledNode):
        for existing_node in src.children:
            if existing_node.action == "QED":
                return
        self.mkNode("QED", 0, src, fillcolor="green", style="filled")
        cur_node = src
        while cur_node != self.start_node:
            self.setNodeOutlineColor(cur_node, "palegreen1")
            assert cur_node.parent
            cur_node = cur_node.parent
        pass

    def setNodeColor(self, node: LabeledNode, color: str) -> None:
        node_handle = self.__graph.get_node(node.node_id)
        node_handle.attr["fillcolor"] = color
        node_handle.attr["style"] = "filled"

    def setNodeOutlineColor(self, node: LabeledNode, color: str) -> None:
        node_handle = self.__graph.get_node(node.node_id)
        node_handle.attr["color"] = color

    def setNodeApproxQScore(self, node: LabeledNode, score: float) -> None:
        node_handle = self.__graph.get_node(node.node_id)
        node_handle.attr["label"] = f"{node.action} (~{score:.2f})"

    def draw(self, filename: str) -> None:
        with nostderr():
            self.__graph.draw(filename, prog="dot")


def reinforce(args: argparse.Namespace) -> None:

    # Load the scraped (demonstrated) samples, the proof environment
    # commands, and the predictor
    replay_memory = assign_rewards(
        dataloader.tactic_transitions_from_file(args.scrape_file,
                                                args.buffer_size))
    env_commands = serapi_instance.load_commands_preserve(
        args, 0, args.prelude / args.environment_file)
    predictor = predict_tactic.loadPredictorByFile(args.predictor_weights)

    q_estimator = FeaturesQEstimator(args.learning_rate)
    if args.start_from:
        q_estimator_name, prev_args, unparsed_prev_args, state_dict = \
            torch.load(args.start_from)
        q_estimator.model.load_state_dict(state_dict)

    epsilon = 0.3
    gamma = 0.9

    with serapi_instance.SerapiContext(
            ["sertop", "--implicit"],
            serapi_instance.get_module_from_filename(args.environment_file),
            str(args.prelude)) as coq:
        coq.quiet = True
        coq.verbose = args.verbose
        # Get us to the correct proof context
        rest_commands, run_commands = coq.run_into_next_proof(env_commands)
        lemma_statement = run_commands[-1]
        if args.proof is not None:
            while coq.cur_lemma_name != args.proof:
                if not rest_commands:
                    eprint("Couldn't find lemma {args.proof}! Exiting...")
                    return
                rest_commands, _ = coq.finish_proof(rest_commands)
                rest_commands, run_commands = coq.run_into_next_proof(
                    rest_commands)
                lemma_statement = run_commands[-1]
        else:
            # Don't use lemmas without names (e.g. "Obligation")
            while coq.cur_lemma_name == "":
                if not rest_commands:
                    eprint("Couldn't find usable lemma! Exiting...")
                    return
                rest_commands, _ = coq.finish_proof(rest_commands)
                rest_commands, run_commands = coq.run_into_next_proof(
                    rest_commands)
                lemma_statement = run_commands[-1]

        lemma_name = coq.cur_lemma_name

        graph = ReinforceGraph(lemma_name)

        for episode in trange(args.num_episodes, disable=(not args.progress)):
            cur_node = graph.start_node
            proof_contexts_seen = [coq.proof_context]
            for t in trange(args.episode_length, disable=(not args.progress),
                            leave=False):
                with print_time("Getting predictions", guard=args.verbose):
                    context_before = coq.tactic_context(coq.local_lemmas[:-1])
                    predictions = predictor.predictKTactics(
                        context_before, args.num_predictions)
                if random.random() < epsilon:
                    ordered_actions = [p.prediction for p in
                                       random.sample(predictions,
                                                     len(predictions))]
                else:
                    with print_time("Picking actions using q_estimator",
                                    guard=args.verbose):
                        q_choices = zip(q_estimator(
                            [(context_before, prediction.prediction)
                             for prediction in predictions]),
                                        [p.prediction for p in predictions])
                        ordered_actions = [p[1] for p in
                                           sorted(q_choices,
                                                  key=lambda q: q[0],
                                                  reverse=True)]

                with print_time("Running actions", guard=args.verbose):
                    action = None
                    for try_action in ordered_actions:
                        try:
                            coq.run_stmt(try_action)
                            proof_context_after = coq.proof_context
                            if any([serapi_instance.contextSurjective(
                                    proof_context_after, path_context)
                                    for path_context in proof_contexts_seen]):
                                coq.cancel_last()
                                if args.ghosts:
                                    graph.addGhostTransition(cur_node,
                                                             try_action)
                                continue
                            action = try_action
                            break
                        except (serapi_instance.ParseError,
                                serapi_instance.CoqExn):
                            if args.ghosts:
                                graph.addGhostTransition(cur_node, try_action)
                            pass
                    if action is None:
                        # We'll hit this case of we tried all of the
                        # predictions, and none worked
                        graph.setNodeColor(cur_node, "red")
                        break  # Break from episode

                context_after = coq.tactic_context(coq.local_lemmas[:-1])
                transition = assign_reward(context_before, context_after,
                                           action)
                cur_node = graph.addTransition(cur_node, action,
                                               transition.reward)
                transition.graph_node = cur_node
                replay_memory.append(transition)
                proof_contexts_seen.append(proof_context_after)

                if coq.goals == "":
                    graph.mkQED(cur_node)
                    break

            with print_time("Assigning scores", guard=args.verbose):
                transition_samples = sample_batch(replay_memory,
                                                  args.batch_size)
                training_samples = assign_scores(transition_samples,
                                                 q_estimator, predictor,
                                                 args.num_predictions,
                                                 gamma,
                                                 # Passing this graph
                                                 # in so we can
                                                 # maintain a record
                                                 # of the most recent
                                                 # q score estimates
                                                 # in the graph
                                                 graph)
            with print_time("Training", guard=args.verbose):
                q_estimator.train(training_samples)

            # Clean up episode
            coq.run_stmt("Admitted.")
            coq.run_stmt(f"Reset {lemma_name}.")
            coq.run_stmt(lemma_statement)
        graph.draw("reinforce.png")
        q_estimator.save_weights(args.out_weights, args)


@dataclass
class LabeledTransition:
    before: TacticContext
    after: TacticContext
    action: str
    reward: float
    graph_node: Optional[LabeledNode]


def sample_batch(transitions: List[LabeledTransition], k: int) -> \
      List[LabeledTransition]:
    return random.sample(transitions, k)


def assign_reward(before: TacticContext, after: TacticContext, tactic: str) \
      -> LabeledTransition:
    if after.goal == "":
        reward = 1000.0
    else:
        goal_size_reward = len(tokenizer.get_words(before.goal)) - \
            len(tokenizer.get_words(after.goal))
        num_hyps_reward = len(before.hypotheses) - len(after.hypotheses)
        reward = goal_size_reward * 3 + num_hyps_reward
    return LabeledTransition(before, after, tactic, reward, None)


def assign_rewards(transitions: List[dataloader.ScrapedTransition]) -> \
      List[LabeledTransition]:
    def generate() -> Iterator[LabeledTransition]:
        for transition in transitions:
            yield assign_reward(context_r2py(transition.before),
                                context_r2py(transition.after),
                                transition.tactic)

    return list(generate())


def assign_scores(transitions: List[LabeledTransition],
                  q_estimator: QEstimator,
                  predictor: tactic_predictor.TacticPredictor,
                  num_predictions: int,
                  discount: float,
                  graph: ReinforceGraph) -> \
                  List[Tuple[TacticContext, str, float]]:
    def generate() -> Iterator[Tuple[TacticContext, str, float]]:
        predictions = predictor.predictKTactics_batch(  # type: ignore
            [transition.after for transition in transitions],
            num_predictions)
        for transition, predictions in zip(transitions, predictions):
            ctxt = transition.after
            new_q = transition.reward + \
                discount * max(q_estimator([(ctxt, prediction.prediction)
                                            for prediction in predictions]))
            assert transition.reward == transition.reward
            assert discount == discount
            assert new_q == new_q
            if transition.graph_node:
                graph.setNodeApproxQScore(transition.graph_node, new_q)
            yield transition.before, transition.action, new_q
    return list(generate())


def context_r2py(r_context: dataloader.ProofContext) -> TacticContext:
    return TacticContext(r_context.lemmas, r_context.tactics,
                         r_context.hyps, r_context.goal)


if __name__ == "__main__":
    main()
