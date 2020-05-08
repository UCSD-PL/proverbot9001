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
import re
import random
import sys

import serapi_instance
import dataloader
import tokenizer
from models import tactic_predictor
import predict_tactic
from util import maybe_cuda, eprint, print_time, nostderr
from models.components import WordFeaturesEncoder, DNNScorer

from dataclasses import dataclass
from typing import List, Tuple, Iterator, TypeVar, Dict, Optional
from format import TacticContext
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from torch import optim
from pathlib_revised import Path2

def main(arg_list : List[str]) -> None:
    parser = argparse.ArgumentParser(
        description="A module for exploring deep Q learning with proverbot9001")

    parser.add_argument("scrape_file")

    parser.add_argument("environment_file", type=Path2)
    parser.add_argument("--proof", default=None)

    parser.add_argument("--prelude", default=".", type=Path2)

    parser.add_argument("--predictor-weights", default=Path2("data/polyarg-weights.dat"),
                        type=Path2)
    parser.add_argument("--num-predictions", default=16, type=int)

    parser.add_argument("--buffer-size", default=256, type=int)
    parser.add_argument("--batch-size", default=32, type=int)

    parser.add_argument("--num-episodes", default=256, type=int)
    parser.add_argument("--episode-length", default=16, type=int)

    parser.add_argument("--learning-rate", default=0.5, type=float)

    parser.add_argument("--progress", "-P", action='store_true')
    parser.add_argument("--verbose", "-v", action='count', default=0)

    args = parser.parse_args()

    reinforce(args)

import pygraphviz as pgv
@dataclass(init=True)
class LabeledNode:
    action : str
    reward : float
    node_id : int
    parent : Optional["LabeledNode"]
    children : List["LabeledNode"]

class ReinforceGraph:
    __graph : pgv.AGraph
    __next_node_id : int
    start_node : LabeledNode
    def __init__(self, lemma_name : str) -> None:
        self.__graph = pgv.AGraph(directed=True)
        self.__next_node_id = 0
        self.start_node = self.mkNode(lemma_name, 0, None)
        pass
    def addTransition(self, src : LabeledNode, action : str, reward : float, **kwargs) -> LabeledNode:
        for child in src.children:
            if child.action == action:
                assert child.reward == reward
                return child
        return self.mkNode(action, reward, src, **kwargs)
    def mkNode(self, action: str, reward: float, previous_node : Optional[LabeledNode],
               **kwargs) -> LabeledNode:
        if reward > 0:
            color = "palegreen"
        elif reward < 0:
            color = "indianred1"
        else:
            color = "white"
        # assert action.strip() != "intros.", (color, reward)
        self.__graph.add_node(self.__next_node_id, label=action + "\n" + str(reward),
                              fillcolor=color, style="filled",
                              **kwargs)
        self.__next_node_id += 1
        newNode = LabeledNode(action, reward, self.__next_node_id-1, previous_node, [])
        if previous_node:
            self.__graph.add_edge(previous_node.node_id, newNode.node_id, **kwargs)
            previous_node.children.append(newNode)
        return newNode
    def mkQED(self, src : LabeledNode):
        qedNode = self.mkNode("QED",
                              src,
                              fillcolor="green", style="filled")
        cur_node = predictionNode
        cur_path = []
        while cur_node != self.start_node:
            self.setNodeColor(cur_node, "palegreen1")
            cur_path.append(cur_node)
            assert cur_node.parent
            cur_node = cur_node.parent
        pass
    def setNodeColor(self, node : LabeledNode, color : str) -> None:
        node_handle = self.__graph.get_node(node.node_id)
        node_handle.attr["fillcolor"] = color
        node_handle.attr["style"] = "filled"
    def draw(self, filename : str) -> None:
        with nostderr():
            self.__graph.draw(filename, prog="dot")

def reinforce(args : argparse.Namespace) -> None:

    # Load the scraped (demonstrated) samples, the proof environment
    # commands, and the predictor
    replay_memory = assign_rewards(
        dataloader.tactic_transitions_from_file(args.scrape_file,
                                                args.buffer_size))
    env_commands = serapi_instance.load_commands_preserve(args, 0, args.prelude / args.environment_file)
    predictor = predict_tactic.loadPredictorByFile(args.predictor_weights)

    q_estimator = FeaturesQEstimator(args.learning_rate)
    epsilon = 0.3
    gamma = 0.9

    with serapi_instance.SerapiContext(
            ["sertop", "--implicit"],
            serapi_instance.get_module_from_filename(args.environment_file),
            str(args.prelude)) as coq:
        coq.quiet = True
        coq.verbose = args.verbose
        ## Get us to the correct proof context
        rest_commands, run_commands = coq.run_into_next_proof(env_commands)
        lemma_statement = run_commands[-1]
        if args.proof != None:
            while coq.cur_lemma_name != args.proof:
                if not rest_commands:
                    eprint("Couldn't find lemma {args.proof}! Exiting...")
                    return
                rest_commands, _ = coq.finish_proof(rest_commands)
                rest_commands, _ = coq.run_into_next_proof(env_commands)
        else:
            # Don't use lemmas without names (e.g. "Obligation")
            while coq.cur_lemma_name == "":
                if not rest_commands:
                    eprint("Couldn't find usable lemma! Exiting...")
                    return
                rest_commands, _ = coq.finish_proof(rest_commands)
                rest_commands, _ = coq.run_into_next_proof(env_commands)

        lemma_name = coq.cur_lemma_name

        graph = ReinforceGraph(lemma_name)

        for episode in trange(args.num_episodes, disable=(not args.progress)):
            cur_node = graph.start_node
            proof_contexts_seen = [coq.proof_context]
            for t in trange(args.episode_length, disable=(not args.progress), leave=False):
                with print_time("Getting predictions", guard=args.verbose):
                    context_before = coq.tactic_context(coq.local_lemmas[:-1])
                    predictions = predictor.predictKTactics(context_before, args.num_predictions)
                if random.random() < epsilon:
                    ordered_actions = [p.prediction for p in
                                       random.sample(predictions, len(predictions))]
                    action = random.choice(predictions).prediction
                else:
                    with print_time("Picking actions using q_estimator", guard=args.verbose):
                        q_choices = zip(q_estimator([(context_before,
                                                      prediction.prediction)
                                                     for prediction in predictions]),
                                        [p.prediction for p in predictions])
                        ordered_actions = [p[1] for p in
                                           sorted(q_choices, key=lambda q: q[0], reverse=True)]

                with print_time("Running actions", guard=args.verbose):
                    for try_action in ordered_actions:
                        try:
                            coq.run_stmt(try_action)
                            proof_context_after = coq.proof_context
                            if any([serapi_instance.contextSurjective(proof_context_after,
                                                                      path_context)
                                    for path_context in proof_contexts_seen]):
                                continue
                            action = try_action
                            break
                        except (serapi_instance.ParseError, serapi_instance.CoqExn):
                            pass

                context_after = coq.tactic_context(coq.local_lemmas[:-1])
                transition = assign_reward(context_before, context_after, action)
                cur_node = graph.addTransition(cur_node, action, transition.reward)
                replay_memory.append(transition)
                proof_contexts_seen.append(proof_context_after)

            with print_time("Assigning scores", guard=args.verbose):
                transition_samples = sample_batch(replay_memory, args.batch_size)
                training_samples = assign_scores(transition_samples,
                                                 q_estimator, predictor,
                                                 args.num_predictions,
                                                 gamma)
            with print_time("Training", guard=args.verbose):
                q_estimator.train(training_samples)

            # Clean up episode
            coq.run_stmt("Admitted.")
            coq.run_stmt(f"Reset {lemma_name}.")
            coq.run_stmt(lemma_statement)
        graph.draw("reinforce.png")

@dataclass
class LabeledTransition:
    before : dataloader.ProofContext
    after : dataloader.ProofContext
    action : str
    reward : float

def sample_batch(transitions: List[LabeledTransition], k: int) -> List[LabeledTransition]:
    return random.sample(transitions, k)

def assign_reward(before: TacticContext, after: TacticContext, tactic: str) -> LabeledTransition:
    if after.goal == "":
        reward = 1000.0
    else:
        goal_size_reward = len(tokenizer.get_words(before.goal)) - \
            len(tokenizer.get_words(after.goal))
        num_hyps_reward = len(before.hypotheses) - len(after.hypotheses)
        reward = goal_size_reward * 3 + num_hyps_reward
    return LabeledTransition(before, after, tactic, reward)


def assign_rewards(transitions : List[dataloader.ScrapedTransition]) -> \
    List[LabeledTransition]:
    def generate() -> Iterator[LabeledTransition]:
        for transition in transitions:
            yield assign_reward(context_r2py(transition.before),
                                context_r2py(transition.after),
                                transition.tactic)

    return list(generate())

class QEstimator(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, state: TacticContext, action: str) -> float:
        pass
    @abstractmethod
    def train(self, samples: List[Tuple[TacticContext, str, float]]) -> None:
        pass

def assign_scores(transitions: List[LabeledTransition],
                  q_estimator: QEstimator,
                  predictor: tactic_predictor.TacticPredictor,
                  num_predictions: int,
                  discount : float) -> List[Tuple[TacticContext, str, float]]:
    def generate() -> Iterator[Tuple[dataloader.ProofContext, str, float]]:
        predictions = predictor.predictKTactics_batch([transition.after for transition in transitions],
                                                      num_predictions)
        for transition, predictions in zip(transitions, predictions):
            ctxt = transition.after
            new_q = transition.reward + \
                discount * max(q_estimator([(ctxt, prediction.prediction)
                                            for prediction in predictions]))
            yield transition.before, transition.action, new_q
    return list(generate())

def context_r2py(r_context : dataloader.ProofContext) -> TacticContext:
    return TacticContext(r_context.lemmas, r_context.tactics,
                         r_context.hyps, r_context.goal)

class FeaturesQEstimator(QEstimator):
    def __init__(self, learning_rate: float) -> None:
        self.model = FeaturesQModel(32, 128,
                                    2, 128, 3)
        self.optimizer = optim.SGD(self.model.parameters(), learning_rate)
        self.criterion = nn.MSELoss()
        self.tactic_map = {}
        self.token_map = {}
        pass
    def __call__(self, inputs: List[Tuple[TacticContext, str]]) -> List[float]:
        state_word_features_batch, vec_features_batch \
            = zip(*[self._features(state) for (state, action) in inputs])
        encoded_actions_batch = [self._encode_action(state, action)
                                 for (state, action) in inputs]
        all_word_features_batch = [list(encoded_action) + state_word_features
                                   for encoded_action, state_word_features in
                                   zip(encoded_actions_batch,
                                       state_word_features_batch)]
        output = self.model(torch.LongTensor(all_word_features_batch),
                            torch.FloatTensor(vec_features_batch))
        return list(output)
    def train(self, samples: List[Tuple[TacticContext, str, float]]) -> None:
        self.optimizer.zero_grad()
        state_word_features, vec_features = zip(*[self._features(state) for state, _, _ in samples])
        encoded_actions = [self._encode_action(state, action) for state, action, _ in samples]
        all_word_features = [list(ea) + swf for ea, swf in zip(encoded_actions, state_word_features)]
        outputs = self.model(torch.LongTensor(all_word_features),
                             torch.FloatTensor(vec_features))
        expected_outputs = maybe_cuda(torch.FloatTensor([output for _, _, output in samples]))
        loss = self.criterion(outputs, expected_outputs)
        loss.backward()
        self.optimizer.step()
    def _features(self, context: TacticContext) -> Tuple[List[int], List[float]]:
        if len(context.prev_tactics) > 0:
            prev_tactic = serapi_instance.get_stem(context.prev_tactics[-1])
            prev_tactic_index = emap_lookup(self.tactic_map, 32, prev_tactic)
        else:
            prev_tactic_index = 0
        if context.goal != "":
            goal_head_index = emap_lookup(self.token_map, 128, tokenizer.get_words(context.goal)[0])
        else:
            goal_head_index = 0
        goal_length_feature = min(len(tokenizer.get_words(context.goal)), 100) / 100
        num_hyps_feature = min(len(context.hypotheses), 30) / 30
        return [prev_tactic_index, goal_head_index], [goal_length_feature, num_hyps_feature]
    def _encode_action(self, context: TacticContext, action: str) -> Tuple[int, int]:
        stem, argument = serapi_instance.split_tactic(action)
        stem_idx = emap_lookup(self.tactic_map, 32, stem)
        all_premises = context.hypotheses + context.relevant_lemmas
        stripped_arg = argument.strip(".").strip()
        if stripped_arg == "":
            arg_idx = 0
        else:
            index_hyp_vars = dict(serapi_instance.get_indexed_vars_in_hyps(all_premises))
            if stripped_arg in index_hyp_vars:
                hyp_varw, _, rest = all_premises[index_hyp_vars[stripped_arg]].partition(":")
                arg_idx = emap_lookup(self.token_map, 128, tokenizer.get_words(rest)[0]) + 2
            else:
                goal_symbols = tokenizer.get_symbols(context.goal)
                if stripped_arg in goal_symbols:
                    arg_idx = emap_lookup(self.token_map, 128, stripped_arg) + 128 + 2
                else:
                    arg_idx = 1
        return stem_idx, arg_idx

T = TypeVar('T')

def emap_lookup(emap: Dict[T, int], size: int, item: T):
    if item in emap:
        return emap[item]
    elif len(emap) < size - 1:
        emap[item] = len(emap) + 1
        return emap[item]
    else:
        return 0

class FeaturesQModel(nn.Module):
    def __init__(self,
                 num_tactics : int,
                 num_tokens : int,
                 vec_features_size : int,
                 hidden_size : int,
                 num_layers : int) -> None:
        super().__init__()
        # Consider making the word embedding the same for all token-type inputs, also for tactic-type inputs
        self._word_features_encoder = maybe_cuda(
            WordFeaturesEncoder([num_tactics, num_tokens * 2 + 2,
                                 num_tactics, num_tokens],
                                hidden_size, 1, hidden_size))
        self._features_classifier = maybe_cuda(
            DNNScorer(hidden_size + vec_features_size,
                      hidden_size, num_layers))
    def forward(self,
                word_features_batch : torch.LongTensor,
                vec_features_batch : torch.FloatTensor) -> torch.FloatTensor:
        encoded_word_features = self._word_features_encoder(
            maybe_cuda(word_features_batch))
        scores = self._features_classifier(
            torch.cat((encoded_word_features, maybe_cuda(vec_features_batch)), dim=1))\
        .view(vec_features_batch.size()[0])
        return scores

if __name__ == "__main__":
    main(sys.argv[1:])
