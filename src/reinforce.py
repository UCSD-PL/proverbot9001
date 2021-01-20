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

from __future__ import annotations
import argparse
import random
import os
import errno
import signal
import traceback
import re
import json
import multiprocessing
from threading import Lock
import sys
import functools
from dataclasses import dataclass
from queue import Queue
import queue
from typing import List, Tuple, Iterator, Optional, cast, TYPE_CHECKING, Dict, Any
if TYPE_CHECKING:
    from multiprocessing.sharedctypes import _Value
    Value = _Value
else:
    Value = int


import torch.multiprocessing as tmp
from torch.multiprocessing import Manager
import torch
from pathlib_revised import Path2
import pygraphviz as pgv
from tqdm import trange, tqdm

import serapi_instance
import dataloader
from models import tactic_predictor
from models.features_q_estimator import FeaturesQEstimator
from models.q_estimator import QEstimator
from models import features_polyarg_predictor
import predict_tactic
import util
from util import eprint, print_time, nostderr, unwrap, progn

from format import (TacticContext, ProofContext, Obligation, truncate_tactic_context)


def main() -> None:
    parser = \
        argparse.ArgumentParser(
            description="A module for exploring deep Q learning "
            "with proverbot9001")

    parser.add_argument("scrape_file")

    parser.add_argument("out_weights", type=Path2)
    parser.add_argument("environment_files", type=Path2, nargs="+")
    parser.add_argument("-j", "--num-threads", type=int, default=5)
    parser.add_argument("--proof", default=None)

    parser.add_argument("--prelude", default=".", type=Path2)

    parser.add_argument("--predictor-weights",
                        default=Path2("data/polyarg-weights.dat"),
                        type=Path2)
    parser.add_argument("--start-from", default=None, type=Path2)
    parser.add_argument("--num-predictions", default=16, type=int)

    parser.add_argument("--buffer-min-size", default=256, type=int)
    parser.add_argument("--buffer-max-size", default=32768, type=int)
    parser.add_argument("--batch-size", default=32, type=int)

    parser.add_argument("--num-episodes", default=256, type=int)
    parser.add_argument("--episode-length", default=16, type=int)

    parser.add_argument("--learning-rate", default=0.0001, type=float)
    parser.add_argument("--batch-step", default=50, type=int)
    parser.add_argument("--gamma", default=0.8, type=float)
    parser.add_argument("--exploration-factor", default=0.3, type=float)
    parser.add_argument("--time-discount", default=0.9, type=float)

    parser.add_argument("--max-term-length", default=512, type=float)

    parser.add_argument("--pretrain-epochs", default=10, type=int)
    parser.add_argument("--no-pretrain", action='store_false',
                        dest='pretrain')
    parser.add_argument("--include-proof-relevant", action="store_true")

    parser.add_argument("--progress", "-P", action='store_true')
    parser.add_argument("--verbose", "-v", action='count', default=0)
    parser.add_argument("--log-anomalies", type=Path2, default=None)
    parser.add_argument("--log-outgoing-messages", type=Path2, default=None)

    parser.add_argument("--hardfail", action="store_true")

    parser.add_argument("--ghosts", action='store_true')
    parser.add_argument("--graphs-dir", default=Path2("graphs"), type=Path2)

    parser.add_argument("--success-repetitions", default=10, type=int)
    parser.add_argument("--careful", action='store_true')
    parser.add_argument("--train-every", default=128, type=int)

    args = parser.parse_args()

    try:
        os.makedirs(str(args.graphs_dir))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    reinforce_multithreaded(args)


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


Job = Tuple[str, str, str]


@dataclass
class LabeledTransition:
    relevant_lemmas: List[str]
    prev_tactics: List[str]
    before: ProofContext
    after: ProofContext
    action: str
    original_certainty: float
    reward: float
    graph_node: Optional[LabeledNode]

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


def reinforce_multithreaded(args: argparse.Namespace) -> None:

    # Load the predictor
    predictor = predict_tactic.loadPredictorByFile(args.predictor_weights)

    # Load the scraped (demonstrated) samples and the proof
    # environment commands. Assigns them an estimated "original
    # predictor certainty" value for use as a feature.
    # Create an initial Q Estimator
    q_estimator = FeaturesQEstimator(args.learning_rate,
                                     args.batch_step,
                                     args.gamma)
    # This sets up a handler so that if the user hits Ctrl-C, we save
    # the weights as we have them and exit.
    signal.signal(
        signal.SIGINT,
        lambda signal, frame:
        progn(q_estimator.save_weights(
            args.out_weights, args),  # type: ignore

              exit()))

    resume_file = args.out_weights.with_suffix('.tmp')
    if resume_file.exists():
        eprint("Looks like there was a session in progress for these weights! Resuming")
        q_estimator_name, *saved = \
            torch.load(args.out_weights)
        q_estimator.load_saved_state(*saved)
        replay_memory = []
        with resume_file.open('r') as f:
            for line in f:
                replay_memory.append(LabeledTransition.from_dict(
                    json.loads(line)))
        already_done = []
        with args.out_weights.with_suffix('.done').open('r') as f:
            for line in f:
                next_done = json.loads(line)
                already_done.append((Path2(next_done[0]), next_done[1],
                                     next_done[2]))
    else:
        with print_time("Loading initial samples from labeled data"):
            replay_memory = assign_rewards(args, predictor,
                                           dataloader.tactic_transitions_from_file(
                                               args.scrape_file, args.buffer_min_size))
        # Load in any starting weights
        if args.start_from:
            q_estimator_name, *saved = \
                torch.load(args.start_from)
            q_estimator.load_saved_state(*saved)
        elif args.pretrain:
            # Pre-train the q scores to zero
            with print_time("Pretraining"):
                pre_train(args, predictor, q_estimator,
                          dataloader.tactic_transitions_from_file(
                              args.scrape_file, args.buffer_min_size * 3))
        already_done = []
        with args.out_weights.with_suffix('.tmp').open('w') as f:
            for sample in replay_memory:
                f.write(json.dumps(sample.to_dict()))
                f.write("\n")
        with args.out_weights.with_suffix('.done').open('w'):
            pass

    ctxt = tmp.get_context('spawn')
    jobs: Queue[Job] = ctxt.Queue()
    done: Queue[Job] = ctxt.Queue()
    samples: Queue[LabeledTransition] = ctxt.Queue()

    # eprint(f"Starting with {len(replay_memory)} samples")
    for sample in replay_memory:
        samples.put(sample)

    with tmp.Pool() as pool:
        jobs_in_files = pool.map(functools.partial(get_proofs, args),
                                 list(enumerate(args.environment_files)))
    all_jobs = [job for job_list in jobs_in_files for job in job_list if job not in already_done]

    for job in all_jobs:
        jobs.put(job)

    with Manager() as manager:
        manager = cast(multiprocessing.managers.SyncManager, manager)
        ns = manager.Namespace()
        ns.predictor = predictor
        ns.estimator = q_estimator
        lock = manager.Lock()

        training_worker = ctxt.Process(
            target=reinforce_training_worker,
            args=(args, len(replay_memory), lock, ns, samples))
        workers = [ctxt.Process(
            target=reinforce_worker,
            args=(widx,
                  args,
                  lock,
                  ns,
                  samples,
                  jobs,
                  done))
                   for widx in range(min(args.num_threads, len(all_jobs)))]
        training_worker.start()
        for worker in workers:
            worker.start()

        with tqdm(total=len(all_jobs) + len(already_done), dynamic_ncols=True) as bar:
            bar.update(len(already_done))
            bar.refresh()
            for _ in range(len(all_jobs)):
                job = done.get()
                bar.update()
                with args.out_weights.with_suffix(".done").open('a') as f:
                    f.write(json.dumps((str(job[0]), job[1], job[2])))
                    f.write("\n")

        for worker in workers:
            worker.join()
        args.out_weights.with_suffix('.tmp').unlink()
        training_worker.join()


def reinforce_worker(worker_idx: int,
                     args: argparse.Namespace,
                     lock: Lock,
                     namespace: multiprocessing.managers.Namespace,
                     samples: Queue[LabeledTransition],
                     jobs: Queue[Job],
                     done: Queue[Job]):

    sys.setrecursionlimit(100000)
    failing_lemma = ""

    try:
        next_file, next_module, next_lemma = jobs.get_nowait()
    except queue.Empty:
        return
    with util.silent():
        all_commands = serapi_instance.load_commands_preserve(
            args, worker_idx + 1, args.prelude / next_file)

    rest_commands = all_commands
    while rest_commands:
        with serapi_instance.SerapiContext(["sertop", "--implicit"],
                                           serapi_instance.
                                           get_module_from_filename(next_file),
                                           str(args.prelude)) as coq:
            coq.quiet = True
            coq.verbose = args.verbose

            while next_lemma:
                try:
                    rest_commands, run_commands = coq.run_into_next_proof(
                        rest_commands)
                    if not rest_commands:
                        eprint(f"Couldn't find lemma {next_lemma}!")
                        break
                except serapi_instance.CoqAnomaly:
                    with util.silent():
                        all_commands = serapi_instance.\
                            load_commands_preserve(
                                args, 0, args.prelude / next_file)
                        rest_commands = all_commands
                    break
                except serapi_instance.SerapiException:
                    eprint(f"Failed getting to before: {next_lemma}")
                    eprint(f"In file {next_file}")
                    raise
                lemma_statement = run_commands[-1]
                if lemma_statement == next_lemma:
                    try:
                        reinforce_lemma_multithreaded(args, coq,
                                                      lock, namespace,
                                                      worker_idx,
                                                      samples,
                                                      next_lemma,
                                                      next_module)
                    except serapi_instance.CoqAnomaly:
                        if args.hardfail:
                            raise
                        if failing_lemma == lemma_statement:
                            eprint("Hit the same anomaly twice! Skipping",
                                   guard=args.verbose >= 1)
                            done.put((next_file, next_module, next_lemma))

                            try:
                                new_file, next_module, next_lemma = jobs.get_nowait()
                            except queue.Empty:
                                return
                            if new_file != next_file:
                                next_file = new_file
                                with util.silent():
                                    all_commands = serapi_instance.\
                                        load_commands_preserve(
                                            args, 0, args.prelude / next_file)
                                    rest_commands = all_commands
                                    break
                            else:
                                rest_commands = all_commands
                        else:
                            rest_commands = all_commands
                            failing_lemma = lemma_statement
                        break
                    except Exception as e:
                        if worker_idx == 0:
                            eprint(
                                f"FAILED in file {next_file}, lemma {next_lemma}")
                            eprint(e)
                        raise
                    serapi_instance.admit_proof(coq, lemma_statement)
                    while not serapi_instance.ending_proof(rest_commands[0]):
                        rest_commands = rest_commands[1:]
                    rest_commands = rest_commands[1:]
                    done.put((next_file, next_module, next_lemma))
                    try:
                        new_file, next_module, next_lemma = jobs.get_nowait()
                    except queue.Empty:
                        return

                    if new_file != next_file:
                        next_file = new_file
                        with util.silent():
                            all_commands = serapi_instance.\
                                load_commands_preserve(
                                    args, 0,
                                    args.prelude / next_file)
                        rest_commands = all_commands
                        break
                else:
                    proof_relevant = False
                    for cmd in rest_commands:
                        if serapi_instance.ending_proof(cmd):
                            if cmd.strip() == "Defined.":
                                proof_relevant = True
                            break
                    proof_relevant = proof_relevant or \
                        bool(re.match(
                            r"\s*Derive",
                            serapi_instance.kill_comments(lemma_statement))
                             ) or\
                        bool(re.match(
                            r"\s*Let",
                            serapi_instance.kill_comments(lemma_statement))
                             ) or\
                        bool(re.match(
                            r"\s*Equations",
                            serapi_instance.kill_comments(lemma_statement))
                             ) or\
                        args.careful
                    if proof_relevant:
                        rest_commands, run_commands = coq.finish_proof(
                            rest_commands)
                    else:
                        try:
                            serapi_instance.admit_proof(coq, lemma_statement)
                        except serapi_instance.SerapiException:
                            next_lemma_name = \
                                serapi_instance.\
                                lemma_name_from_statement(next_lemma)
                            eprint(
                                f"{next_file}: Failed to admit proof {next_lemma_name}")
                            raise

                        while not serapi_instance.ending_proof(
                                rest_commands[0]):
                            rest_commands = rest_commands[1:]
                        rest_commands = rest_commands[1:]


def reinforce_lemma_multithreaded(
        args: argparse.Namespace,
        coq: serapi_instance.SerapiInstance,
        lock: Lock,
        namespace: multiprocessing.managers.Namespace,
        worker_idx: int,
        samples: Queue[LabeledTransition],
        lemma_statement: str,
        _module_prefix: str):

    lemma_name = serapi_instance.lemma_name_from_statement(lemma_statement)
    graph = ReinforceGraph(lemma_name)
    lemma_memory = []
    for _ in trange(args.num_episodes, disable=(not args.progress),
                    leave=False, position=worker_idx + 1):
        cur_node = graph.start_node
        proof_contexts_seen = [unwrap(coq.proof_context)]
        episode_memory: List[LabeledTransition] = []
        for _ in range(args.episode_length):
            with print_time("Getting predictions", guard=args.verbose):
                context_before = coq.tactic_context(coq.local_lemmas[:-1])
                proof_context_before = unwrap(coq.proof_context)
                with lock:
                    eprint(f"Locked in thread {worker_idx}",
                           guard=args.verbose >= 2)
                    predictor = namespace.predictor
                    estimator = namespace.estimator
                    with print_time("Making predictions", guard=args.verbose >= 3):
                        context_trunced = truncate_tactic_context(
                            context_before, args.max_term_length)
                        predictions = predictor.predictKTactics(
                            context_trunced, args.num_predictions)
                    if random.random() < args.exploration_factor:
                        ordered_actions = list(random.sample(predictions,
                                                             len(predictions)))
                    else:
                        with print_time("Picking actions with q_estimator",
                                        guard=args.verbose):
                            q_choices = zip(estimator(
                                [(context_before, p.prediction, p.certainty)
                                 for p in predictions]),
                                            predictions)
                            ordered_actions = [p[1] for p in
                                               sorted(q_choices,
                                                      key=lambda q: q[0],
                                                      reverse=True)]
                eprint(f"Unlocked in thread {worker_idx}",
                       guard=args.verbose >= 2)
            with print_time("Running actions", guard=args.verbose):
                action = None
                original_certainty = None
                for try_action, try_original_certainty in ordered_actions:
                    try:
                        coq.run_stmt(try_action)
                        proof_context_after = unwrap(coq.proof_context)
                        if any([serapi_instance.contextSurjective(
                                proof_context_after, path_context)
                                for path_context in proof_contexts_seen]):
                            coq.cancel_last()
                            transition = assign_failed_reward(
                                context_before.relevant_lemmas,
                                context_before.prev_tactics,
                                proof_context_before,
                                proof_context_after,
                                try_action,
                                try_original_certainty,
                                -1)
                            assert transition.reward < 2000
                            # eprint(f"Pushing one sample in thread {worker_idx}")
                            samples.put(transition)
                            if args.ghosts:
                                ghost_node = graph.addGhostTransition(
                                    cur_node, try_action)
                                transition.graph_node = ghost_node
                            continue
                        action = try_action
                        original_certainty = try_original_certainty
                        break
                    except (serapi_instance.ParseError,
                            serapi_instance.CoqExn,
                            serapi_instance.TimeoutError):
                        transition = assign_failed_reward(
                            context_before.relevant_lemmas,
                            context_before.prev_tactics,
                            proof_context_before,
                            proof_context_before,
                            try_action,
                            try_original_certainty,
                            -5)
                        assert transition.reward < 2000
                        # eprint(f"Pushing one sample in thread {worker_idx}")
                        samples.put(transition)
                        if args.ghosts:
                            ghost_node = graph.addGhostTransition(cur_node,
                                                                  try_action)
                            transition.graph_node = ghost_node
                if action is None:
                    # We'll hit this case of we tried all of the
                    # predictions, and none worked
                    graph.setNodeColor(cur_node, "red")
                    break  # Break from episode
            transition = assign_reward(context_before.relevant_lemmas,
                                       context_before.prev_tactics,
                                       proof_context_before,
                                       proof_context_after,
                                       action,
                                       unwrap(original_certainty))
            cur_node = graph.addTransition(cur_node, action,
                                           transition.reward)
            transition.graph_node = cur_node
            assert transition.reward < 2000
            # eprint(f"Pushing one sample in thread {worker_idx}")
            samples.put(transition)

            lemma_memory += episode_memory
            if coq.goals == "":
                graph.mkQED(cur_node)
                for sample in (episode_memory *
                               (args.success_repetitions - 1)):
                    samples.put(sample)
                break

        # Clean up episode
        coq.run_stmt("Admitted.")
        coq.run_stmt(f"Reset {lemma_name}.")
        coq.run_stmt(lemma_statement)

        # Write out lemma memory to progress file for resuming
        for sample in lemma_memory:
            with args.out_weights.with_suffix('.tmp').open('a') as f:
                f.write(json.dumps(sample.to_dict()))
                f.write("\n")
    graphpath = (args.graphs_dir / lemma_name).with_suffix(".png")
    graph.draw(str(graphpath))


def reinforce_training_worker(args: argparse.Namespace,
                              initial_buffer_size: int,
                              lock: Lock,
                              namespace: multiprocessing.managers.Namespace,
                              samples: Queue[LabeledTransition]):
    memory: List[LabeledTransition] = []
    while True:
        next_sample = samples.get()
        memory.append(next_sample)
        if len(memory) > args.buffer_max_size:
            del memory[0:args.train_every+1]
        if len(memory) > initial_buffer_size and \
           (len(memory) - initial_buffer_size) % args.train_every == 0:
            transition_samples = sample_batch(memory, args.batch_size)
            with lock:
                eprint(
                    f"Locked in training thread for {len(memory)} samples",
                    guard=args.verbose >= 2)
                q_estimator = namespace.estimator
                predictor = namespace.predictor
                with print_time("Assigning scores", guard=args.verbose):
                    training_samples = assign_scores(args,
                                                     transition_samples,
                                                     q_estimator,
                                                     predictor)
                with print_time("Training", guard=args.verbose):
                    q_estimator.train(training_samples)
                q_estimator.save_weights(args.out_weights, args)
                eprint("Unlocked in training thread",
                       guard=args.verbose >= 2)

    pass


def get_proofs(args: argparse.Namespace,
               t: Tuple[int, str]) -> List[Tuple[str, str, str]]:
    idx, filename = t
    cmds = serapi_instance.load_commands_preserve(
        args, idx, args.prelude / filename)
    return [(filename, module, cmd) for module, cmd in
            serapi_instance.lemmas_in_file(
                filename, cmds, args.include_proof_relevant)]
def sample_batch(transitions: List[LabeledTransition], k: int) -> \
      List[LabeledTransition]:
    return random.sample(transitions, k)


def assign_failed_reward(relevant_lemmas: List[str], prev_tactics: List[str],
                         before: ProofContext, after: ProofContext,
                         tactic: str, certainty: float, reward: int) \
                         -> LabeledTransition:
    return LabeledTransition(relevant_lemmas, prev_tactics, before, after,
                             tactic, certainty, reward, None)


def assign_reward(relevant_lemmas: List[str], prev_tactics: List[str],
                  before: ProofContext, after: ProofContext, tactic: str,
                  certainty: float) \
      -> LabeledTransition:
    goals_changed = len(after.all_goals) - len(before.all_goals)
    if len(after.all_goals) == 0:
        reward = 10.0
    elif goals_changed != 0:
        reward = -(goals_changed * 2.0)
    else:
        reward = 0
    # else:
    #     goal_size_reward = len(tokenizer.get_words(before.focused_goal)) - \
    #         len(tokenizer.get_words(after.focused_goal))
    #     num_hyps_reward = len(before.focused_hyps) - len(after.focused_hyps)
    #     reward = goal_size_reward * 3 + num_hyps_reward
    return LabeledTransition(relevant_lemmas, prev_tactics, before, after,
                             tactic, certainty, reward, None)


def assign_rewards(args: argparse.Namespace,
                   predictor: tactic_predictor.TacticPredictor,
                   transitions: List[dataloader.ScrapedTransition]) -> \
      List[LabeledTransition]:
    def generate() -> Iterator[LabeledTransition]:
        for transition in transitions:
            if len(transition.before.fg_goals) == 0:
                context = TacticContext(transition.relevant_lemmas,
                                        transition.prev_tactics,
                                        [], "")
            else:
                context = TacticContext(transition.relevant_lemmas,
                                        transition.prev_tactics,
                                        transition.before.fg_goals[0].hypotheses,
                                        transition.before.fg_goals[0].goal)
            yield assign_reward(transition.relevant_lemmas,
                                transition.prev_tactics,
                                context_r2py(transition.before),
                                context_r2py(transition.after),
                                transition.tactic,
                                certainty_of(predictor, args.num_predictions * 2,
                                             context,
                                             transition.tactic))
    return list(generate())


def assign_scores(args: argparse.Namespace,
                  transitions: List[LabeledTransition],
                  q_estimator: FeaturesQEstimator,
                  predictor: tactic_predictor.TacticPredictor) -> \
                  List[Tuple[TacticContext, str, float, float]]:
    def generate() -> Iterator[Tuple[TacticContext, str, float, float]]:
        contexts_trunced = [truncate_tactic_context(
            transition.after_context,
            args.max_term_length)
                            for transition in transitions]
        prediction_lists = cast(features_polyarg_predictor
                                .FeaturesPolyargPredictor,
                                predictor) \
                                .predictKTactics_batch(
                                    contexts_trunced,
                                    args.num_predictions)
        for transition, predictions in zip(transitions, prediction_lists):
            tactic_ctxt = transition.after_context

            if len(transition.after.all_goals) == 0:
                new_q = transition.reward
            else:
                estimates = q_estimator(
                    [(tactic_ctxt, prediction.prediction, prediction.certainty)
                     for prediction in predictions])
                estimated_future_q = \
                    args.time_discount * max(estimates)
                estimated_current_q = q_estimator([(transition.before_context,
                                                    transition.action,
                                                    transition.original_certainty)])[0]
                new_q = transition.reward + estimated_future_q \
                    - estimated_current_q

            # if transition.graph_node:
            #     graph.setNodeApproxQScore(transition.graph_node, new_q)
            yield TacticContext(
                transition.relevant_lemmas,
                transition.prev_tactics,
                transition.before.focused_hyps,
                transition.before.focused_goal), \
                transition.action, transition.original_certainty, new_q
    return list(generate())


def obligation_r2py(r_obl: dataloader.Obligation) -> Obligation:
    return Obligation(r_obl.hypotheses, r_obl.goal)


def context_r2py(r_context: dataloader.ProofContext) -> ProofContext:
    return ProofContext(list(map(obligation_r2py, r_context.fg_goals)),
                        list(map(obligation_r2py, r_context.bg_goals)),
                        list(map(obligation_r2py, r_context.shelved_goals)),
                        list(map(obligation_r2py, r_context.given_up_goals)))


def pre_train(args: argparse.Namespace,
              predictor: tactic_predictor.TacticPredictor,
              estimator: QEstimator,
              transitions: List[dataloader.ScrapedTransition]) -> None:
    def gen_samples():
        for transition in transitions:
            if len(transition.before.fg_goals) == 0:
                continue
            context = TacticContext(transition.relevant_lemmas,
                                    transition.prev_tactics,
                                    transition.before.fg_goals[0].hypotheses,
                                    transition.before.fg_goals[0].goal)
            certainty = certainty_of(predictor, args.num_predictions * 2,
                                     context, transition.tactic)
            yield (context, transition.tactic, certainty, 0.0)

    samples = list(gen_samples())
    estimator.train(samples, args.batch_size, args.pretrain_epochs)


def certainty_of(predictor: tactic_predictor.TacticPredictor, k: int,
                 context: TacticContext, tactic: str) -> float:
    predictions = predictor.predictKTactics(context, k)
    for p in predictions:
        if p.prediction == tactic:
            return p.certainty
    return 0.0


if __name__ == "__main__":
    main()
