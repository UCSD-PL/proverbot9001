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
import re
import json
import multiprocessing
import math
from threading import Lock
import sys
import functools
from dataclasses import dataclass
from queue import Queue
import queue
from typing import (List, Tuple, Iterator, Optional,
                    cast, TYPE_CHECKING, Dict, Any,
                    TypeVar)
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

import coq_serapy as serapi_instance
from coq_serapy.contexts import (TacticContext, ProofContext, Obligation,
                                 truncate_tactic_context)
import dataloader
from models import tactic_predictor
from models.polyarg_q_estimator import PolyargQEstimator
from models.features_q_estimator import FeaturesQEstimator
from models.q_estimator import QEstimator
from models import features_polyarg_predictor
import predict_tactic
import util
from util import eprint, print_time, nostderr, unwrap, progn, safe_abbrev


serapi_instance.set_parseSexpOneLevel_fn(util.parseSexpOneLevel)


def main() -> None:
    parser = \
        argparse.ArgumentParser(
            description="A module for exploring deep Q learning "
            "with proverbot9001")

    parser.add_argument("scrape_file")

    parser.add_argument("out_weights", type=Path2)
    parser.add_argument("environment_files", type=Path2, nargs="+")
    parser.add_argument("-j", "--num-threads", type=int, default=5)
    proofsGroup = parser.add_mutually_exclusive_group()
    proofsGroup.add_argument("--proof", default=None)
    proofsGroup.add_argument("--proofs-file", default=None)

    parser.add_argument("--prelude", default=".", type=Path2)

    parser.add_argument("--predictor-weights",
                        default=Path2("data/polyarg-weights.dat"),
                        type=Path2)
    parser.add_argument("--estimator",
                        choices=["polyarg", "features"],
                        default="polyarg")

    parser.add_argument("--start-from", default=None, type=Path2)
    parser.add_argument("--num-predictions", default=16, type=int)

    parser.add_argument("--buffer-min-size", default=256, type=int)
    parser.add_argument("--buffer-max-size", default=32768, type=int)
    parser.add_argument("--batch-size", default=32, type=int)

    parser.add_argument("--num-episodes", default=256, type=int)
    parser.add_argument("--episode-length", default=16, type=int)

    parser.add_argument("--learning-rate", default=0.02, type=float)
    parser.add_argument("--batch-step", default=50, type=int)
    parser.add_argument("--gamma", default=0.8, type=float)
    parser.add_argument("--exploration-factor", default=0.3, type=float)
    parser.add_argument("--time-discount", default=0.9, type=float)

    parser.add_argument("--max-term-length", default=512, type=float)

    parser.add_argument("--pretrain-epochs", default=10, type=int)
    parser.add_argument("--no-pretrain", action='store_false',
                        dest='pretrain')
    parser.add_argument("--include-proof-relevant", action="store_true")
    parser.add_argument("--demonstrate-from", default=None, type=Path2)
    parser.add_argument("--demonstration-steps", default=2, type=int)

    parser.add_argument("--progress", "-P", action='store_true')
    parser.add_argument("--verbose", "-v", action='count', default=0)
    parser.add_argument("--log-anomalies", type=Path2, default=None)
    parser.add_argument("--log-outgoing-messages", type=Path2, default=None)

    parser.add_argument("--hardfail", action="store_true")

    parser.add_argument("--ghosts", action='store_true')
    parser.add_argument("--graphs-dir", default=Path2("graphs"), type=Path2)

    parser.add_argument("--success-repetitions", default=10, type=int)
    parser.add_argument("--careful", action='store_true')
    parser.add_argument("--train-every-min", default=32, type=int)
    parser.add_argument("--train-every-max", default=2048, type=int)
    parser.add_argument("--show-loss", action='store_true')

    args = parser.parse_args()

    try:
        os.makedirs(str(args.graphs_dir))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    reinforce_multithreaded(args)


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

    def assignApproximateQScores(self, args: argparse.Namespace,
                                 predictor: tactic_predictor.TacticPredictor,
                                 estimator: QEstimator,
                                 node: Optional[LabeledNode] = None) -> None:
        if node is None:
            node = self.start_node
        elif node.transition:
            ctxt = truncate_tactic_context(
                node.transition.before_context,
                args.max_term_length)
            score = estimator([(ctxt,
                                node.transition.action,
                                node.transition.original_certainty)])[0]
            self.setNodeApproxQScore(
                node, score)
        for child in node.children:
            self.assignApproximateQScores(
                args, predictor, estimator, child)

    def draw(self, filename: str) -> None:
        graph = pgv.AGraph(directed=True)
        for (nidx, props) in self.graph_nodes:
            graph.add_node(nidx, **props)
        for (a, b, props) in self.graph_edges:
            graph.add_edge(a, b, **props)
        with nostderr():
            graph.draw(filename, prog="dot")


Job = Tuple[Path2, str, str]
Demonstration = List[str]


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

    def resume(resume_file: Path2, weights: Path2) -> \
      Tuple[List[LabeledTransition],
            List[Job]]:
        eprint("Looks like there was a session in progress for these weights! "
               "Resuming")
        q_estimator_name, *saved = \
            torch.load(str(weights))
        q_estimator.load_saved_state(*saved)
        replay_memory = []
        with resume_file.open('r') as f:
            num_samples = sum(1 for _ in f)
        if num_samples > args.buffer_max_size:
            samples_to_use = random.sample(range(num_samples),
                                           args.buffer_max_size)
        else:
            samples_to_use = None
        with resume_file.open('r') as f:
            for (idx, line) in enumerate(f, start=1):
                if num_samples > args.buffer_max_size and \
                  idx not in samples_to_use:
                    continue
                try:
                    replay_memory.append(LabeledTransition.from_dict(
                        json.loads(line)))
                except json.decoder.JSONDecodeError:
                    eprint(f"Problem loading line {idx}: {line}")
                    raise
        already_done = []
        with weights.with_suffix('.done').open('r') as f:
            for line in f:
                next_done = json.loads(line)
                already_done.append((Path2(next_done[0]), next_done[1],
                                     next_done[2]))
        return replay_memory, already_done

    # Load the predictor
    predictor = cast(features_polyarg_predictor.
                     FeaturesPolyargPredictor,
                     predict_tactic.loadPredictorByFile(
                         args.predictor_weights))

    q_estimator: QEstimator
    # Create an initial Q Estimator
    if args.estimator == "polyarg":
        q_estimator = PolyargQEstimator(args.learning_rate,
                                        args.batch_step,
                                        args.gamma,
                                        predictor)
    else:
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
        replay_memory, already_done = resume(resume_file, args.out_weights)
    else:
        # Load the scraped (demonstrated) samples and the proof
        # environment commands. Assigns them an estimated "original
        # predictor certainty" value for use as a feature.
        with print_time("Loading initial samples from labeled data"):
            replay_memory = assign_rewards(
                args, predictor,
                dataloader.tactic_transitions_from_file(
                    predictor.dataloader_args,
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
                              predictor.dataloader_args,
                              args.scrape_file, args.buffer_min_size * 3))
        already_done = []
        with args.out_weights.with_suffix('.tmp').open('w') as f:
            for sample in replay_memory:
                f.write(json.dumps(sample.to_dict()))
                f.write("\n")
        with args.out_weights.with_suffix('.done').open('w'):
            pass

    q_estimator.save_weights(args.out_weights, args)
    if args.num_episodes == 0:
        args.out_weights.with_suffix('.tmp').unlink()
        args.out_weights.with_suffix('.done').unlink()
        return

    ctxt = tmp.get_context('spawn')
    jobs: Queue[Tuple[Job, Optional[Demonstration]]] = ctxt.Queue()
    done: Queue[Tuple[Job, Tuple[str, ReinforceGraph]]] = ctxt.Queue()
    samples: Queue[LabeledTransition] = ctxt.Queue()

    for sample in replay_memory:
        samples.put(sample)

    with tmp.Pool() as pool:
        jobs_in_files = list(tqdm(pool.imap(
            functools.partial(get_proofs, args),
            list(enumerate(args.environment_files))),
                                  total=len(args.environment_files),
                                  leave=False))
    unfiltered_jobs = [job for job_list in jobs_in_files for job in job_list
                       if job not in already_done]
    if args.proofs_file:
        with open(args.proofs_file, 'r') as f:
            proof_names = [line.strip() for line in f]
        all_jobs = [
            job for job in unfiltered_jobs if
            serapi_instance.lemma_name_from_statement(job[2]) in proof_names]
    elif args.proof:
        all_jobs = [
            job for job in unfiltered_jobs if
            serapi_instance.lemma_name_from_statement(job[2]) == args.proof] \
             * args.num_threads
    else:
        all_jobs = unfiltered_jobs

    all_jobs_and_dems: List[Tuple[Job, Optional[Demonstration]]]
    if args.demonstrate_from:
        all_jobs_and_dems = [(job, extract_solution(args,
                                                    args.demonstrate_from,
                                                    job))
                             for job in all_jobs]
    else:
        all_jobs_and_dems = [(job, None) for job in all_jobs]

    for job in all_jobs_and_dems:
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

        graphs_done: List[Tuple[str, ReinforceGraph]] = []
        with tqdm(total=len(all_jobs) + len(already_done),
                  dynamic_ncols=True) as bar:
            bar.update(len(already_done))
            bar.refresh()
            for _ in range(len(all_jobs)):
                done_job, graph_job = done.get()
                if graph_job:
                    graphs_done.append(graph_job)
                bar.update()
                with args.out_weights.with_suffix(".done").open('a') as f:
                    f.write(json.dumps((str(done_job[0]),
                                        done_job[1],
                                        done_job[2])))
        for worker in workers:
            worker.kill()
        training_worker.kill()

        for graphpath, graph in graphs_done:
            graph.assignApproximateQScores(args, predictor,
                                           q_estimator)
            graph.draw(graphpath)

        args.out_weights.with_suffix('.tmp').unlink()
        args.out_weights.with_suffix('.done').unlink()


def reinforce_worker(worker_idx: int,
                     args: argparse.Namespace,
                     lock: Lock,
                     namespace: multiprocessing.managers.Namespace,
                     samples: Queue[LabeledTransition],
                     jobs: Queue[Tuple[Job, Optional[Demonstration]]],
                     done: Queue[Tuple[Job,
                                       Optional[Tuple[str,
                                                      ReinforceGraph]]]]):

    sys.setrecursionlimit(100000)
    failing_lemma = ""

    try:
        (next_file, next_module, next_lemma), demonstration = jobs.get_nowait()
    except queue.Empty:
        return
    with util.silent():
        all_commands = serapi_instance.load_commands_preserve(
            args, worker_idx + 1, args.prelude / next_file)

    rest_commands = all_commands
    while rest_commands:
        with serapi_instance.SerapiContext(["sertop", "--implicit"],
                                           serapi_instance.
                                           get_module_from_filename(str(next_file)),
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
                        graph_job = \
                          reinforce_lemma_multithreaded(args, coq,
                                                        lock, namespace,
                                                        worker_idx,
                                                        samples,
                                                        next_lemma,
                                                        next_module,
                                                        demonstration)
                    except serapi_instance.CoqAnomaly:
                        if args.hardfail:
                            raise
                        if failing_lemma == lemma_statement:
                            eprint("Hit the same anomaly twice! Skipping",
                                   guard=args.verbose >= 1)
                            done.put(((next_file, next_module, next_lemma),
                                      None))

                            try:
                                (new_file, next_module, next_lemma), \
                                  demonstration = jobs.get_nowait()
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
                                f"FAILED in file {next_file}, "
                                f"lemma {next_lemma}")
                            eprint(e)
                        raise
                    serapi_instance.admit_proof(coq, lemma_statement)
                    while not serapi_instance.ending_proof(rest_commands[0]):
                        rest_commands = rest_commands[1:]
                    rest_commands = rest_commands[1:]
                    done.put(((next_file, next_module, next_lemma),
                              graph_job))
                    try:
                        (new_file, next_module, next_lemma), demonstration = \
                          jobs.get_nowait()
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
                                f"{next_file}: Failed to admit proof "
                                f"{next_lemma_name}")
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
        _module_prefix: str,
        demonstration: Optional[Demonstration]) -> Tuple[str, ReinforceGraph]:

    lemma_name = serapi_instance.lemma_name_from_statement(lemma_statement)
    graph = ReinforceGraph(lemma_name)
    lemma_memory = []
    for i in trange(args.num_episodes, disable=(not args.progress),
                    leave=False, position=worker_idx + 1):
        cur_node = graph.start_node
        proof_contexts_seen = [unwrap(coq.proof_context)]
        episode_memory: List[LabeledTransition] = []
        reached_qed = False
        for t in range(args.episode_length):
            context_before = coq.tactic_context(coq.local_lemmas[:-1])
            proof_context_before = unwrap(coq.proof_context)
            context_trunced = truncate_tactic_context(
                context_before, args.max_term_length)
            if (demonstration and
                t < len(demonstration) -
                    ((i//args.demonstration_steps)+1)):
                eprint("Getting demonstration", guard=args.verbose >= 2)
                ordered_actions = [(demonstration[t],
                                    certainty_of(namespace.predictor,
                                                 args.num_predictions * 2,
                                                 context_trunced,
                                                 demonstration[t]))]
            else:
                with print_time("Getting predictions", guard=args.verbose >= 2):
                    with lock:
                        eprint(f"Locked in thread {worker_idx}",
                               guard=args.verbose >= 2)
                        predictor = namespace.predictor
                        estimator = namespace.estimator
                        with print_time("Making predictions",
                                        guard=args.verbose >= 3):
                            predictions = predictor.predictKTactics(
                                context_trunced, args.num_predictions)
                        if random.random() < args.exploration_factor:
                            eprint("Picking random action",
                                   guard=args.verbose >= 2)
                            ordered_actions = order_by_score(predictions)
                        else:
                            with print_time("Picking actions with q_estimator",
                                            guard=args.verbose >= 2):
                                q_choices = zip(estimator(
                                    [(context_trunced,
                                      p.prediction, p.certainty)
                                     for p in predictions]),
                                                predictions)
                                ordered_actions = [p[1] for p in
                                                   sorted(q_choices,
                                                          key=lambda q: q[0],
                                                          reverse=True)]
                    eprint(f"Unlocked in thread {worker_idx}",
                           guard=args.verbose >= 2)
            with print_time("Running actions", guard=args.verbose >= 2):
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
                            if args.ghosts:
                                transition = assign_failed_reward(
                                    context_trunced.relevant_lemmas,
                                    context_trunced.prev_tactics,
                                    proof_context_before,
                                    proof_context_after,
                                    try_action,
                                    try_original_certainty,
                                    0)
                                ghost_node = graph.addGhostTransition(
                                    cur_node, transition)
                                transition.graph_node = ghost_node
                            continue
                        action = try_action
                        original_certainty = try_original_certainty
                        break
                    except (serapi_instance.ParseError,
                            serapi_instance.CoqExn,
                            serapi_instance.TimeoutError):
                        if args.ghosts:
                            transition = assign_failed_reward(
                                context_trunced.relevant_lemmas,
                                context_trunced.prev_tactics,
                                proof_context_before,
                                proof_context_before,
                                try_action,
                                try_original_certainty,
                                0)
                            ghost_node = graph.addGhostTransition(cur_node,
                                                                  transition)
                            transition.graph_node = ghost_node
                if action is None:
                    # We'll hit this case of we tried all of the
                    # predictions, and none worked
                    graph.setNodeColor(cur_node, "red")
                    transition = assign_failed_reward(
                        context_trunced.relevant_lemmas,
                        context_trunced.prev_tactics,
                        proof_context_before,
                        proof_context_before,
                        "Abort.",
                        try_original_certainty,
                        -25)
                    samples.put(transition)
                    episode_memory.append(transition)
                    break  # Break from episode
            transition = assign_reward(args,
                                       context_trunced.relevant_lemmas,
                                       context_trunced.prev_tactics,
                                       proof_context_before,
                                       proof_context_after,
                                       action,
                                       unwrap(original_certainty))
            cur_node = graph.addTransition(cur_node, transition)
            transition.graph_node = cur_node
            assert transition.reward < 2000
            samples.put(transition)
            episode_memory.append(transition)
            proof_contexts_seen.append(proof_context_after)

            lemma_memory += episode_memory
            if coq.goals == "":
                eprint("QED!", guard=args.verbose >= 2)
                graph.mkQED(cur_node)
                for sample in (episode_memory *
                               (args.success_repetitions - 1)):
                    samples.put(sample)
                reached_qed = True
                break

        if not reached_qed:
            # We'll hit this case of we tried all of the
            # predictions, and none worked
            graph.setNodeColor(cur_node, "red")
            transition = assign_failed_reward(
                context_trunced.relevant_lemmas,
                context_trunced.prev_tactics,
                proof_context_before,
                proof_context_before,
                "Abort.",
                try_original_certainty,
                -25)
            samples.put(transition)
            episode_memory.append(transition)

        # Clean up episode
        if lemma_name:
            coq.run_stmt("Admitted.")
            coq.run_stmt(f"Reset {lemma_name}.")
        else:
            coq.cancel_last()
            while coq.goals:
                coq.cancel_last()

        coq.run_stmt(lemma_statement)

        # Write out lemma memory to progress file for resuming
        for sample in lemma_memory:
            with args.out_weights.with_suffix('.tmp').open('a') as f:
                f.write(json.dumps(sample.to_dict()) + "\n")
    graphpath = (args.graphs_dir / lemma_name).with_suffix(".png")
    return str(graphpath), graph


def reinforce_training_worker(args: argparse.Namespace,
                              initial_buffer_size: int,
                              lock: Lock,
                              namespace: multiprocessing.managers.Namespace,
                              samples: Queue[LabeledTransition]):
    last_trained_at = 0
    samples_retrieved = 0
    memory: List[LabeledTransition] = []
    while True:
        if samples_retrieved - last_trained_at < args.train_every_min:
            next_sample = samples.get()
            memory.append(next_sample)
            samples_retrieved += 1
            continue
        else:
            try:
                next_sample = samples.get(timeout=.01)
                memory.append(next_sample)
                samples_retrieved += 1
                if samples_retrieved - last_trained_at > args.train_every_max:
                    eprint("Forcing training", guard=args.verbose >= 2)
                else:
                    continue
            except queue.Empty:
                pass
        if len(memory) > args.buffer_max_size:
            memory = random.sample(memory, args.buffer_max_size -
                                   args.train_every_max)
            # del memory[0:args.train_every_max+1]
        if samples_retrieved - last_trained_at >= args.train_every_min:
            last_trained_at = samples_retrieved
            transition_samples = sample_batch(memory, args.batch_size)
            with lock:
                eprint(
                    f"Locked in training thread for {len(memory)} samples",
                    guard=args.verbose >= 2)
                q_estimator = namespace.estimator
                predictor = namespace.predictor
                with print_time("Assigning scores", guard=args.verbose >= 2):
                    training_samples = assign_scores(args,
                                                     transition_samples,
                                                     q_estimator,
                                                     predictor)
                with print_time("Training", guard=args.verbose >= 2):
                    q_estimator.train(training_samples,
                                      show_loss=args.show_loss)
                q_estimator.save_weights(args.out_weights, args)
                namespace.estimator = q_estimator
                eprint("Unlocked in training thread",
                       guard=args.verbose >= 2)

    pass


def get_proofs(args: argparse.Namespace,
               t: Tuple[int, str]) -> List[Tuple[str, str, str]]:
    idx, filename = t
    with util.silent():
        cmds = serapi_instance.load_commands_preserve(
            args, idx, args.prelude / filename)
    return [(filename, module, cmd) for module, cmd in
            serapi_instance.lemmas_in_file(
                filename, cmds, args.include_proof_relevant)]


def extract_solution(args: argparse.Namespace,
                     report_dir: Path2, job: Job) -> Optional[Demonstration]:
    job_file, job_module, job_lemma = job
    proofs_filename = report_dir / (safe_abbrev(Path2(job_file),
                                                args.environment_files)
                                    + "-proofs.txt")
    try:
        with proofs_filename.open('r') as proofs_file:
            for line in proofs_file:
                entry, sol = json.loads(line)
                if (entry[0] == str(job_file) and
                        entry[1] == job_module and
                        entry[2] == job_lemma):
                    return [cmd["tactic"] for cmd in sol["commands"]
                            if cmd["tactic"] != "Proof."]
            else:
                eprint(f"Couldn't find solution for lemma {job_lemma} "
                       f"in module {job_module} in proofs file")
                raise FileNotFoundError()
    except FileNotFoundError:
        eprint(f"Couldn't find proofs file "
               f"in search directory for file {job_file}")
        raise
    pass


def sample_batch(transitions: List[LabeledTransition], k: int) -> \
      List[LabeledTransition]:
    return random.sample(transitions, k)


def assign_failed_reward(relevant_lemmas: List[str], prev_tactics: List[str],
                         before: ProofContext, after: ProofContext,
                         tactic: str, certainty: float, reward: int) \
                         -> LabeledTransition:
    return LabeledTransition(relevant_lemmas, prev_tactics, before, after,
                             tactic, certainty, reward, None)


def assign_reward(args: argparse.Namespace,
                  relevant_lemmas: List[str], prev_tactics: List[str],
                  before: ProofContext, after: ProofContext, tactic: str,
                  certainty: float) \
      -> LabeledTransition:
    # goals_changed = len(after.all_goals) - len(before.all_goals)
    if len(after.all_goals) == 0:
        reward = 50.0
    # elif goals_changed != 0:
    #     if goals_changed > 0:
    #         reward = -(goals_changed * 2.0) * \
    #           (args.time_discount ** goals_changed)
    #     else:
    #         reward = -(goals_changed * 2.0)
    else:
        reward = 0
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
                context = TacticContext(
                    transition.relevant_lemmas,
                    transition.prev_tactics,
                    transition.before.fg_goals[0].hypotheses,
                    transition.before.fg_goals[0].goal)
            yield assign_reward(args,
                                transition.relevant_lemmas,
                                transition.prev_tactics,
                                context_r2py(transition.before),
                                context_r2py(transition.after),
                                transition.tactic,
                                certainty_of(predictor,
                                             args.num_predictions * 2,
                                             context,
                                             transition.tactic))
    return list(generate())


def assign_scores(args: argparse.Namespace,
                  transitions: List[LabeledTransition],
                  q_estimator: QEstimator,
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
            tactic_ctxt = truncate_tactic_context(transition.after_context,
                                                  args.max_term_length)
            before_ctxt = truncate_tactic_context(transition.before_context,
                                                  args.max_term_length)

            if len(transition.after.all_goals) == 0:
                new_q = transition.reward
            else:
                estimates = q_estimator(
                    [(tactic_ctxt, prediction.prediction, prediction.certainty)
                     for prediction in predictions])
                estimated_future_q = \
                    args.time_discount * max(estimates)
                new_q = transition.reward + estimated_future_q

            yield TacticContext(
                transition.relevant_lemmas,
                transition.prev_tactics,
                before_ctxt.hypotheses,
                before_ctxt.goal), \
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

            random_certainty = random.random()
            yield (context, transition.tactic,
                   random_certainty, random_certainty)

    samples = list(gen_samples())
    estimator.train(samples, args.batch_size,
                    args.pretrain_epochs,
                    show_loss=args.show_loss)


T = TypeVar('T')


def index_sample_probabilities(probs: List[float]) -> int:
    interval_val = random.random()
    for idx, prob in enumerate(probs):
        if interval_val < prob:
            return idx
        else:
            interval_val -= prob
    assert False, "Probs don't add up to one!"


def sample_by_score(options: List[Tuple[T, float]],
                    interval_val: float) \
  -> Tuple[T, float]:
    vals, scores = cast(Tuple[List[T], List[float]], zip(*options))
    e_xs = [math.exp(x) for x in scores]
    e_sum = sum(e_xs)
    softmax_probs = [e_x / e_sum for e_x in e_xs]
    selected_idx = index_sample_probabilities(softmax_probs)
    return options[selected_idx]


def order_by_score(items: List[Tuple[T, float]]) \
  -> List[Tuple[T, float]]:
    items_left = list(items)
    result = []
    for i in range(len(items)):
        next_item = sample_by_score(items_left, random.random())
        result.append(next_item)
        items_left.remove(next_item)
    return result


def certainty_of(predictor: tactic_predictor.TacticPredictor, k: int,
                 context: TacticContext, tactic: str) -> float:
    predictor = cast(features_polyarg_predictor.
                     FeaturesPolyargPredictor,
                     predictor)
    return predictor.predictionCertainty(context, tactic)


if __name__ == "__main__":
    main()
