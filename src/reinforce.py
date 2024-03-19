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
import json.decoder
import multiprocessing
import math
import sys
import functools
from queue import Queue
import queue
from typing import (List, Tuple, Iterator, Optional,
                    cast, TYPE_CHECKING,
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
from util import eprint, print_time, unwrap, progn, safe_abbrev

from rgraph import (LabeledTransition,
                    ReinforceGraph, assignApproximateQScores)


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
    parser.add_argument("--num-predictions", default=32, type=int)
    parser.add_argument("--gpu", default=0, type=int)

    parser.add_argument("--buffer-min-size", default=256, type=int)
    parser.add_argument("--buffer-max-size", default=32768, type=int)
    parser.add_argument("--batch-size", default=32, type=int)

    parser.add_argument("--num-episodes", default=32, type=int)
    parser.add_argument("--episode-length", default=16, type=int)

    parser.add_argument("--learning-rate", default=0.02, type=float)
    parser.add_argument("--batch-step", default=4, type=int)
    parser.add_argument("--gamma", default=0.5, type=float)
    parser.add_argument("--exploration-factor", default=0.4, type=float)
    parser.add_argument("--exploration-smoothing-factor", default=2,
                        type=float)
    parser.add_argument("--time-discount", default=0.9, type=float)

    parser.add_argument("--max-term-length", default=512, type=int)

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
    parser.add_argument("--epochs-per-batch", default=32, type=int)
    parser.add_argument("--show-loss", action='store_true')

    args = parser.parse_args()

    if util.use_cuda:
        torch.cuda.set_device(args.gpu)
        util.cuda_device = f"cuda:{args.gpu}"

    try:
        os.makedirs(str(args.graphs_dir))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    reinforce_multithreaded(args)


Job = Tuple[Path2, str, str]
Demonstration = List[str]


def reinforce_multithreaded(args: argparse.Namespace) -> None:

    def resume(resume_file: Path2,
               jobs_in_files: List[Job],
               weights: Path2,
               q_estimator: QEstimator) -> \
      Tuple[List[LabeledTransition],
            List[Job],
            List[Tuple[str, ReinforceGraph]]]:
        eprint("Looks like there was a session in progress for these weights! "
               "Resuming")
        q_estimator_name, *saved = \
            torch.load(str(weights))
        q_estimator.load_saved_state(*saved)
        already_done = []
        graphs_done = []
        with weights.with_suffix('.done').open('r') as f:
            for (idx, line) in enumerate(f):
                try:
                    next_done = json.loads(line)
                except json.decoder.JSONDecodeError:
                    print(f"Loading line {idx} failed")
                    print(line)
                    raise
                already_done.append((Path2(next_done[0]), next_done[1],
                                     next_done[2]))
                graphpath = (args.graphs_dir /
                             serapi_instance.lemma_name_from_statement(
                               next_done[2]))\
                    .with_suffix(".svg")
                graph = ReinforceGraph.load(
                    graphpath.with_suffix(".svg.json"))
                graphs_done.append((graphpath, graph))
        jobs_todo = [job for job in jobs_in_files
                     if job not in already_done]

        replay_memory = []
        if len(jobs_todo) > 0:
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
        else:
            eprint("Warning: no jobs left to do")
        return replay_memory, jobs_todo, already_done, graphs_done

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

    ctxt = tmp.get_context('spawn')
    with ctxt.Pool() as pool:
        jobs_in_files = [
            job
            for job_list in
            list(tqdm(pool.imap(
                 functools.partial(get_proofs, args),
                 list(enumerate(args.environment_files))),
                                       total=len(args.environment_files),
                                       leave=False,
                                       desc="Finding proofs"))
            for job in job_list]
    if args.proofs_file:
        with open(args.proofs_file, 'r') as f:
            proof_names = [line.strip() for line in f]
        all_jobs = [
            job for job in jobs_in_files if
            serapi_instance.lemma_name_from_statement(job[2]) in proof_names]
    elif args.proof:
        all_jobs = [
            job for job in jobs_in_files if
            serapi_instance.lemma_name_from_statement(job[2]) == args.proof] \
             * args.num_threads
    else:
        all_jobs = jobs_in_files

    resume_file = args.out_weights.with_suffix('.tmp')
    if resume_file.exists():
        replay_memory, jobs_todo, already_done, graphs_done = \
            resume(resume_file,
                   all_jobs,
                   args.out_weights,
                   q_estimator)
    else:
        jobs_todo = all_jobs
        graphs_done = []
        # Load the scraped (demonstrated) samples and the proof
        # environment commands. Assigns them an estimated "original
        # predictor certainty" value for use as a feature.
        replay_memory = []
        if args.buffer_min_size > 0:
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

    jobs: Queue[Tuple[Job, Optional[Demonstration]]] = ctxt.Queue()
    done: Queue[Tuple[Job, Tuple[str, ReinforceGraph]]] = ctxt.Queue()
    samples: Queue[LabeledTransition] = ctxt.Queue()

    for sample in replay_memory:
        samples.put(sample)

    all_jobs_and_dems: List[Tuple[Job, Optional[Demonstration]]]
    if args.demonstrate_from:
        all_jobs_and_dems = [(job, extract_solution(args,
                                                    args.demonstrate_from,
                                                    job))
                             for job in jobs_todo]
    else:
        all_jobs_and_dems = [(job, None) for job in jobs_todo]

    if len(all_jobs_and_dems) > 0:
        for job in all_jobs_and_dems:
            jobs.put(job)

        q_estimator.share_memory()

        training_worker = ctxt.Process(
            target=reinforce_training_worker,
            args=(args, len(replay_memory), q_estimator, predictor, samples))
        workers = [ctxt.Process(
            target=reinforce_worker,
            args=(widx,
                  args,
                  predictor,
                  q_estimator,
                  samples,
                  jobs,
                  done))
                   for widx in range(min(args.num_threads, len(all_jobs)))]
        training_worker.start()
        for worker in workers:
            worker.start()

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
                    f.write("\n")
        for worker in workers:
            worker.kill()
        training_worker.kill()

    for graphpath, graph in tqdm(graphs_done, desc="Drawing graphs"):
        assignApproximateQScores(graph, args.max_term_length, predictor,
                                 q_estimator)
        graph.draw(graphpath)

    args.out_weights.with_suffix('.tmp').unlink()
    args.out_weights.with_suffix('.done').unlink()


def reinforce_worker(worker_idx: int,
                     args: argparse.Namespace,
                     estimator: QEstimator,
                     predictor: TacticPredictor,
                     samples: Queue[LabeledTransition],
                     jobs: Queue[Tuple[Job, Optional[Demonstration]]],
                     done: Queue[Tuple[Job,
                                       Optional[Tuple[str,
                                                      ReinforceGraph]]]]):

    if util.use_cuda:
        torch.cuda.set_device(args.gpu)
        util.cuda_device = f"cuda:{args.gpu}"
    sys.setrecursionlimit(100000)
    failing_lemma = ""

    try:
        (next_file, next_module, next_lemma), demonstration = jobs.get_nowait()
    except queue.Empty:
        return
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
                                                        estimator, predictor,
                                                        worker_idx,
                                                        samples,
                                                        next_lemma,
                                                        next_module,
                                                        demonstration)
                        graphpath, graph = graph_job
                        graph.save(graphpath + ".json")
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
                    while not serapi_instance.ending_proof(rest_commands[0]):
                        rest_commands.pop(0)
                    ending_comamnd = rest_commands.pop(0)
                    serapi_instance.admit_proof(coq, lemma_statement, ending_command)
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
                    ending_command = None
                    for cmd in rest_commands:
                        if serapi_instance.ending_proof(cmd):
                            ending_command = cmd
                            break
                    proof_relevant = ending_command.strip() == "Defined." or \
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
    del estimator


def reinforce_lemma_multithreaded(
        args: argparse.Namespace,
        coq: serapi_instance.SerapiInstance,
        predictor: TacticPredictor,
        estimator: QEstimator,
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
                                    certainty_of(predictor,
                                                 args.num_predictions * 2,
                                                 context_trunced,
                                                 demonstration[t]))]
            else:
                with print_time("Getting predictions", guard=args.verbose >= 2):
                    # eprint(f"Locked in thread {worker_idx}",
                    #        guard=args.verbose >= 2)
                    with print_time("Making predictions",
                                    guard=args.verbose >= 3):
                        predictions = predictor.predictKTactics(
                            context_trunced, args.num_predictions)
                    if random.random() < args.exploration_factor:
                        eprint("Picking random action",
                               guard=args.verbose >= 2)
                        ordered_actions = order_by_score(
                            [(prediction,
                             score * (1/args.exploration_smoothing_factor))
                             for prediction, score in predictions])
                    else:
                        with print_time("Picking actions with q_estimator",
                                        guard=args.verbose >= 2):
                            q_choices = zip(estimator(
                                [(context_trunced,
                                  p.prediction, p.certainty)
                                 for p in predictions],
                                 progress=args.verbose >= 2),
                                            predictions)
                            ordered_actions = [p[1] for p in
                                               sorted(q_choices,
                                                      key=lambda q: q[0],
                                                      reverse=True)]
                    # eprint(f"Unlocked in thread {worker_idx}",
                    #        guard=args.verbose >= 2)
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
                if any([len(obligation.goal) > 5120 for
                        obligation in proof_context_after.all_goals]):
                    break
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
            if len(unwrap(coq.proof_context).all_goals) == 0:
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

        # # Write out lemma memory to progress file for resuming
        # for sample in lemma_memory:
        #     with args.out_weights.with_suffix('.tmp').open('a') as f:
        #         f.write(json.dumps(sample.to_dict()) + "\n")
    graphpath = (args.graphs_dir / lemma_name).with_suffix(".svg")
    return str(graphpath), graph


def reinforce_training_worker(args: argparse.Namespace,
                              initial_buffer_size: int,
                              q_estimator: QEstimator,
                              predictor: TacticPredictor,
                              samples: Queue[LabeledTransition]):
    if util.use_cuda:
        torch.cuda.set_device(args.gpu)
        util.cuda_device = f"cuda:{args.gpu}"
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
            with print_time("Assigning scores", guard=args.verbose >= 2):
                training_samples = normalize_batch_size(
                    assign_scores(args,
                                  q_estimator,
                                  predictor,
                                  transition_samples,
                                  progress=args.verbose >= 2),
                    args.batch_size)
            with print_time("Training", guard=args.verbose >= 2):
                q_estimator.train(training_samples,
                                  show_loss=args.show_loss,
                                  num_epochs=args.epochs_per_batch)
            q_estimator.save_weights(args.out_weights, args)
            with args.out_weights.with_suffix('.tmp').open('w') as f:
                for sample in memory:
                    f.write(json.dumps(sample.to_dict()))
                    f.write("\n")

    pass


def get_proofs(args: argparse.Namespace,
               t: Tuple[int, str]) -> List[Tuple[str, str, str]]:
    idx, filename = t
    with util.silent():
        cmds = serapi_instance.load_commands_preserve(
            args, idx, args.prelude / filename)
    return [(filename, module, cmd) for module, cmd in
            serapi_instance.lemmas_in_file(
                filename, cmds, args.include_proof_relevant,
                disambiguated_goal_stmts)]


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


def normalize_batch_size(samples: List[Tuple[TacticContext, str,
                                             float, float]],
                         k: int) -> \
      List[Tuple[TacticContext, str, float, float]]:
    assert k >= len(samples), \
      "Pre-normalized batch must not exceed target size"
    result = samples * (k // len(samples)) + \
        random.sample(samples, k % len(samples))
    random.shuffle(result)
    return result


def sample_batch(transitions: List[LabeledTransition], k: int) -> \
      List[LabeledTransition]:
    if k >= len(transitions):
        return transitions
    else:
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


# The "progress" parameter is currently only used in another module
def assign_scores(args: argparse.Namespace,
                  q_estimator: QEstimator,
                  predictor: tactic_predictor.TacticPredictor,
                  transitions: List[LabeledTransition],
                  progress: bool = False) -> \
                  List[Tuple[TacticContext, str, float, float]]:
    easy_transitions = [transition for transition in transitions
                        if len(transition.after.all_goals) == 0 or
                        transition.action == "Abort."]
    easy_qs = [(50 if len(transition.after.all_goals) == 0 else -25)
               for transition in easy_transitions]
    hard_transitions = [transition for transition in transitions
                        if len(transition.after.all_goals) != 0 and
                        transition.action != "Abort."]
    contexts_trunced = [truncate_tactic_context(
        transition.after_context,
        args.max_term_length)
                        for transition in hard_transitions]

    prediction_lists = cast(features_polyarg_predictor
                            .FeaturesPolyargPredictor,
                            predictor) \
        .predictKTactics_batch(
            contexts_trunced,
            args.num_predictions,
            args.verbose)
    queries = [(truncate_tactic_context(transition.after_context,
                                        args.max_term_length),
                prediction.prediction, prediction.certainty)
               for transition, predictions in zip(hard_transitions,
                                                  prediction_lists)
               for prediction in predictions]
    estimate_lists_flattened = q_estimator(queries, progress=progress)
    estimate_lists = [estimate_lists_flattened
                      [i:i+args.num_predictions]
                      for i in range(0, len(estimate_lists_flattened),
                                     args.num_predictions)]
    hard_qs = []
    for transition, estimates in zip(transitions, estimate_lists):

        estimated_future_q = args.time_discount * max(estimates)
        new_q = transition.reward + estimated_future_q
        hard_qs.append(new_q)

    results = []
    for transition, new_q in zip(easy_transitions + hard_transitions,
                                 easy_qs + hard_qs):
        before_ctxt = truncate_tactic_context(
            transition.before_context, args.max_term_length)
        results.append((TacticContext(
             transition.relevant_lemmas,
             transition.prev_tactics,
             before_ctxt.hypotheses,
             before_ctxt.goal),
             transition.action, transition.original_certainty, new_q))

    return results


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
                   random_certainty, random_certainty * 50)

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
    tmp.set_start_method('spawn')
    main()
