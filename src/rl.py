#!/usr/bin/env python

import argparse
import json
import random
import re
import os
import time
import contextlib
import math
import pickle
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import (List, Optional, Dict, Tuple, Union, Any, Set,
                    Sequence, TypeVar, Callable, OrderedDict, Iterable,
                    Iterator)
from typing_extensions import Self
import warnings

from util import unwrap, eprint, print_time, nostderr

with print_time("Importing torch"):
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch import optim
    import torch.optim.lr_scheduler as scheduler
    import torch.cuda

#pylint: disable=wrong-import-order
from tqdm import tqdm

import coq_serapy
from coq_serapy.contexts import (FullContext, truncate_tactic_context,
                                 Obligation, TacticContext, ProofContext)
import coq2vec
#pylint: enable=wrong-import-order

from gen_rl_tasks import RLTask

with print_time("Importing search code"):
    from search_file import get_all_jobs
    from search_worker import ReportJob, Worker, get_predictor
    from search_strategies import completed_proof

    from models.tactic_predictor import (TacticPredictor, Prediction)
    from models.features_polyarg_predictor import FeaturesPolyargPredictor

optimizers = {
  "RMSprop": optim.RMSprop,
  "SGD": optim.SGD,
  "Adam": optim.Adam,
}

def main():
    eprint("Starting main")
    parser = argparse.ArgumentParser(
        description="Train a state estimator using reinforcement learning"
        "to complete proofs using Proverbot9001.")
    add_args_to_parser(parser)
    args = parser.parse_args()

    if args.filenames[0].suffix == ".json":
        args.splits_file = args.filenames[0]
        args.filenames = None
    else:
        args.splits_file = None

    reinforce_jobs(args)

def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--prelude", default=".", type=Path)
    parser.add_argument("--output", "-o", dest="output_file",
                        help="output data folder name",
                        default="data/rl_weights.dat",
                        type=Path)
    parser.add_argument("--verbose", "-v", help="verbose output",
                        action="count", default=0)
    parser.add_argument("--progress", "-P", help="show progress of files",
                        action='store_true')
    parser.add_argument("--print-timings", "-t", action='store_true')
    parser.add_argument("--no-set-switch", dest="set_switch", action='store_false')
    parser.add_argument("--include-proof-relevant", action="store_true")
    parser.add_argument("--backend", choices=['serapi', 'lsp', 'auto'], default='auto')
    parser.add_argument('filenames', help="proof file name (*.v)",
                        nargs='+', type=Path)
    proofsGroup = parser.add_mutually_exclusive_group()
    proofsGroup.add_argument("--proof", default=None)
    proofsGroup.add_argument("--proofs-file", default=None)
    parser.add_argument("--tasks-file", default=None)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--no-interleave", dest="interleave", action="store_false")
    parser.add_argument('--supervised-weights', type=Path, dest="weightsfile")
    parser.add_argument("--coq2vec-weights", type=Path)
    parser.add_argument("--max-sertop-workers", default=16, type=int)
    parser.add_argument("-l", "--learning-rate", default=2.5e-4, type=float)
    parser.add_argument("-g", "--gamma", default=0.9, type=float)
    parser.add_argument("--starting-epsilon", default=0, type=float)
    parser.add_argument("--ending-epsilon", default=1.0, type=float)
    parser.add_argument("--smoothing-factor", default=4.0, type=float)
    parser.add_argument("-s", "--steps-per-episode", default=16, type=int)
    parser.add_argument("-n", "--num-episodes", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-w", "--window-size", default=2560)
    parser.add_argument("-p", "--num-predictions", default=5, type=int)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--tactic-embedding-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--learning-rate-step", default=None, type=int)
    parser.add_argument("--learning-rate-decay", default=0.8, type=float)
    parser.add_argument("--batches-per-proof", default=1, type=int)
    parser.add_argument("--train-every", default=1, type=int)
    parser.add_argument("--optimizer", choices=optimizers.keys(),
                        default=list(optimizers.keys())[0])
    parser.add_argument("--start-from", type=Path, default=None)
    parser.add_argument("--print-loss-every", default=None, type=int)
    parser.add_argument("--sync-target-every",
                        help="Sync target network to v network every <n> episodes",
                        default=10, type=int)
    parser.add_argument("--allow-partial-batches", action='store_true')
    parser.add_argument("--blacklist-tactic", action="append",
                        dest="blacklisted_tactics", default=[])
    parser.add_argument("--resume", choices=["no", "yes", "ask"], default="ask")
    parser.add_argument("--save-every", type=int, default=20)
    evalGroup = parser.add_mutually_exclusive_group()
    evalGroup.add_argument("--evaluate", action="store_true")
    evalGroup.add_argument("--evaluate-baseline", action="store_true")
    evalGroup.add_argument("--evaluate-random-baseline", action="store_true")
    parser.add_argument("--curriculum",action="store_true")
    parser.add_argument("--verifyvval",action="store_true")
    parser.add_argument("--verifyv-every", type=int, default=None)
    parser.add_argument("--verifyv-steps", action="store_true")
    return parser


class FileReinforcementWorker(Worker):
    def finish_proof(self) -> None:
        assert self.coq
        while not coq_serapy.ending_proof(self.remaining_commands[0]):
            self.remaining_commands.pop(0)
        ending_cmd = self.remaining_commands.pop(0)
        coq_serapy.admit_proof(self.coq, self.coq.prev_tactics[0], ending_cmd)

    def run_into_task(self, job: ReportJob, tactic_prefix: List[str],
                      restart_anomaly: bool = True, careful: bool = False) -> None:
        try:
            assert self.coq
            if job.project_dir != self.cur_project or job.filename != self.cur_file \
               or job.module_prefix != self.coq.sm_prefix \
                   or self.coq.proof_context is None \
                   or job.lemma_statement.strip() != self.coq.prev_tactics[0].strip():
                if self.coq.proof_context is not None:
                    self.finish_proof()
                with print_time("Running into job", guard=self.args.print_timings):
                    self.run_into_job(job, restart_anomaly, careful)
                with print_time("Running task prefix", guard=self.args.print_timings):
                    for statement in tactic_prefix:
                        self.coq.run_stmt(statement)
                    self.coq.update_state()
            else:
                with print_time("Traversing to tactic prefix", self.args.print_timings):
                    cur_path = self.coq.tactic_history.getFullHistory()[1:]
                    # this is needed because commands that are just comments don't
                    # show up in the history (because cancelling skips them).
                    target_path = [tac for tac in tactic_prefix
                                   if coq_serapy.kill_comments(tac).strip() != ""]
                    common_prefix_len = 0
                    for item1, item2 in zip(cur_path, target_path):
                        if item1.strip() != item2.strip():
                            break
                        common_prefix_len += 1
                    # Heuristically assume that cancelling a command is about
                    # 1/20th as expensive as running one.
                    if len(target_path) < \
                            (len(cur_path) - common_prefix_len) * 0.05 + \
                            (len(target_path) - common_prefix_len):
                        eprint(f"Decided to abort because the target path is only {len(target_path)} tactics long, "
                               f"but the common prefix is only {common_prefix_len} tactics long, "
                               f"so we would have to cancel {len(cur_path) - common_prefix_len} tactics "
                               f"and rerun {len(target_path) - common_prefix_len} tactics.",
                               guard=self.args.verbose >= 1)
                        self.coq.run_stmt("Abort.")
                        self.coq.run_stmt(job.lemma_statement)
                        for cmd in target_path:
                            self.coq.run_stmt_noupdate(cmd)
                    else:
                        for _ in range(len(cur_path) - common_prefix_len):
                            self.coq.cancel_last_noupdate()
                        for cmd in target_path[common_prefix_len:]:
                            self.coq.run_stmt_noupdate(cmd)
                    self.coq.update_state()
        except coq_serapy.CoqAnomaly:
            if restart_anomaly:
                self.restart_coq()
                self.reset_file_state()
                self.enter_file(job.filename)
                eprint("Hit a coq anomaly! Restarting...",
                       guard=self.args.verbose >= 1)
                self.run_into_task(job, tactic_prefix, False, careful)
                return
            raise

class ReinforcementWorker:
    v_network: 'VNetwork'
    target_v_network: 'VNetwork'
    replay_buffer: 'ReplayBuffer'
    verbosity: int
    predictor: TacticPredictor
    last_worker_idx: int
    file_workers: Dict[str, Tuple[FileReinforcementWorker, int]]
    def __init__(self, args: argparse.Namespace,
                 predictor: TacticPredictor,
                 v_network: 'VNetwork',
                 target_network: 'VNetwork',
                 switch_dict: Optional[Dict[str, str]] = None,
                 initial_replay_buffer: Optional['ReplayBuffer'] = None) -> None:
        self.args = args
        self.v_network = v_network
        self.target_v_network = target_network
        self.predictor = predictor
        self.last_worker_idx = 0
        if initial_replay_buffer:
            self.replay_buffer = initial_replay_buffer
        else:
            self.replay_buffer = ReplayBuffer(args.window_size,
                                              args.allow_partial_batches)
        self.verbosity = args.verbose
        self.file_workers = {}
    def _get_worker(self, filename: str) -> FileReinforcementWorker:
        if filename not in self.file_workers:
            args_copy = argparse.Namespace(**vars(self.args))
            args_copy.verbose = self.args.verbose - 2
            worker = FileReinforcementWorker(args_copy, None)
            worker.enter_instance()
            self.file_workers[filename] = (worker, self.last_worker_idx)
            self.last_worker_idx += 1
        if len(self.file_workers) > self.args.max_sertop_workers:
            removing_worker_filename = None
            target_worker_idx = self.last_worker_idx - self.args.max_sertop_workers - 1
            for w_filename, (worker, idx) in self.file_workers.items():
                if idx == target_worker_idx:
                    removing_worker_filename = w_filename
                    break
            assert removing_worker_filename is not None
            worker_coq_instance = self.file_workers[removing_worker_filename][0].coq
            assert worker_coq_instance is not None
            worker_coq_instance.kill()
            del self.file_workers[removing_worker_filename]

        return self.file_workers[filename][0]

    def train(self, step: int) -> None:
        for batch_idx in range(self.args.batches_per_proof):
            train_v_network(self.args, self.v_network, self.target_v_network,
                            self.replay_buffer)
            if (step * self.args.batches_per_proof + batch_idx) % \
               self.args.sync_target_every == 0:
                self.sync_networks()

    def run_job_reinforce(self, job: ReportJob, tactic_prefix: List[str],
                          epsilon: float, restart: bool = True) -> Optional[int]:
        if not tactic_prefix_is_usable(tactic_prefix):
            if self.args.verbose >= 2:
                eprint(f"Skipping job {job} with prefix {tactic_prefix} "
                       "because it can't purely focused")
            else:
                eprint("Skipping a job because it can't be purely focused")
            return None
        with print_time("Getting worker", guard=self.args.print_timings):
            file_worker = self._get_worker(job.filename)
        assert file_worker.coq is not None
        try:
            with print_time("Running into task", guard=self.args.print_timings):
                file_worker.run_into_task(job, tactic_prefix)
            with print_time("Experiencing", guard=self.args.print_timings):
                return experience_proof(self.args,
                                        file_worker.coq,
                                        self.predictor, self.v_network,
                                        self.replay_buffer,
                                        epsilon)
        except coq_serapy.CoqAnomaly:
            eprint("Encountered Coq anomaly.")
            file_worker.restart_coq()
            file_worker.reset_file_state()
            file_worker.enter_file(job.filename)
            if restart:
                return self.run_job_reinforce(job, tactic_prefix, epsilon, restart=False)
            eprint("Encountered anomaly without restart, closing current job")
        return None
    def evaluate_job(self, job: ReportJob, tactic_prefix: List[str], restart: bool = True) \
            -> Optional[bool]:
        if not tactic_prefix_is_usable(tactic_prefix):
            if self.args.verbose >= 2:
                eprint(f"Skipping job {job} with prefix {tactic_prefix} "
                       "because it can't purely focused")
            else:
                eprint("Skipping a job because it can't be purely focused")
            return None
        file_worker = self._get_worker(job.filename)
        assert file_worker.coq is not None
        success = False
        try:
            file_worker.run_into_task(job, tactic_prefix)
            success = evaluate_proof(self.args, file_worker.coq, self.predictor,
                                     self.v_network)
        except coq_serapy.CoqAnomaly:
            file_worker.restart_coq()
            file_worker.reset_file_state()
            file_worker.enter_file(job.filename)
            if restart:
                return self.evaluate_job(job, tactic_prefix, restart=False)
        proof_name = coq_serapy.lemma_name_from_statement(job.lemma_statement)
        if success:
            eprint(f"Solved proof {proof_name}!")
        else:
            eprint(f"Failed to solve proof {proof_name}")
        return success

    def check_vval_steps_task(self, task: RLTask) -> None:
        file_worker = self._get_worker(str(task.src_file))
        assert file_worker.coq
        file_worker.run_into_task(task.to_job(), task.tactic_prefix)
        check_vval_steps(self.args, file_worker.coq,
                         self.predictor, self.v_network)


    def estimate_starting_vval(self,
                               job: ReportJob, tactic_prefix: List[str],
                               restart: bool = True) -> float :
        file_worker = self._get_worker(job.filename)
        assert file_worker.coq
        try:
            file_worker.run_into_task(job, tactic_prefix)
        except coq_serapy.CoqAnomaly:
            if restart:
                file_worker.restart_coq()
                file_worker.reset_file_state()
                file_worker.enter_file(job.filename)
                return self.estimate_starting_vval(job, tactic_prefix, restart=False)
            raise

        context: Optional[ProofContext] = file_worker.coq.proof_context
        if len(tactic_prefix) > 0:
            prev_tactic = tactic_prefix[-1]
        else:
            prev_tactic = "Proof."
        assert context is not None
        all_obl_scores = self.v_network(
          context.fg_goals,
          [prev_tactic]
          * len(context.fg_goals))
        assert len(context.fg_goals) == 1

        resulting_state_val = math.prod([s.item() for s in all_obl_scores])

        return resulting_state_val


    def sync_networks(self) -> None:
        assert self.target_v_network.network
        assert self.v_network.network
        self.target_v_network.network.load_state_dict(self.v_network.network.state_dict())

def switch_dict_from_args(args: argparse.Namespace) -> Optional[Dict[str, str]]:
    if args.splits_file:
        with args.splits_file.open('r') as f:
            project_dicts = json.loads(f.read())
        if any("switch" in item for item in project_dicts):
            return  {item["project_name"]: item["switch"]
                     for item in project_dicts}
        else:
            return None
    else:
        return None

def possibly_resume_rworker(args: argparse.Namespace) \
        -> Tuple[bool, ReinforcementWorker, int, Any, Dict[RLTask, int]]:
    # predictor = get_predictor(args)
    # predictor = DummyPredictor()
    predictor = MemoizingPredictor(get_predictor(args))
    assert isinstance(predictor.underlying_predictor, FeaturesPolyargPredictor)
    tactic_vocab_size = predictor.underlying_predictor.prev_tactic_vocab_size
    if args.resume == "ask" and args.output_file.exists():
        print(f"Found existing weights at {args.output_file}. Resume?")
        response = input("[Y/n] ")
        if response.lower() not in ["no", "n"]:
            resume = "yes"
        else:
            resume = "no"
    elif not args.output_file.exists():
        assert args.resume != "yes", \
                "Can't resume because output file " \
                f"{args.output_file} doesn't already exist."
        resume = "no"
    else:
        resume = args.resume

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if resume == "yes":
        is_singlethreaded, replay_buffer, steps_already_done, network_state, \
          tnetwork_state, shorter_proofs_dict, random_state = \
            torch.load(str(args.output_file), map_location=device)
        if is_singlethreaded:
            random.setstate(random_state)
        print(f"Resuming from existing weights of {steps_already_done} steps")
        if args.start_from is not None:
            print("WARNING: Not starting from weights passed in --start-from because we're resuming from a partial run")
        v_network = VNetwork(args, None, predictor.underlying_predictor)
        target_network = VNetwork(args, None, predictor.underlying_predictor)
        v_network.load_state(network_state)
        target_network.load_state(tnetwork_state)
        assert v_network.network
        assert target_network.network
        v_network.network.to(device)
        target_network.network.to(device)
        
        # This ensures that the target and obligation will share a cache for coq2vec encodings
        target_network.obligation_encoder = v_network.obligation_encoder

    else:
        assert resume == "no"
        steps_already_done = 0
        replay_buffer = None
        random_state = random.getstate()
        with print_time("Building models"):
            v_network = VNetwork(args, args.coq2vec_weights,
                                 predictor.underlying_predictor)
            target_network = VNetwork(args, args.coq2vec_weights,
                                      predictor.underlying_predictor)
            # This ensures that the target and obligation will share a cache for coq2vec encodings
            target_network.obligation_encoder = v_network.obligation_encoder
            if args.start_from is not None:
               _, _, _, network_state, \
                 tnetwork_state, shorter_proofs_dict, _ = \
                   torch.load(str(args.start_from), map_location=device)
               v_network.obligation_encoder = None
               target_network.obligation_encoder = None
               v_network.load_state(network_state)
               target_network.load_state(tnetwork_state)
               assert v_network.network
               assert target_network.network
               v_network.network.to(device)
               target_network.network.to(device)
            else:
                shorter_proofs_dict = {}

        is_singlethreaded = True
    worker = ReinforcementWorker(args, predictor, v_network, target_network,
                                 switch_dict_from_args(args),
                                 initial_replay_buffer = replay_buffer)
    return is_singlethreaded, worker, steps_already_done, random_state, shorter_proofs_dict

def reinforce_jobs(args: argparse.Namespace) -> None:
    eprint("Starting reinforce_jobs")

    is_singlethreaded, worker, steps_already_done, random_state, shorter_proofs_dict = \
        possibly_resume_rworker(args)
    if is_singlethreaded:
        random.setstate(random_state)

    if args.tasks_file:
        tasks = read_tasks_file(args.tasks_file, args.curriculum)
    else:
        tasks = [RLTask.from_job(job) for job in get_all_jobs(args)]

    if args.interleave:
        task_episodes = tasks * args.num_episodes
    else:
        task_episodes = [task_ep for task in tasks for task_ep in [task] * args.num_episodes]

    if not is_singlethreaded:
        assert steps_already_done >= len(task_episodes), (steps_already_done, len(task_episodes))

    assert worker.v_network.adjuster is not None
    if is_singlethreaded:
        for step in range(steps_already_done):
            with nostderr():
                worker.v_network.adjuster.step()

    step = 0
    for step, task in enumerate(tqdm(task_episodes[steps_already_done:],
                                     initial=steps_already_done,
                                     total=len(task_episodes)),
                                     start=steps_already_done+1):
        cur_epsilon = args.starting_epsilon + ((step / len(task_episodes)) *
                                               (args.ending_epsilon - args.starting_epsilon))
        sol_length = worker.run_job_reinforce(task.to_job(), task.tactic_prefix, cur_epsilon)
        if sol_length and sol_length < shorter_proofs_dict.get(task, task.target_length):
            shorter_proofs_dict[task] = sol_length
        if (step + 1) % args.train_every == 0:
            if os.path.isfile("vvalverify.dat") :
                os.remove('vvalverify.dat')
            with print_time("Training", guard=args.print_timings):
                worker.train(step)
        if (step + 1) % args.save_every == 0:
            save_state(args, worker, shorter_proofs_dict, step + 1)
        if args.verifyv_every is not None and (step + 1) % args.verifyv_every == 0:
            verify_vvals(args, worker, [task], shorter_proofs_dict)
    if steps_already_done < len(task_episodes) or len(task_episodes) == 0:
        with print_time("Saving"):
            save_state(args, worker, shorter_proofs_dict, step)
    if args.evaluate or args.evaluate_baseline or args.evaluate_random_baseline:
        if args.test_file:
            test_tasks = read_tasks_file(args.test_file, False)
            evaluate_results(args, worker, test_tasks)
        else:
            evaluate_results(args, worker, tasks)
    if args.verifyv_steps:
        for idx, task in enumerate(tasks):
            eprint(f"Task {idx}:")
            worker.check_vval_steps_task(task)
    elif args.verifyvval:
        print("Verifying VVals")
        verify_vvals(args, worker, tasks, shorter_proofs_dict)

def read_tasks_file(task_file: str, curriculum: bool):
    tasks = []
    with open(task_file, "r") as f:
        tasks = [RLTask(**json.loads(line)) for line in f]
    if curriculum:
        tasks = sorted(tasks, key=lambda t: t.target_length)
    return tasks

class CachedObligationEncoder(coq2vec.CoqContextVectorizer):
    obl_cache: OrderedDict[Obligation, torch.FloatTensor]
    max_size: int
    def __init__(self, term_encoder: 'coq2vec.CoqTermRNNVectorizer',
            max_num_hypotheses: int, max_size: int=5000) -> None:
        super().__init__(term_encoder, max_num_hypotheses)
        self.obl_cache = OrderedDict()
        self.max_size = max_size #TODO: Add in arguments if desired
    def obligations_to_vectors_cached(self, obls: List[Obligation]) \
            -> torch.FloatTensor:
        encoded_obl_size = self.term_encoder.hidden_size * (self.max_num_hypotheses + 1)

        cached_results = []
        for obl in obls:
            r = self.obl_cache.get(obl, None)
            if r is not None:
                self.obl_cache.move_to_end(obl)
            cached_results.append(r)

        encoded = run_network_with_cache(
            lambda x: self.obligations_to_vectors(x).view(len(x), encoded_obl_size),
            [coq2vec.Obligation(list(obl.hypotheses), obl.goal) for obl in obls],
            cached_results)

        # for row, obl in zip(encoded, obls):
        #     assert obl not in self.obl_cache or (self.obl_cache[obl] == row).all(), \
        #         (self.obl_cache[obl] == row)

        for row, obl in zip(encoded, obls):
            self.obl_cache[obl] = row
            self.obl_cache.move_to_end(obl)
            if len(self.obl_cache) > self.max_size:
                self.obl_cache.popitem(last=False)
        return encoded

class VModel(nn.Module):
    tactic_embedding: nn.Embedding
    prediction_network: nn.Module
    prev_tactic_vocab_size: int

    def __init__(self, local_context_embedding_size: int,
                 previous_tactic_vocab_size: int,
                 previous_tactic_embedding_size: int,
                 hidden_size: int,
                 num_layers: int) -> None:
        super().__init__()
        self.tactic_embedding = nn.Embedding(previous_tactic_vocab_size,
                                             previous_tactic_embedding_size)
        layers: List[nn.Module] = [nn.Linear(local_context_embedding_size +
                                   previous_tactic_embedding_size,
                                   hidden_size)]
        for layer_idx in range(num_layers - 1):
            layers += [nn.ReLU(), nn.Linear(hidden_size, hidden_size)]
        layers += [nn.ReLU(), nn.Linear(hidden_size, 1), nn.Sigmoid()]
        self.prediction_network = nn.Sequential(*layers)
        self.prev_tactic_vocab_size = previous_tactic_vocab_size
        pass
    def forward(self, local_context_embeddeds: torch.FloatTensor,
                prev_tactic_indices: torch.LongTensor) -> torch.FloatTensor:
        return self.prediction_network(torch.cat(
            (local_context_embeddeds,
             self.tactic_embedding(prev_tactic_indices)),
            dim=1))

class VNetwork:
    obligation_encoder: Optional[CachedObligationEncoder]
    predictor: FeaturesPolyargPredictor
    network: Optional[VModel]
    steps_trained: int
    optimizer: Optional[optim.Optimizer]
    adjuster: Optional[scheduler.LRScheduler]
    device: str
    learning_rate_step: int
    learning_rate_decay: float
    learning_rate: float
    hidden_size: int
    num_layers: int
    trainable: bool

    def get_state(self) -> Any:
        assert self.obligation_encoder is not None
        assert self.network is not None
        return (self.network.state_dict(),
                self.obligation_encoder.term_encoder.get_state(),
                self.obligation_encoder.obl_cache,
                self.training_args)

    def _load_encoder_state(self, encoder_state: Any) -> None:
        term_encoder = coq2vec.CoqTermRNNVectorizer()
        term_encoder.load_state(encoder_state)
        num_hyps = 5
        assert self.obligation_encoder is None, "Can't load weights twice!"
        self.obligation_encoder = CachedObligationEncoder(
          term_encoder, num_hyps)
        local_context_embedding_size = \
          term_encoder.hidden_size * (num_hyps + 1)
        self.tactic_vocab_size: int = self.predictor.prev_tactic_vocab_size
        self.network = VModel(local_context_embedding_size,
                              self.tactic_vocab_size,
                              self.training_args.tactic_embedding_size,
                              self.training_args.hidden_size,
                              self.training_args.num_layers)
        self.network.to(self.device)
        if self.trainable:
            self.optimizer: optim.Optimizer = \
                optimizers[self.training_args.optimizer](
                  self.network.parameters(),
                  lr=self.training_args.learning_rate)
            self.adjuster = scheduler.StepLR(self.optimizer,
                                             self.training_args.learning_rate_step,
                                             self.training_args.learning_rate_decay)
        else:
            self.optimizer = None
            self.adjuster = None

    def load_state(self, state: Any) -> None:
        assert len(state) == 4
        network_state, encoder_state, obl_cache, training_args = state
        self.training_args = training_args
        if "learning_rate" not in vars(training_args):
            self.trainable = False
        else:
            self.trainable = True
        self._load_encoder_state(encoder_state)
        assert self.obligation_encoder
        self.obligation_encoder.obl_cache = obl_cache
        assert self.network
        self.network.load_state_dict(network_state)
        self.network.to(self.device)


    def __init__(self, args: argparse.Namespace,
                 coq2vec_weights: Optional[Path],
                 predictor: FeaturesPolyargPredictor,
                 device: Optional[str] = None) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.obligation_encoder = None
        self.training_args = args
        if "learning_rate" not in vars(args):
            self.trainable = False
        else:
            self.trainable = True
        # Steps trained only counts from the last resume! Don't use
        # for anything more essential than printing.
        self.steps_trained = 0
        self.total_loss = torch.FloatTensor([0.0]).to(self.device)
        self.predictor = predictor
        if coq2vec_weights is not None:
            self._load_encoder_state(torch.load(coq2vec_weights,
                                                map_location=self.device))

    def __call__(self, obls: Union[Obligation, List[Obligation]],
                 prev_tactics: Union[str, List[str]]) -> torch.FloatTensor:
        assert self.obligation_encoder, \
            "No loaded encoder! Pass encoder weights to __init__ or call set_state()"
        assert self.network
        if isinstance(obls, Obligation):
            assert isinstance(prev_tactics, int)
            obls = [obls]
            prev_tactics = [prev_tactics]
        else:
            assert isinstance(obls, list)
            assert isinstance(prev_tactics, list)
            if len(obls) == 0:
                return torch.FloatTensor([])

        encoded = self.obligation_encoder.\
            obligations_to_vectors_cached(obls).to(self.device)
        encoded_prev_tactics = torch.LongTensor([
          self.predictor.prev_tactic_stem_idx(prev_tactic)
          for prev_tactic in prev_tactics]).to(self.device)
        scores = self.network(encoded, encoded_prev_tactics).view(len(obls))
        return scores
    def call_encoded(self, obls: Union[Obligation, List[Obligation]],
                     encoded_prev_tactics: Union[int, List[int]]) \
          -> torch.FloatTensor:
        assert self.obligation_encoder, \
            "No loaded encoder! Pass encoder weights to __init__ or call set_state()"
        assert self.network
        if isinstance(obls, Obligation):
            assert isinstance(encoded_prev_tactics, int)
            obls = [obls]
            prev_tactics = [encoded_prev_tactics]
        else:
            assert isinstance(obls, list)
            assert isinstance(encoded_prev_tactics, list)
            if len(obls) == 0:
                return torch.FloatTensor([])

        encoded = self.obligation_encoder.\
            obligations_to_vectors_cached(obls)
        scores = self.network(encoded, encoded_prev_tactics).view(len(obls))
        return scores

    def train(self, inputs: List[Tuple[int, Obligation]],
              target_outputs: List[float],
              verbosity: int = 0) -> float:
        del verbosity
        assert self.obligation_encoder, \
            "No loaded encoder! Pass encoder weights to __init__ or call set_state()"
        assert self.trainable
        assert self.optimizer and self.adjuster
        # with print_time("Training", guard=verbosity >= 1):
        input_contexts = [input_context for prev_tactic, input_context in
                          inputs]
        input_prev_tactics = [prev_tactic
                              for prev_tactic, input_context in inputs]
        actual = self.call_encoded(input_contexts, input_prev_tactics)
        target = torch.FloatTensor(target_outputs).to(self.device)
        loss = F.mse_loss(actual, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.adjuster.step()
        # eprint(f"Actual: {actual}; Target: {target}", guard=verbosity >= 2)
        # eprint(f"Loss: {loss}", guard=verbosity >= 1)
        self.steps_trained += 1
        self.total_loss += loss
        return loss.item()

def experience_proof(args: argparse.Namespace,
                     coq: coq_serapy.CoqAgent,
                     predictor: TacticPredictor,
                     v_network: VNetwork,
                     replay_buffer: 'ReplayBuffer',
                     epsilon: float) -> Optional[int]:
    assert coq.proof_context
    path: List[ProofContext] = [unwrap(coq.proof_context)]
    initial_open_obligations = len(unwrap(coq.proof_context).all_goals)
    trace: List[str] = []
    for step in range(args.steps_per_episode):
        before_obl = unwrap(coq.proof_context).fg_goals[0]
        if args.verbose >= 3:
            coq_serapy.summarizeContext(unwrap(coq.proof_context))
        actions = predictor.predictKTactics(
            truncate_tactic_context(FullContext(coq.local_lemmas[:-1],
                                                coq.prev_tactics,
                                                unwrap(coq.proof_context)).as_tcontext(),
                                    30),
            args.num_predictions,
            blacklist=args.blacklisted_tactics)
        eprint(f"Using predictions {[action.prediction for action in actions]}",
               guard=args.verbose >= 3)
        if random.random() < epsilon:
            eprint("Using best-scoring action", guard=args.verbose >= 3)
            assert isinstance(predictor, FeaturesPolyargPredictor)
            action_scores = evaluate_actions(
                coq, v_network, path,
                [action.prediction for action in actions])
            chosen_action, chosen_score = max(zip(actions, action_scores),
                                              key=lambda p: p[1])
            if chosen_score == -float("Inf"):
                break
        else:
            eprint("Using action sampled based on supervised certainties",
                   guard=args.verbose >= 3)
            chosen_action = None
            action_list = randomly_order_by_scores(
              actions,
              lambda a: a.certainty ** ( 1 / args.smoothing_factor ))
            for action in action_list:
                try:
                    coq.run_stmt(action.prediction)
                    resulting_context = coq.proof_context
                    assert resulting_context is not None
                    coq.cancel_last_noupdate()
                    if any(coq_serapy.contextSurjective(resulting_context,
                                                        path_context)
                           for path_context in path):
                        continue
                    chosen_action = action
                    break
                except (coq_serapy.CoqTimeoutError, coq_serapy.ParseError,
                        coq_serapy.CoqExn, coq_serapy.CoqOverflowError,
                        coq_serapy.ParseError,
                        RecursionError,
                        coq_serapy.UnrecognizedError) as e:
                    eprint(f"Action produced error {e}", guard=args.verbose >= 3)
                    pass
            if chosen_action is None:
                break
        assert chosen_action is not None
        resulting_obls = execute_action(coq, chosen_action.prediction)
        trace.append(chosen_action.prediction)

        eprint(f"Taking action {chosen_action}",
               guard=args.verbose >= 2)

        assert coq.proof_context is not None
        if args.verbose >= 3:
            coq_serapy.summarizeContext(coq.proof_context)
        replay_buffer.add_transition((coq.prev_tactics[-1], before_obl,
                                      chosen_action.prediction,
                                      set(resulting_obls)))
        path.append(coq.proof_context)

        current_open_obligations = len(coq.proof_context.all_goals)
        if current_open_obligations < initial_open_obligations:
            eprint(f"Completed task with trace {trace}", guard=args.verbose >= 1)
            return step+1
    return None

def evaluate_actions(coq: coq_serapy.CoqAgent,
                     v_network: VNetwork, path: List[ProofContext],
                     actions: List[str], verbosity: int = 0) -> List[float]:
    assert isinstance(actions[0], str)
    resulting_contexts: List[Optional[ProofContext]] = \
        [action_result(coq, path, action, verbosity) for action in actions]
    num_output_obls: List[Optional[int]] = [len(context.fg_goals) if context else None
                                            for context in resulting_contexts]
    all_obls = [obl for context in resulting_contexts
                for obl in (context.fg_goals if context else [])]
    all_prev_tactics = \
      [prev_tac for prev_tac, num_obls in
       zip(actions, num_output_obls)
       for _ in (range(num_obls) if num_obls is not None else [])]
    all_obl_scores = v_network(all_obls, all_prev_tactics)
    resulting_action_scores = []
    cur_obl_idx = 0
    for num_obls in num_output_obls:
        if num_obls is None:
            resulting_action_scores.append(float("-Inf"))
        else:
            resulting_action_scores.append(math.prod(
                all_obl_scores[cur_obl_idx:cur_obl_idx+num_obls]))
            cur_obl_idx += num_obls
    return resulting_action_scores

def action_result(coq: coq_serapy.CoqAgent, path: List[ProofContext],
                  action: str, verbosity: int = 0) -> Optional[ProofContext]:
    assert isinstance(action, str)
    try:
        coq.run_stmt(action)
    except (coq_serapy.CoqTimeoutError, coq_serapy.ParseError,
            coq_serapy.CoqExn, coq_serapy.CoqOverflowError,
            coq_serapy.ParseError,
            RecursionError,
            coq_serapy.UnrecognizedError) as e:
        eprint(f"Action produced error {e}", guard=verbosity >= 3)
        return None
    context_after = unwrap(coq.proof_context)
    coq.cancel_last_noupdate()
    if any(coq_serapy.contextSurjective(context_after, path_context)
           for path_context in path):
        return None
    return context_after

def execute_action_trace(coq: coq_serapy.CoqAgent,
                         action: str) -> \
      List[Union[Tuple[str, ProofContext], str]]:
    coq.run_stmt(action)
    trace: List[Union[Tuple[str, ProofContext], str]]  = \
      [action]

    subgoals_closed = 0
    if len(unwrap(coq.proof_context).fg_goals) == 0 and \
       len(unwrap(coq.proof_context).shelved_goals) > 0: # type: ignore
        coq.run_stmt("Unshelve.")
    while len(unwrap(coq.proof_context).fg_goals) == 0 \
            and not completed_proof(coq):
        coq.run_stmt("}")
        trace.append("}")
        subgoals_closed += 1
    if coq.count_fg_goals() > 1 or \
       (coq.count_fg_goals() > 0 and subgoals_closed > 0):
        coq.run_stmt("{")
        trace.append("{")
    return trace


def execute_action(coq: coq_serapy.CoqAgent,
                   action: str) -> List[Obligation]:

    coq.run_stmt(action)
    resulting_obls = unwrap(coq.proof_context).fg_goals

    subgoals_closed = 0
    if len(unwrap(coq.proof_context).fg_goals) == 0 and \
       len(unwrap(coq.proof_context).shelved_goals) > 0: # type: ignore
        coq.run_stmt("Unshelve.")
    while len(unwrap(coq.proof_context).fg_goals) == 0 \
            and not completed_proof(coq):
        coq.run_stmt("}")
        subgoals_closed += 1
    if coq.count_fg_goals() > 1 or \
       (coq.count_fg_goals() > 0 and subgoals_closed > 0):
        coq.run_stmt("{")

    assert len(unwrap(coq.proof_context).all_goals) == 0 or len(unwrap(coq.proof_context).fg_goals) > 0

    return resulting_obls

def train_v_network(args: argparse.Namespace,
                    v_network: VNetwork,
                    target_network: VNetwork,
                    replay_buffer: 'ReplayBuffer'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = replay_buffer.sample(args.batch_size)
    if samples is None:
        return
    inputs = [(prev_tactic, start_obl)
              for (prev_tactic, start_obl), action_records
              in samples]
    num_resulting_obls = [[len(resulting_obls)
                           for action, resulting_obls in action_records]
                          for starting_obl, action_records in samples]
    all_resulting_contexts  = \
      [obl
       for (prev_tactic, starting_obl) , action_records in samples
       for action, resulting_obls in action_records
       for obl in resulting_obls]
    all_resulting_prev_tactics = [
      action
      for _, action_records in samples
      for action, resulting_obls in action_records
      for obl in resulting_obls]
    with torch.no_grad():
        all_obl_scores = target_network(all_resulting_contexts,
                                        all_resulting_prev_tactics)
    outputs = []
    cur_row = 0
    for resulting_obl_lens in num_resulting_obls:
        action_outputs = []
        for num_obls in resulting_obl_lens:
            selected_obl_scores = all_obl_scores[cur_row:cur_row+num_obls]
            action_outputs.append(args.gamma * math.prod(
                selected_obl_scores))
            cur_row += num_obls
        outputs.append(max(action_outputs))

    v_network.train(inputs, outputs, verbosity=args.verbose)
    if args.print_loss_every and (v_network.steps_trained + 1) % args.print_loss_every == 0:
        avg_loss = v_network.total_loss / args.print_loss_every
        v_network.total_loss = torch.FloatTensor([0.0]).to(device)
        assert v_network.optimizer is not None
        print(f"Loss: {avg_loss}; Learning rate: {v_network.optimizer.param_groups[0]['lr']}")

class MemoizingPredictor(TacticPredictor):
    underlying_predictor: TacticPredictor
    cache: Dict[Tuple[TacticContext, int, Optional[Sequence[str]]], List[Prediction]]
    def __init__(self, underlying_predictor: TacticPredictor) -> None:
        self.underlying_predictor = underlying_predictor
        self.cache = {}
    def getOptions(self) -> List[Tuple[str, str]]:
        raise NotImplementedError()
    def predictKTactics(self, in_data : TacticContext, k : int,
                        blacklist: Optional[List[str]] = None) \
        -> List[Prediction]:
        if in_data in self.cache:
            return self.cache[(in_data, k, tuple(blacklist) if blacklist else None)]
        predictions = self.underlying_predictor.predictKTactics(in_data, k, blacklist)
        self.cache[(in_data, k, tuple(blacklist) if blacklist else None)] = predictions
        return predictions
    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) -> \
        Tuple[List[Prediction], float]:
        raise NotImplementedError()
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, correct : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        raise NotImplementedError()

class DummyPredictor(TacticPredictor):
    def __init__(self) -> None:
        pass
    def getOptions(self) -> List[Tuple[str, str]]:
        raise NotImplementedError()

    def predictKTactics(self, in_data : TacticContext, k : int,
                        blacklist: Optional[List[str]] = None) \
        -> List[Prediction]:
        del blacklist
        del in_data
        del k
        return [Prediction("intro.", 0.25), Prediction("apply conj.", 0.25),
                Prediction("reflexivity.", 0.25), Prediction("simpl.", 0.25)]
    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) -> \
        Tuple[List[Prediction], float]:
        raise NotImplementedError()
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, correct : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        raise NotImplementedError()

def save_state(args: argparse.Namespace, worker: ReinforcementWorker,
               shorter_proofs_dict: Dict[RLTask, int],
               step: int) -> None:
    with args.output_file.open('wb') as f:
        torch.save((True,
                    worker.replay_buffer, step,
                    worker.v_network.get_state(),
                    worker.target_v_network.get_state(),
                    shorter_proofs_dict,
                    random.getstate()), f)

def path_obl_length(path: List[Union[Tuple[str, ProofContext], str]]) -> int:
    bracket_depth = 0
    cur_length = 0
    for context_or_bracket in path:
        if isinstance(context_or_bracket, str):
            bracket = context_or_bracket
            if bracket == "{":
                bracket_depth += 1
            if bracket == "}":
                if bracket_depth == 0:
                    return cur_length
                bracket_depth -= 1
        else:
            cur_length += 1
    return cur_length

def print_path_vval_errors(args: argparse.Namespace,
                           v_network: VNetwork,
                           path: List[Union[Tuple[str, ProofContext], str]]) \
      -> None:
    for step_idx, path_item in enumerate(path):
        if isinstance(path_item, str):
            continue
        prev_tactic, context = path_item
        target_steps = path_obl_length(path[step_idx:])
        assert len(context.fg_goals) == 1, len(context.fg_goals)
        vval_predicted = v_network(context.fg_goals[0], prev_tactic)
        steps_predicted = math.log(max(sys.float_info.min, vval_predicted.item())) \
            / math.log(args.gamma)
        print(f"At obligation: {coq_serapy.summarizeObligation(context.fg_goals[0])}")
        print(f"Predicted vval {vval_predicted} ({steps_predicted} steps) vs "
               f"{target_steps} actual steps")
        step_error = abs(steps_predicted - target_steps)
        print(f"Step error: {step_error}")

def check_vval_steps(args: argparse.Namespace,
                     coq: coq_serapy.CoqAgent,
                     predictor: TacticPredictor,
                     v_network: VNetwork) -> None:
    path: List[ProofContext] = [unwrap(coq.proof_context)]
    trace: List[Union[Tuple[str, ProofContext], str]] = \
        [(coq.prev_tactics[-1], unwrap(coq.proof_context))]
    initial_open_obligations = len(unwrap(coq.proof_context).all_goals)
    for _step in range(args.steps_per_episode):
        actions = predictor.predictKTactics(
            truncate_tactic_context(FullContext(
                coq.local_lemmas[:-1],
                coq.prev_tactics,
                unwrap(coq.proof_context)).as_tcontext(),
                                    30),
            args.num_predictions,
            blacklist=args.blacklisted_tactics)
        assert isinstance(predictor, FeaturesPolyargPredictor)
        action_scores = evaluate_actions(coq, v_network, path,
                                         [action.prediction for action in actions],
                                         args.verbose)
        best_action, best_score = max(zip(actions, action_scores), key=lambda p: p[1])
        if best_score == -float("Inf"):
            break
        action_trace = execute_action_trace(coq, best_action.prediction)
        current_open_obligations = len(unwrap(coq.proof_context).all_goals)
        if current_open_obligations < initial_open_obligations:
            break
        trace += action_trace
        trace.append((best_action.prediction, unwrap(coq.proof_context)))
        path.append(unwrap(coq.proof_context))
    print_path_vval_errors(args, v_network, trace)

def evaluate_proof(args: argparse.Namespace,
                   coq: coq_serapy.CoqAgent,
                   predictor: TacticPredictor,
                   v_network: VNetwork) -> bool:
    path: List[ProofContext] = [unwrap(coq.proof_context)]
    proof_succeeded = False
    initial_open_obligations = len(unwrap(coq.proof_context).all_goals)
    for _step in range(args.steps_per_episode):
        actions = predictor.predictKTactics(
            truncate_tactic_context(FullContext(
                coq.local_lemmas[:-1],
                coq.prev_tactics,
                unwrap(coq.proof_context)).as_tcontext(),
                                    30),
            args.num_predictions,
            blacklist=args.blacklisted_tactics)
        if args.verbose >= 1:
            coq_serapy.summarizeContext(unwrap(coq.proof_context))
        eprint(f"Trying predictions {[action.prediction for action in actions]}",
               guard=args.verbose >= 2)
        if args.evaluate_baseline :
            best_action,best_score = None, float("-Inf")
            for action in actions :
                if action_result(coq, path, action.prediction, args.verbose) :
                    best_action, best_score = action, action.certainty
                    break
        else:
            assert isinstance(predictor, FeaturesPolyargPredictor)
            action_scores = evaluate_actions(coq, v_network, path,
                                             [action.prediction for action in actions],
                                             args.verbose)
            best_action, best_score = max(zip(actions, action_scores), key=lambda p: p[1])
        if best_score == -float("Inf"):
            break
        assert best_action
        eprint(f"Taking action {best_action} with estimated value {best_score}",
               guard=args.verbose >= 1)
        execute_action(coq, best_action.prediction)
        path.append(unwrap(coq.proof_context))
        current_open_obligations = len(unwrap(coq.proof_context).all_goals)
        if current_open_obligations < initial_open_obligations:
            proof_succeeded = True
            break
    return proof_succeeded


def evaluate_results(args: argparse.Namespace,
                     worker: ReinforcementWorker,
                     tasks: List[RLTask]) -> None:
    proofs_completed = 0
    for task in tasks:
        if worker.evaluate_job(task.to_job(), task.tactic_prefix):
            proofs_completed += 1
    print(f"{proofs_completed} out of {len(tasks)} "
          f"tasks successfully proven "
          f"({stringified_percent(proofs_completed, len(tasks))}%)")


def verify_vvals(args: argparse.Namespace,
                 worker: ReinforcementWorker,
                 tasks: List[RLTask],
                 shorter_proofs_dict: Dict[RLTask, int]) -> None:
    resumepath = Path("vvalverify.dat")
    if args.resume == "ask" and resumepath.exists():
        print(f"Found vval verification in progress at {str(resumepath)}. Resume?")
        response = input("[Y/n] ")
        if response.lower() not in ["no", "n"]:
            args.resume = "yes"
        else:
            args.resume = "no"
    elif not resumepath.exists():
        args.resume = "no"
    if resumepath.exists() and args.resume == "yes":
        with resumepath.open('rb') as f:
            vval_err_sum, steps_already_done, jobs_skipped = pickle.load(f)
    else :
        vval_err_sum  = 0
        steps_already_done = 0
        jobs_skipped = 0


    for idx, task in enumerate(tqdm(tasks[steps_already_done:], desc="Tasks checked",
                                    initial=steps_already_done, total=len(tasks),
                                    disable=len(tasks) < 25),
                                    start=steps_already_done + 1):
        target_steps = shorter_proofs_dict.get(task, task.target_length)
        if not tactic_prefix_is_usable(task.tactic_prefix):
            eprint(f"Skipping job {task} with prefix {task.tactic_prefix} because it can't purely focused")
            jobs_skipped += 1
            continue
        assert isinstance(worker.predictor, MemoizingPredictor)
        assert isinstance(worker.predictor.underlying_predictor,
                          FeaturesPolyargPredictor)
        vval_predicted = worker.estimate_starting_vval(
            task.to_job(),
            task.tactic_prefix)
        steps_predicted = math.log(max(sys.float_info.min, vval_predicted)) / math.log(args.gamma)
        eprint(f"Predicted vval {vval_predicted} ({steps_predicted} steps) vs "
               f"{target_steps} actual steps", guard=args.verbose >= 1)
        step_error = abs(steps_predicted - target_steps)
        vval_err_sum += step_error
        if idx%100 == 0 :
            with open('vvalverify.dat','wb') as f :
                pickle.dump((vval_err_sum ,idx, jobs_skipped),f)
    print(f"Average step error: {vval_err_sum/(len(tasks) - jobs_skipped)}")

    if len(tasks) > 100:
        os.remove('vvalverify.dat')


def stringified_percent(total : float, outof : float) -> str:
    if outof == 0:
        return "NaN"
    return f"{(total * 100 / outof):10.2f}"

Transition = Tuple[str, Set[Obligation]]
FullTransition = Tuple[int, Obligation, str, Set[Obligation]]

class ReplayBuffer:
    _contents: Dict[Tuple[int, Obligation], Tuple[int, Set[Transition]]]
    window_size: int
    window_end_position: int
    allow_partial_batches: bool
    def __init__(self, window_size: int,
                 allow_partial_batches: bool) -> None:
        self.window_size = window_size
        self.window_end_position = 0
        self.allow_partial_batches = allow_partial_batches
        self._contents = {}

    def sample(self, batch_size) -> Optional[List[Tuple[Tuple[int, Obligation], Set[Transition]]]]:
        sample_pool: List[Tuple[Tuple[int, Obligation], Set[Transition]]] = []
        for obl, (last_updated, transitions) in self._contents.copy().items():
            if last_updated <= self.window_end_position - self.window_size:
                del self._contents[obl]
            else:
                sample_pool.append((obl, transitions))
        if len(sample_pool) >= batch_size:
            return random.sample(sample_pool, batch_size)
        if self.allow_partial_batches and len(sample_pool) > 0:
            return sample_pool
        return None

    def add_transition(self, transition: FullTransition) -> None:
        prev_tactic = transition[0]
        from_obl = transition[1]
        action = transition[2]
        to_obls = set(transition[3])
        self._contents[(prev_tactic, from_obl)] = \
            (self.window_end_position,
             {(action, to_obls)} |
             self._contents.get((prev_tactic, from_obl), (0, set()))[1])
        self.window_end_position += 1

# NOT THREAD SAFE
@contextlib.contextmanager
def log_time(msg: str) -> Iterator[None]:
    start = time.time()
    try:
        yield
    finally:
        time_taken = time.time() - start
        try:
            with open("timings.json", 'r') as f:
                timings = json.load(f)
        except FileNotFoundError:
            timings = {}
        timings[msg] = time_taken + timings.get(msg, 0.0)
        with open("timings.json", 'w') as f:
            json.dump(timings, f)

# This function takes a function which can run on multiple inputs in
# batch, a list of input values, and a list containing outputs for
# some of the values. It then calls the function on all the values
# that don't have an output, and puts combines their outputs with the
# given output list to create a non-None output value for every input,
# efficiently.
T = TypeVar('T')
def run_network_with_cache(f: Callable[[List[T]], torch.FloatTensor],
                           values: List[T],
                           cached_outputs: List[Optional[torch.FloatTensor]]) \
                           -> torch.FloatTensor:
    assert len(values) == len(cached_outputs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_list: List[Optional[torch.FloatTensor]] = list(cached_outputs)
    uncached_values: List[T] = []
    uncached_value_indices: List[int] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i, value in enumerate(values):
        if output_list[i] is None:
            uncached_values.append(value)
            uncached_value_indices.append(i)
    if len(uncached_values) > 0:
        new_results = f(uncached_values).to(device)
        for idx, result in zip(uncached_value_indices, new_results):
            output_list[idx] = result
    
    return torch.cat([unwrap(t).unsqueeze(0) for t in output_list], dim=0)

def tactic_prefix_is_usable(tactic_prefix: List[str]):
    for tactic in tactic_prefix:
        if re.match("\s*(\d\s*,?\s*)+\s*:", tactic) or re.match("\s*all\s*:", tactic) \
           or re.match("\s*Focus(\s+\d+)?\s*", tactic):
            warnings.warn("Warning: Tactic has banned prefix. This should have had been filtered out during gen_rl_task")
            return False
    return True

def sample_distribution(distribution: List[float]) -> int:
    rval = random.random()
    for idx, prob in enumerate(distribution):
        if rval <= prob:
            return idx
        rval -= prob
    assert False, "This shouldn't happen I think"

def softmax_scores(scores: List[float]) -> List[float]:
    exps = [math.exp(score) for score in scores]
    exp_sum = sum(exps)
    return [exp / exp_sum for exp in exps]

def randomly_order_by_scores(xs: List[T], scorer: Callable[[T], float]) -> List[T]:
    result = []
    xs_left = list(xs)
    scores_left = [scorer(x) for x in xs]
    for i in range(len(xs)):
        next_el_idx = sample_distribution(softmax_scores(scores_left))
        result.append(xs_left[next_el_idx])
        del xs_left[next_el_idx]
        del scores_left[next_el_idx]
    return result

def prev_tactic_from_prefix(tactic_prefix: List[str]) -> str:
    bracket_depth = 0
    for tac in reversed(tactic_prefix):
        if tac == "{":
            if bracket_depth > 0:
                bracket_depth -= 1
        elif tac == "}":
            bracket_depth += 1
        elif bracket_depth == 0:
            return tac
    return "Proof"

@dataclass
class EObligation:
  local_context: torch.FloatTensor
  previous_tactic: int
  context_tokens: torch.LongTensor
  def __hash__(self) -> int:
    return int.from_bytes(hashlib.md5(
      json.dumps(self.context_tokens.view(-1).tolist() +
                 [self.previous_tactic],
                 sort_keys=True).encode("utf-8")).digest(), byteorder='big')
  def context_hash(self) -> int:
    return int.from_bytes(hashlib.md5(
      json.dumps(self.context_tokens.view(-1).tolist(),
                 sort_keys=True).encode("utf-8")).digest(), byteorder='big')
  def __eq__(self, other: object) -> bool:
    if not isinstance(other, EObligation):
      return False
    return bool(torch.all(self.context_tokens == other.context_tokens)) \
             and self.previous_tactic == other.previous_tactic
  def to_dict(self) -> Dict[str, Any]:
    return {"local_context": self.local_context.view(-1).tolist(),
            "previous_tactic": self.previous_tactic,
            "context_tokens": self.context_tokens.view(-1).tolist()}
  @classmethod
  def from_dict(cls, d: Dict[str, Any]) -> 'EObligation':
    return EObligation(torch.FloatTensor(d["local_context"]),
                       d["previous_tactic"],
                       torch.LongTensor(d["context_tokens"]))

if __name__ == "__main__":
    main()
