#!/usr/bin/env python
impory ray
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
from pathlib import Path
<<<<<<< HEAD
from operator import itemgetter
from typing import (
    List,
    Optional,
    Dict,
    Tuple,
    Union,
    Any,
    Set,
    Sequence,
    TypeVar,
    Callable,
)
=======
from typing import (List, Optional, Dict, Tuple, Union, Any, Set,
                    Sequence, TypeVar, Callable, OrderedDict)

>>>>>>> rl_from_scratch

from util import unwrap, eprint, print_time, nostderr

with print_time("Importing torch"):
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch import optim
    import torch.optim.lr_scheduler as scheduler
    from models.tactic_predictor import TacticPredictor, Prediction

#pylint: disable=wrong-import-order
from tqdm import tqdm

import coq_serapy
from coq_serapy.contexts import (
    FullContext,
    truncate_tactic_context,
    Obligation,
    TacticContext,
    ProofContext,
)
import coq2vec
#pylint: enable=wrong-import-order

from gen_rl_tasks import RLTask

with print_time("Importing search code"):
    from search_file import get_all_jobs
    from search_worker import ReportJob, Worker, get_predictor
    from search_strategies import completed_proof

from util import unwrap, eprint, print_time, nostderr
from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch


def main():
    eprint("Starting main")
    parser = argparse.ArgumentParser(
        description="Train a state estimator using reinforcement learning"
<<<<<<< HEAD
        "to complete proofs using Proverbot9001."
    )
=======
        "to complete proofs using Proverbot9001.")
    add_args_to_parser(parser)
    args = parser.parse_args()

    if args.filenames[0].suffix == ".json":
        args.splits_file = args.filenames[0]
        args.filenames = None
    else:
        args.splits_file = None

    reinforce_jobs(args)

def add_args_to_parser(parser: argparse.ArgumentParser) -> None:
>>>>>>> rl_from_scratch
    parser.add_argument("--prelude", default=".", type=Path)
    parser.add_argument(
        "--output",
        "-o",
        dest="output_file",
        help="output data folder name",
        default="data/rl_weights.dat",
        type=Path,
    )
    parser.add_argument(
        "--verbose", "-v", help="verbose output", action="count", default=0
    )
    parser.add_argument(
        "--progress", "-P", help="show progress of files", action="store_true"
    )
    parser.add_argument("--print-timings", "-t", action="store_true")
    parser.add_argument("--no-set-switch", dest="set_switch", action="store_false")
    parser.add_argument("--include-proof-relevant", action="store_true")
    parser.add_argument("--backend", choices=["serapi", "lsp", "auto"], default="auto")
    parser.add_argument("filenames", help="proof file name (*.v)", nargs="+", type=Path)
    proofsGroup = parser.add_mutually_exclusive_group()
    proofsGroup.add_argument("--proof", default=None)
    proofsGroup.add_argument("--proofs-file", default=None)
    parser.add_argument("--tasks-file", default=None)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--no-interleave", dest="interleave", action="store_false")
    parser.add_argument("--supervised-weights", type=Path, dest="weightsfile")
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
    parser.add_argument("--batch-step", default=25, type=int)
    parser.add_argument("--lr-step", default=0.8, type=float)
    parser.add_argument("--batches-per-proof", default=1, type=int)
    parser.add_argument("--train-every", default=1, type=int)
    parser.add_argument("--print-loss-every", default=None, type=int)
<<<<<<< HEAD
    parser.add_argument(
        "--sync-target-every",
        help="Sync target network to v network every <n> episodes",
        default=-1,
        type=int,
    )
    parser.add_argument("--allow-partial-batches", action="store_true")
    parser.add_argument(
        "--blacklist-tactic", action="append", dest="blacklisted_tactics"
    )
    parser.add_argument("--resume", choices=["no", "yes", "ask"], default="ask")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--hyperparameter_search", action="store_true")
    args, _ = parser.parse_known_args()
    if args.hyperparameter_search:
        parser.add_argument(
            "--num-trails",
            default=20,
            type=int,
            help="Number of trails for hyperparameter search",
        )
        parser.add_argument(
            "--num-cpus",
            default=1,
            type=int,
            help="Number of CPUs available for a hyperparameter search",
        )
        parser.add_argument(
            "--num-gpus",
            default=0,
            type=int,
            help="Number of GPUs available for a hyperparameter search",
        )
        args = parser.parse_args()
    if args.filenames[0].suffix == ".json":
        args.splits_file = args.filenames[0]
        args.filenames = None
    else:
        args.splits_file = None
    if args.hyperparameter_search:
        tuning(args)
    else:
        result = reinforce_jobs(args)
=======
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
>>>>>>> rl_from_scratch


class FileReinforcementWorker(Worker):
    def finish_proof(self) -> None:
        assert self.coq
        while not coq_serapy.ending_proof(self.remaining_commands[0]):
            self.remaining_commands.pop(0)
        ending_cmd = self.remaining_commands.pop(0)
        coq_serapy.admit_proof(self.coq, self.coq.prev_tactics[0], ending_cmd)

    def run_into_task(
        self,
        job: ReportJob,
        tactic_prefix: List[str],
        restart_anomaly: bool = True,
        careful: bool = False,
    ) -> None:
        assert self.coq
        if (
            job.project_dir != self.cur_project
            or job.filename != self.cur_file
            or job.module_prefix != self.coq.sm_prefix
            or self.coq.proof_context is None
            or job.lemma_statement.strip() != self.coq.prev_tactics[0].strip()
        ):
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
                target_path = [
                    tac
                    for tac in tactic_prefix
                    if coq_serapy.kill_comments(tac).strip() != ""
                ]
                common_prefix_len = 0
                for item1, item2 in zip(cur_path, target_path):
                    if item1.strip() != item2.strip():
                        break
                    common_prefix_len += 1
                # Heuristically assume that cancelling a command is about
                # 1/20th as expensive as running one.
                if len(target_path) < (len(cur_path) - common_prefix_len) * 0.05 + (
                    len(target_path) - common_prefix_len
                ):
                    eprint(
                        f"Decided to abort because the target path is only {len(target_path)} tactics long, "
                        f"but the common prefix is only {common_prefix_len} tactics long, "
                        f"so we would have to cancel {len(cur_path) - common_prefix_len} tactics "
                        f"and rerun {len(target_path) - common_prefix_len} tactics.",
                        guard=self.args.verbose >= 1,
                    )
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


class ReinforcementWorker:
    v_network: "VNetwork"
    target_v_network: "VNetwork"
    replay_buffer: "ReplayBuffer"
    verbosity: int
    predictor: TacticPredictor
    last_worker_idx: int
    file_workers: Dict[str, Tuple[FileReinforcementWorker, int]]

    def __init__(
        self,
        args: argparse.Namespace,
        predictor: TacticPredictor,
        v_network: "VNetwork",
        target_network: "VNetwork",
        switch_dict: Optional[Dict[str, str]] = None,
        initial_replay_buffer: Optional["ReplayBuffer"] = None,
    ) -> None:
        self.args = args
        self.v_network = v_network
        self.target_v_network = target_network
        self.predictor = predictor
        self.last_worker_idx = 0
        if initial_replay_buffer:
            self.replay_buffer = initial_replay_buffer
        else:
            self.replay_buffer = ReplayBuffer(
                args.window_size, args.allow_partial_batches
            )
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

<<<<<<< HEAD
    def train(self) -> None:
        train_v_network(
            self.args, self.v_network, self.target_v_network, self.replay_buffer
        )

    def run_job_reinforce(
        self,
        job: ReportJob,
        tactic_prefix: List[str],
        epsilon: float,
        restart: bool = True,
    ) -> None:
        file_worker = self._get_worker(job.filename)
=======
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
>>>>>>> rl_from_scratch
        assert file_worker.coq is not None
        try:
            with print_time("Running into task", guard=self.args.print_timings):
                file_worker.run_into_task(job, tactic_prefix)
            with print_time("Experiencing", guard=self.args.print_timings):
<<<<<<< HEAD
                experience_proof(
                    self.args,
                    file_worker.coq,
                    self.predictor,
                    self.v_network,
                    self.replay_buffer,
                    epsilon,
                )
=======
                return experience_proof(self.args,
                                        file_worker.coq,
                                        self.predictor, self.v_network,
                                        self.replay_buffer,
                                        epsilon)
>>>>>>> rl_from_scratch
        except coq_serapy.CoqAnomaly:
            eprint("Encountered Coq anomaly.")
            file_worker.restart_coq()
            file_worker.reset_file_state()
            file_worker.enter_file(job.filename)
<<<<<<< HEAD

    def evaluate_job(
        self, job: ReportJob, tactic_prefix: List[str], restart: bool = True
    ) -> bool:
=======
            if restart:
                return self.run_job_reinforce(job, tactic_prefix, epsilon, restart=False)
            eprint("Encountered anomaly without restart, closing current job")
        return None
    def evaluate_job(self, job: ReportJob, tactic_prefix: List[str], restart: bool = True) \
            -> bool:
        if not tactic_prefix_is_usable(tactic_prefix):
            if self.args.verbose >= 2:
                eprint(f"Skipping job {job} with prefix {tactic_prefix} "
                       "because it can't purely focused")
            else:
                eprint("Skipping a job because it can't be purely focused")
            return None
>>>>>>> rl_from_scratch
        file_worker = self._get_worker(job.filename)
        assert file_worker.coq is not None
        success = False
        try:
<<<<<<< HEAD
            success = evaluate_proof(
                self.args, file_worker.coq, self.predictor, self.v_network
            )
=======
            file_worker.run_into_task(job, tactic_prefix)
            success = evaluate_proof(self.args, file_worker.coq, self.predictor,
                                     self.v_network)
>>>>>>> rl_from_scratch
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
        file_worker = self._get_worker(task.src_file)
        file_worker.run_into_task(task.to_job(), task.tactic_prefix)
        check_vval_steps(self.args, file_worker.coq, self.predictor, self.v_network)


    def estimate_starting_vval(self, job: ReportJob, tactic_prefix: List[str],
                               restart: bool = True) -> float :
        file_worker = self._get_worker(job.filename)
        try:
            file_worker.run_into_task(job, tactic_prefix)
        except coq_serapy.CoqAnomaly:
            if restart:
                file_worker.restart_coq()
                file_worker.reset_file_state()
                file_worker.enter_file(job.filename)
                return self.estimate_starting_vval(job, tactic_prefix, restart=False)
            raise

        context: List[Optional[ProofContext]] = file_worker.coq.proof_context
        assert context is not None
        all_obl_scores = self.v_network(context.fg_goals)

        resulting_state_val = math.prod(all_obl_scores).item()

        return resulting_state_val


    def sync_networks(self) -> None:
        self.target_v_network.network.load_state_dict(
            self.v_network.network.state_dict()
        )

def switch_dict_from_args(args: argparse.Namespace) -> Dict[str, str]:
    if args.splits_file:
        with args.splits_file.open("r") as f:
            project_dicts = json.loads(f.read())
        if any("switch" in item for item in project_dicts):
<<<<<<< HEAD
            switch_dict = {
                item["project_name"]: item["switch"] for item in project_dicts
            }
=======
            return  {item["project_name"]: item["switch"]
                     for item in project_dicts}
>>>>>>> rl_from_scratch
        else:
            return None
    else:
        return None

def possibly_resume_rworker(args: argparse.Namespace) \
        -> Tuple[bool, ReinforcementWorker, int, Any, Dict[RLTask, int]]:
    # predictor = get_predictor(args)
    # predictor = DummyPredictor()
    predictor = MemoizingPredictor(get_predictor(args))
    if args.resume == "ask" and args.output_file.exists():
        print(f"Found existing weights at {args.output_file}. Resume?")
        response = input("[Y/n] ")
        if response.lower() not in ["no", "n"]:
            resume = "yes"
        else:
            resume = "no"
    elif not args.output_file.exists():
<<<<<<< HEAD
        assert args.resume != "yes", (
            "Can't resume because output file "
            f"{args.output_file} doesn't already exist."
        )
        args.resume = "no"

    if args.resume == "yes":
        (
            replay_buffer,
            steps_already_done,
            network_state,
            tnetwork_state,
            random_state,
        ) = torch.load(str(args.output_file))
        random.setstate(random_state)
        print(f"Resuming from existing weights of {steps_already_done} steps")
        v_network = VNetwork(None, args.learning_rate, args.batch_step, args.lr_step)
        target_network = VNetwork(
            None, args.learning_rate, args.batch_step, args.lr_step
        )
        # This ensures that the target and obligation will share a cache for coq2vec encodings
        target_network.obligation_encoder = v_network.obligation_encoder
=======
        assert args.resume != "yes", \
                "Can't resume because output file " \
                f"{args.output_file} doesn't already exist."
        resume = "no"
    else:
        resume = args.resume

    if resume == "yes":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_singlethreaded, replay_buffer, steps_already_done, network_state, \
          tnetwork_state, shorter_proofs_dict, random_state = \
            torch.load(str(args.output_file), map_location=device)
        if is_singlethreaded:
            random.setstate(random_state)
        print(f"Resuming from existing weights of {steps_already_done} steps")
        v_network = VNetwork(None, args.learning_rate,
                             args.batch_step, args.lr_step)
        target_network = VNetwork(None, args.learning_rate,
                                  args.batch_step, args.lr_step)
>>>>>>> rl_from_scratch
        v_network.load_state(network_state)
        target_network.load_state(tnetwork_state)
        v_network.network.to(device)
        target_network.network.to(device)
        
        # This ensures that the target and obligation will share a cache for coq2vec encodings
        target_network.obligation_encoder = v_network.obligation_encoder

    else:
        assert resume == "no"
        steps_already_done = 0
        replay_buffer = None
        random_state = random.getstate()
        shorter_proofs_dict = {}
        with print_time("Building models"):
            v_network = VNetwork(
                args.coq2vec_weights, args.learning_rate, args.batch_step, args.lr_step
            )
            target_network = VNetwork(
                args.coq2vec_weights, args.learning_rate, args.batch_step, args.lr_step
            )
            # This ensures that the target and obligation will share a cache for coq2vec encodings
            target_network.obligation_encoder = v_network.obligation_encoder
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

<<<<<<< HEAD
    worker = ReinforcementWorker(
        args,
        predictor,
        v_network,
        target_network,
        switch_dict,
        initial_replay_buffer=replay_buffer,
    )
=======
>>>>>>> rl_from_scratch
    if args.interleave:
        task_episodes = tasks * args.num_episodes
    else:
        task_episodes = [task_ep for task in tasks for task_ep in [task] * args.num_episodes]

    if not is_singlethreaded:
        assert steps_already_done >= len(task_episodes), (steps_already_done, len(task_episodes))

<<<<<<< HEAD
    for step, (job, task_tactic_prefix) in enumerate(
        tqdm(tasks[steps_already_done:], initial=steps_already_done, total=len(tasks)),
        start=steps_already_done + 1,
    ):
        cur_epsilon = args.starting_epsilon + (
            (step / len(tasks)) * (args.ending_epsilon - args.starting_epsilon)
        )
        worker.run_job_reinforce(job, task_tactic_prefix, cur_epsilon)
=======
    if is_singlethreaded:
        for step in range(steps_already_done):
            with nostderr():
                worker.v_network.adjuster.step()

    for step, task in enumerate(tqdm(task_episodes[steps_already_done:],
                                     initial=steps_already_done,
                                     total=len(task_episodes)),
                                     start=steps_already_done+1):
        cur_epsilon = args.starting_epsilon + ((step / len(task_episodes)) *
                                               (args.ending_epsilon - args.starting_epsilon))
        sol_length = worker.run_job_reinforce(task.to_job(), task.tactic_prefix, cur_epsilon)
        if sol_length and sol_length < shorter_proofs_dict.get(task, task.target_length):
            shorter_proofs_dict[task] = sol_length
>>>>>>> rl_from_scratch
        if (step + 1) % args.train_every == 0:
            if os.path.isfile("vvalverify.dat") :
                os.remove('vvalverify.dat')
            with print_time("Training", guard=args.print_timings):
                worker.train(step)
        if (step + 1) % args.save_every == 0:
            save_state(args, worker, shorter_proofs_dict, step + 1)
        if args.verifyv_every is not None and (step + 1) % args.verifyv_every == 0:
            verify_vvals(args, worker, [task], shorter_proofs_dict)
    if steps_already_done < len(task_episodes):
        with print_time("Saving"):
            save_state(args, worker, shorter_proofs_dict, step)
    if args.evaluate or args.evaluate_baseline or args.evaluate_random_baseline:
        if args.test_file:
<<<<<<< HEAD
            test_jobs = get_job_and_prefix_from_task_file(args.test_file, args)
            evaluation_worker = ReinforcementWorker(
                args,
                predictor,
                v_network,
                target_network,
                switch_dict,
                initial_replay_buffer=replay_buffer,
            )
            evaluate_results(args, evaluation_worker, test_jobs)
        else:
            raise ValueError("No Test File Specified")


def get_job_and_prefix_from_task_file(task_file, args):
    jobs = []
    with open(task_file, "r") as f:
        readjobs = [json.loads(line) for line in f]
    if args.curriculum:
        readjobs = sorted(readjobs, key=itemgetter("target_length"), reverse=False)
    for task in readjobs:
        task_job = ReportJob(
            project_dir=".",
            filename=task["src_file"],
            module_prefix=task["module_prefix"],
            lemma_statement=task["proof_statement"],
        )
        jobs.append((task_job, task["tactic_prefix"]))
    return jobs


class CachedObligationEncoder(coq2vec.CoqContextVectorizer):
    obl_cache: Dict[Obligation, torch.FloatTensor]

    def __init__(
        self, term_encoder: "coq2vec.CoqTermRNNVectorizer", max_num_hypotheses: int
    ) -> None:
        super().__init__(term_encoder, max_num_hypotheses)
        self.obl_cache = {}

    def obligations_to_vectors_cached(
        self, obls: List[Obligation]
    ) -> torch.FloatTensor:
=======
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
>>>>>>> rl_from_scratch
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
<<<<<<< HEAD
            [self.obl_cache.get(obl, None) for obl in obls],
        )
=======
            cached_results)

>>>>>>> rl_from_scratch
        # for row, obl in zip(encoded, obls):
        #     assert obl not in self.obl_cache or (self.obl_cache[obl] == row).all(), \
        #         (self.obl_cache[obl] == row)

        for row, obl in zip(encoded, obls):
            self.obl_cache[obl] = row
            self.obl_cache.move_to_end(obl)
            if len(self.obl_cache) > self.max_size:
                self.obl_cache.popitem(last=False)
        return encoded


class VNetwork:
    obligation_encoder: Optional[coq2vec.CoqContextVectorizer]
    network: nn.Module
    steps_trained: int
    optimizer: optim.Optimizer
    batch_step: int
    lr_step: int
    learning_rate: float

    def get_state(self) -> Any:
        assert self.obligation_encoder is not None
        return (
            self.network.state_dict(),
            self.obligation_encoder.term_encoder.get_state(),
            self.obligation_encoder.obl_cache,
        )

    def _load_encoder_state(self, encoder_state: Any) -> None:
        term_encoder = coq2vec.CoqTermRNNVectorizer()
        term_encoder.load_state(encoder_state)
        num_hyps = 5
        assert self.obligation_encoder is None, "Can't load weights twice!"
        self.obligation_encoder = CachedObligationEncoder(term_encoder, num_hyps)
        insize = term_encoder.hidden_size * (num_hyps + 1)
        self.network = model_setup(insize).to(self.device)
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate)
        self.adjuster = scheduler.StepLR(self.optimizer, self.batch_step, self.lr_step)

    def load_state(self, state: Any) -> None:
        # This case exists for compatibility with older resume files that
        # didn't save the obligation cache.
        if len(state) == 2:
            network_state, encoder_state = state
            self._load_encoder_state(encoder_state)
        else:
            assert len(state) == 3
            network_state, encoder_state, obl_cache = state
            self._load_encoder_state(encoder_state)
            self.obligation_encoder.obl_cache = obl_cache
        self.network.load_state_dict(network_state)
        self.network.to(self.device)

<<<<<<< HEAD
    def __init__(
        self,
        coq2vec_weights: Optional[Path],
        learning_rate: float,
        batch_step: int,
        lr_step: float,
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
=======

    def __init__(self, coq2vec_weights: Optional[Path], learning_rate: float,
                 batch_step: int, lr_step: float) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>>>>>>> rl_from_scratch
        self.batch_step = batch_step
        self.lr_step = lr_step
        self.learning_rate = learning_rate
        self.obligation_encoder = None
        self.network = None
        # Steps trained only counts from the last resume! Don't use
        # for anything more essential than printing.
        self.steps_trained = 0
        self.total_loss = torch.tensor(0.0).to(self.device)
        if coq2vec_weights is not None:
            self._load_encoder_state(torch.load(coq2vec_weights, map_location=self.device))

    def __call__(self, obls: Union[Obligation, List[Obligation]]) -> torch.FloatTensor:
        assert (
            self.obligation_encoder
        ), "No loaded encoder! Pass encoder weights to __init__ or call set_state()"
        if isinstance(obls, Obligation):
            obls = [obls]
        else:
            assert isinstance(obls, list)
            if len(obls) == 0:
                return torch.tensor([])

        encoded = self.obligation_encoder.obligations_to_vectors_cached(obls)
        scores = self.network(encoded).view(len(obls))
        return scores

    def train(
        self, inputs: List[Obligation], target_outputs: List[float], verbosity: int = 0
    ) -> float:
        del verbosity
        assert (
            self.obligation_encoder
        ), "No loaded encoder! Pass encoder weights to __init__ or call set_state()"
        # with print_time("Training", guard=verbosity >= 1):
        actual = self(inputs)
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
        return loss

<<<<<<< HEAD

def experience_proof(
    args: argparse.Namespace,
    coq: coq_serapy.CoqAgent,
    predictor: TacticPredictor,
    v_network: VNetwork,
    replay_buffer: "ReplayBuffer",
    epsilon: float,
) -> None:
=======
def experience_proof(args: argparse.Namespace,
                     coq: coq_serapy.CoqAgent,
                     predictor: TacticPredictor,
                     v_network: VNetwork,
                     replay_buffer: 'ReplayBuffer',
                     epsilon: float) -> Optional[int]:
>>>>>>> rl_from_scratch
    path: List[ProofContext] = [coq.proof_context]
    initial_open_obligations = len(coq.proof_context.all_goals)
    trace: List[str] = []
    for step in range(args.steps_per_episode):
        before_obl = unwrap(coq.proof_context).fg_goals[0]
        if args.verbose >= 3:
            coq_serapy.summarizeContext(coq.proof_context)
        actions = predictor.predictKTactics(
            truncate_tactic_context(
                FullContext(
                    coq.local_lemmas[:-1], coq.prev_tactics, unwrap(coq.proof_context)
                ).as_tcontext(),
                30,
            ),
            args.num_predictions,
            blacklist=args.blacklisted_tactics,
        )
        eprint(
            f"Using predictions {[action.prediction for action in actions]}",
            guard=args.verbose >= 3,
        )
        if random.random() < epsilon:
            eprint("Using best-scoring action", guard=args.verbose >= 3)
            action_scores = evaluate_actions(
                coq, v_network, path, [action.prediction for action in actions]
            )
            chosen_action, chosen_score = max(
                zip(actions, action_scores), key=lambda p: p[1]
            )
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
                    coq.cancel_last_noupdate()
                    if any(
                        coq_serapy.contextSurjective(resulting_context, path_context)
                        for path_context in path
                    ):
                        continue
                    chosen_action = action
                    break
                except (
                    coq_serapy.CoqTimeoutError,
                    coq_serapy.ParseError,
                    coq_serapy.CoqExn,
                    coq_serapy.CoqOverflowError,
                    coq_serapy.ParseError,
                    RecursionError,
                    coq_serapy.UnrecognizedError,
                ) as e:
                    eprint(f"Action produced error {e}", guard=args.verbose >= 3)
                    pass
            if chosen_action is None:
                break
        resulting_obls = execute_action(coq, chosen_action.prediction)
        trace.append(chosen_action.prediction)

        eprint(f"Taking action {chosen_action}", guard=args.verbose >= 2)

        if args.verbose >= 3:
            coq_serapy.summarizeContext(coq.proof_context)
        replay_buffer.add_transition(
            (before_obl, chosen_action.prediction, resulting_obls)
        )
        path.append(coq.proof_context)

        current_open_obligations = len(coq.proof_context.all_goals)
        if current_open_obligations < initial_open_obligations:
            eprint(f"Completed task with trace {trace}", guard=args.verbose >= 1)
            return step+1
    return None


def evaluate_actions(
    coq: coq_serapy.CoqAgent,
    v_network: VNetwork,
    path: List[ProofContext],
    actions: List[str],
    verbosity: int = 0,
) -> List[float]:
    resulting_contexts: List[Optional[ProofContext]] = [
        action_result(coq, path, action, verbosity) for action in actions
    ]
    num_output_obls: List[Optional[int]] = [
        len(context.fg_goals) if context else None for context in resulting_contexts
    ]
    all_obls = [
        obl
        for context in resulting_contexts
        for obl in (context.fg_goals if context else [])
    ]
    all_obl_scores = v_network(all_obls)
    resulting_action_scores = []
    cur_obl_idx = 0
    for num_obls in num_output_obls:
        if num_obls is None:
            resulting_action_scores.append(float("-Inf"))
        else:
            resulting_action_scores.append(
                math.prod(all_obl_scores[cur_obl_idx : cur_obl_idx + num_obls])
            )
            cur_obl_idx += num_obls
    return resulting_action_scores


def action_result(
    coq: coq_serapy.CoqAgent, path: List[ProofContext], action: str, verbosity: int = 0
) -> Optional[ProofContext]:
    try:
        coq.run_stmt(action)
    except (
        coq_serapy.CoqTimeoutError,
        coq_serapy.ParseError,
        coq_serapy.CoqExn,
        coq_serapy.CoqOverflowError,
        coq_serapy.ParseError,
        RecursionError,
        coq_serapy.UnrecognizedError,
    ) as e:
        eprint(f"Action produced error {e}", guard=verbosity >= 3)
        return None
    context_after = coq.proof_context
    coq.cancel_last_noupdate()
    if any(
        coq_serapy.contextSurjective(context_after, path_context)
        for path_context in path
    ):
        return None
    return context_after

<<<<<<< HEAD
=======
def execute_action_trace(coq: coq_serapy.CoqAgent,
                         action: str) -> List[str]:
    coq.run_stmt(action)
    trace = [action]

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
>>>>>>> rl_from_scratch

def execute_action(coq: coq_serapy.CoqAgent, action: str) -> List[Obligation]:
    coq.run_stmt(action)
    resulting_obls = unwrap(coq.proof_context).fg_goals

    subgoals_closed = 0
    if (
        len(unwrap(coq.proof_context).fg_goals) == 0
        and len(unwrap(coq.proof_context).shelved_goals) > 0
    ):  # type: ignore
        coq.run_stmt("Unshelve.")
    while len(unwrap(coq.proof_context).fg_goals) == 0 and not completed_proof(coq):
        coq.run_stmt("}")
        subgoals_closed += 1
    if coq.count_fg_goals() > 1 or (coq.count_fg_goals() > 0 and subgoals_closed > 0):
        coq.run_stmt("{")

    return resulting_obls

<<<<<<< HEAD

def train_v_network(
    args: argparse.Namespace,
    v_network: VNetwork,
    target_network: VNetwork,
    replay_buffer: "ReplayBuffer",
):
    for _batch_idx in range(args.batches_per_proof):
=======
def train_v_network(args: argparse.Namespace,
                    v_network: VNetwork,
                    target_network: VNetwork,
                    replay_buffer: 'ReplayBuffer'):
>>>>>>> rl_from_scratch
        samples = replay_buffer.sample(args.batch_size)
        if samples is None:
            return
        inputs = [start_obl for start_obl, action_records in samples]
        num_resulting_obls = [
            [len(resulting_obls) for action, resulting_obls in action_records]
            for starting_obl, action_records in samples
        ]
        all_obls = [
            obl
            for starting_obl, action_records in samples
            for action, resulting_obls in action_records
            for obl in resulting_obls
        ]
        with torch.no_grad():
            all_obl_scores = target_network(all_obls)
        outputs = []
        cur_row = 0
        for resulting_obl_lens in num_resulting_obls:
            action_outputs = []
            for num_obls in resulting_obl_lens:
                selected_obl_scores = all_obl_scores[cur_row : cur_row + num_obls]
                action_outputs.append(args.gamma * math.prod(selected_obl_scores))
                cur_row += num_obls
            outputs.append(max(action_outputs))

        v_network.train(inputs, outputs, verbosity=args.verbose)
        if (
            args.print_loss_every
            and (v_network.steps_trained + 1) % args.print_loss_every == 0
        ):
            avg_loss = v_network.total_loss / args.print_loss_every
<<<<<<< HEAD
            v_network.total_loss = torch.tensor(0.0)
            print(
                f"Loss: {avg_loss}; Learning rate: {v_network.optimizer.param_groups[0]['lr']}"
            )

=======
            v_network.total_loss = torch.tensor(0.0).to(self.device)
            print(f"Loss: {avg_loss}; Learning rate: {v_network.optimizer.param_groups[0]['lr']}")
>>>>>>> rl_from_scratch

class MemoizingPredictor(TacticPredictor):
    underlying_predictor: TacticPredictor
    cache: Dict[Tuple[TacticContext, int, Optional[Sequence[str]]], List[Prediction]]

    def __init__(self, underlying_predictor: TacticPredictor) -> None:
        self.underlying_predictor = underlying_predictor
        self.cache = {}

    def getOptions(self) -> List[Tuple[str, str]]:
        raise NotImplementedError()

    def predictKTactics(
        self, in_data: TacticContext, k: int, blacklist: Optional[List[str]] = None
    ) -> List[Prediction]:
        if in_data in self.cache:
            return self.cache[(in_data, k, tuple(blacklist) if blacklist else None)]
        predictions = self.underlying_predictor.predictKTactics(in_data, k, blacklist)
        self.cache[(in_data, k, tuple(blacklist) if blacklist else None)] = predictions
        return predictions

    def predictKTacticsWithLoss(
        self, in_data: TacticContext, k: int, correct: str
    ) -> Tuple[List[Prediction], float]:
        raise NotImplementedError()

    def predictKTacticsWithLoss_batch(
        self, in_data: List[TacticContext], k: int, correct: List[str]
    ) -> Tuple[List[List[Prediction]], float]:
        raise NotImplementedError()


class DummyPredictor(TacticPredictor):
    def __init__(self) -> None:
        pass

    def getOptions(self) -> List[Tuple[str, str]]:
        raise NotImplementedError()

    def predictKTactics(
        self, in_data: TacticContext, k: int, blacklist: Optional[List[str]] = None
    ) -> List[Prediction]:
        del blacklist
        del in_data
        del k
        return [
            Prediction("intro.", 0.25),
            Prediction("apply conj.", 0.25),
            Prediction("reflexivity.", 0.25),
            Prediction("simpl.", 0.25),
        ]

    def predictKTacticsWithLoss(
        self, in_data: TacticContext, k: int, correct: str
    ) -> Tuple[List[Prediction], float]:
        raise NotImplementedError()

    def predictKTacticsWithLoss_batch(
        self, in_data: List[TacticContext], k: int, correct: List[str]
    ) -> Tuple[List[List[Prediction]], float]:
        raise NotImplementedError()

<<<<<<< HEAD

def save_state(
    args: argparse.Namespace, worker: ReinforcementWorker, step: int
) -> None:
    with args.output_file.open("wb") as f:
        torch.save(
            (
                worker.replay_buffer,
                step,
                worker.v_network.get_state(),
                worker.target_v_network.get_state(),
                random.getstate(),
            ),
            f,
        )


def evaluate_proof(
    args: argparse.Namespace,
    coq: coq_serapy.CoqAgent,
    predictor: TacticPredictor,
    v_network: VNetwork,
) -> bool:
=======
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

def path_obl_length(path: List[Union[ProofContext, str]]) -> None:
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
                           path: List[Union[ProofContext, str]]) -> None:
    for step_idx, context in enumerate(path):
        if isinstance(context, str):
            continue
        target_steps = path_obl_length(path[step_idx:])
        assert len(context.fg_goals) == 1, len(context.fg_goals)
        vval_predicted = v_network(context.fg_goals[0])
        steps_predicted = math.log(max(sys.float_info.min, vval_predicted)) \
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
    path: List[Union[ProofContext, str]] = [coq.proof_context]
    trace = list(path)
    initial_open_obligations = len(coq.proof_context.all_goals)
    for _step in range(args.steps_per_episode):
        actions = predictor.predictKTactics(
            truncate_tactic_context(FullContext(
                coq.local_lemmas[:-1],
                coq.prev_tactics,
                unwrap(coq.proof_context)).as_tcontext(),
                                    30),
            args.num_predictions,
            blacklist=args.blacklisted_tactics)
        action_scores = evaluate_actions(coq, v_network, path,
                                         [action.prediction for action in actions],
                                         args.verbose)
        best_action, best_score = max(zip(actions, action_scores), key=lambda p: p[1])
        if best_score == -float("Inf"):
            break
        action_trace = execute_action_trace(coq, best_action.prediction)
        current_open_obligations = len(coq.proof_context.all_goals)
        if current_open_obligations < initial_open_obligations:
            break
        trace += action_trace
        trace.append(coq.proof_context)
        path.append(coq.proof_context)
    print_path_vval_errors(args, v_network, trace)

def evaluate_proof(args: argparse.Namespace,
                   coq: coq_serapy.CoqAgent,
                   predictor: TacticPredictor,
                   v_network: VNetwork) -> bool:
>>>>>>> rl_from_scratch
    path: List[ProofContext] = [coq.proof_context]
    proof_succeeded = False
    initial_open_obligations = len(coq.proof_context.all_goals)
    for _step in range(args.steps_per_episode):
        actions = predictor.predictKTactics(
            truncate_tactic_context(
                FullContext(
                    coq.local_lemmas[:-1], coq.prev_tactics, unwrap(coq.proof_context)
                ).as_tcontext(),
                30,
            ),
            args.num_predictions,
            blacklist=args.blacklisted_tactics,
        )
        if args.verbose >= 1:
            coq_serapy.summarizeContext(coq.proof_context)
<<<<<<< HEAD
        eprint(
            f"Trying predictions {[action.prediction for action in actions]}",
            guard=args.verbose >= 2,
        )
        action_scores = evaluate_actions(
            coq,
            v_network,
            path,
            [action.prediction for action in actions],
            args.verbose,
        )
        best_action, best_score = max(zip(actions, action_scores), key=lambda p: p[1])
=======
        eprint(f"Trying predictions {[action.prediction for action in actions]}",
               guard=args.verbose >= 2)
        if args.evaluate_baseline :
            best_action,best_score = None, float("-Inf")
            for action in actions :
                if action_result(coq, path, action.prediction, args.verbose) :
                    best_action, best_score = action, action.certainty
                    break
        else :
            action_scores = evaluate_actions(coq, v_network, path,
                                             [action.prediction for action in actions],
                                             args.verbose)
            best_action, best_score = max(zip(actions, action_scores), key=lambda p: p[1])
>>>>>>> rl_from_scratch
        if best_score == -float("Inf"):
            break
        eprint(
            f"Taking action {best_action} with estimated value {best_score}",
            guard=args.verbose >= 1,
        )
        execute_action(coq, best_action.prediction)
        path.append(coq.proof_context)
        current_open_obligations = len(coq.proof_context.all_goals)
        if current_open_obligations < initial_open_obligations:
            proof_succeeded = True
            break
    return proof_succeeded


<<<<<<< HEAD
def evaluate_results(
    args: argparse.Namespace, worker: ReinforcementWorker, jobs: List[tuple]
) -> None:
=======
def evaluate_results(args: argparse.Namespace,
                     worker: ReinforcementWorker,
                     tasks: List[RLTask]) -> None:
>>>>>>> rl_from_scratch
    proofs_completed = 0
    for task in tasks:
        if worker.evaluate_job(task.to_job(), task.tactic_prefix):
            proofs_completed += 1
<<<<<<< HEAD
    print(
        f"{proofs_completed} out of {len(jobs)} "
        f"tasks successfully proven "
        f"({stringified_percent(proofs_completed, len(jobs))}%)"
    )
    return proofs_completed
=======
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
        vval_predicted = worker.estimate_starting_vval(task.to_job(), task.tactic_prefix)
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

>>>>>>> rl_from_scratch


def stringified_percent(total: float, outof: float) -> str:
    if outof == 0:
        return "NaN"
    return f"{(total * 100 / outof):10.2f}"


Transition = Tuple[str, Sequence[Obligation]]
FullTransition = Tuple[Obligation, str, List[Obligation]]


class ReplayBuffer:
    _contents: Dict[Obligation, Tuple[int, Set[Transition]]]
    window_size: int
    window_end_position: int
<<<<<<< HEAD
    allow_partial_batches: int

    def __init__(self, window_size: int, allow_partial_batches: bool) -> None:
=======
    allow_partial_batches: bool
    def __init__(self, window_size: int,
                 allow_partial_batches: bool) -> None:
>>>>>>> rl_from_scratch
        self.window_size = window_size
        self.window_end_position = 0
        self.allow_partial_batches = allow_partial_batches
        self._contents = {}

    def sample(self, batch_size) -> Optional[List[Tuple[Obligation, Set[Transition]]]]:
        sample_pool: List[Tuple[Obligation, List[Transition]]] = []
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
        from_obl = transition[0]
        action = transition[1]
        to_obls = tuple(transition[2])
        self._contents[from_obl] = (
            self.window_end_position,
            {(action, to_obls)} | self._contents.get(from_obl, (0, set()))[1],
        )
        self.window_end_position += 1


# NOT THREAD SAFE
@contextlib.contextmanager
def log_time(msg: str) -> None:
    start = time.time()
    try:
        yield
    finally:
        time_taken = time.time() - start
        try:
            with open("timings.json", "r") as f:
                timings = json.load(f)
        except FileNotFoundError:
            timings = {}
        timings[msg] = time_taken + timings.get(msg, 0.0)
        with open("timings.json", "w") as f:
            json.dump(timings, f)


# This function takes a function which can run on multiple inputs in
# batch, a list of input values, and a list containing outputs for
# some of the values. It then calls the function on all the values
# that don't have an output, and puts combines their outputs with the
# given output list to create a non-None output value for every input,
# efficiently.
T = TypeVar("T")


def run_network_with_cache(
    f: Callable[[List[T]], torch.FloatTensor],
    values: List[T],
    cached_outputs: List[Optional[torch.FloatTensor]],
) -> torch.FloatTensor:
    assert len(values) == len(cached_outputs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_list: List[Optional[torch.FloatTensor]] = list(cached_outputs)
    uncached_values: List[T] = []
    uncached_value_indices: List[int] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, value in enumerate(values):
        if output_list[i] is None:
            uncached_values.append(value)
            uncached_value_indices.append(i)
    if len(uncached_values) > 0:
        new_results = f(uncached_values).to(device)
        for idx, result in zip(uncached_value_indices, new_results):
            output_list[idx] = result
    
    return torch.cat([t.unsqueeze(0) for t in output_list], dim=0)

<<<<<<< HEAD

def tuning(args) -> None:
    """
    This function is used for tuning hyperparameters with Ray Tune.
    """

    def objective(config, args):
        """
        This function is passed to Ray Tune to optimize hyperparameters.
        config: A dictionary of hyperparameters to optimize from ray tune.
        args: The arguments passed to the main function.
        """

        setattr(args, "gamma", config["gamma"])
        setattr(args, "starting_epsilon", config["starting_epsilon"])
        setattr(args, "batch_step", config["batch_step"])
        setattr(args, "lr_step", config["lr_step"])
        setattr(args, "batches_per_proof", config["batches_per_proof"])
        setattr(args, "sync_target_every", config["sync_target_every"])
        if args.splits_file:
            with args.splits_file.open("r") as f:
                project_dicts = json.loads(f.read())
            if any("switch" in item for item in project_dicts):
                switch_dict = {
                    item["project_name"]: item["switch"] for item in project_dicts
                }
            else:
                switch_dict = None
        else:
            switch_dict = None
        if args.test_file:
            test_jobs = get_job_and_prefix_from_task_file(args.test_file, args)
        else:
            raise ValueError("No Test File Specified")
        predictor = MemoizingPredictor(get_predictor(args))
        args.resume = "no"
        if args.resume == "yes":
            (
                replay_buffer,
                steps_already_done,
                network_state,
                tnetwork_state,
                random_state,
            ) = torch.load(str(args.output_file))
            random.setstate(random_state)
            print(f"Resuming from existing weights of {steps_already_done} steps")
            v_network = VNetwork(
                None, args.learning_rate, args.batch_step, args.lr_step
            )
            target_network = VNetwork(
                None, args.learning_rate, args.batch_step, args.lr_step
            )
            v_network.load_state(network_state)
            target_network.load_state(tnetwork_state)

        else:
            assert args.resume == "no"
            steps_already_done = 0
            replay_buffer = None
            v_network = VNetwork(
                args.coq2vec_weights, args.learning_rate, args.batch_step, args.lr_step
            )
            target_network = VNetwork(
                args.coq2vec_weights, args.learning_rate, args.batch_step, args.lr_step
            )

            if args.tasks_file:
                jobs = get_job_and_prefix_from_task_file(args.tasks_file, args)
            else:
                jobs = [(job, []) for job in get_all_jobs(args)]

        worker = ReinforcementWorker(
            args,
            predictor,
            v_network,
            target_network,
            switch_dict,
            initial_replay_buffer=replay_buffer,
        )
        if args.interleave:
            tasks = jobs * args.num_episodes
        else:
            tasks = [task for job in jobs for task in [job] * args.num_episodes]

        for step in range(steps_already_done):
            with nostderr():
                worker.v_network.adjuster.step()

        for step, (job, task_tactic_prefix) in enumerate(
            tqdm(
                tasks[steps_already_done:], initial=steps_already_done, total=len(tasks)
            ),
            start=steps_already_done + 1,
        ):
            cur_epsilon = args.starting_epsilon + (
                (step / len(tasks)) * (args.ending_epsilon - args.starting_epsilon)
            )
            worker.run_job_reinforce(job, task_tactic_prefix, cur_epsilon)
            if (step + 1) % args.train_every == 0:
                worker.train()
            if (step + 1) % args.sync_target_every == 0:
                worker.sync_networks()

        evaluation_worker = ReinforcementWorker(
            args,
            predictor,
            v_network,
            target_network,
            switch_dict,
            initial_replay_buffer=replay_buffer,
        )
        proofs_solved = evaluate_results(args, evaluation_worker, test_jobs) # Evaluate on test set

        session.report(
            {"score": proofs_solved}
        )  # report the metric you want to optimize - currently is proofs solved.
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "gamma": tune.uniform(0.1, 0.99),
        "starting_epsilon": tune.uniform(0, 1),
        "batch_step": tune.randint(1, 30),
        "lr_step": tune.uniform(0.8, 1),
        "batches_per_proof": tune.randint(10, 50),
        "sync_target_every": tune.randint(1, 250),
    } # Define the search space for hyperparameters
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(objective, args=args),
            {"cpu": args.num_cpus, "gpu": args.num_gpus},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(num_samples=args.num_trails),
    )

    results = tuner.fit()
    print(results.get_best_result(metric="score", mode="max").config) # Print the best hyperparameters

=======
def tactic_prefix_is_usable(tactic_prefix: List[str]):
    for tactic in tactic_prefix:
        if re.match("\s*\d+\s*:", tactic):
            return False
    return True

def model_setup(insize: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(insize, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 1),
        nn.Sigmoid(),
    )

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
>>>>>>> rl_from_scratch

if __name__ == "__main__":
    ray.init()
    main()
