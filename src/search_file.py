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
import os
import sys
import re
import datetime
import time
import csv
import multiprocessing
import threading
import json
import queue
import traceback
import subprocess
import cProfile
import copy
from typing import (List, Tuple, NamedTuple, Optional, Dict,
                    Union, Callable, cast, IO, TypeVar,
                    Any, Iterator, Iterable)
from shutil import copyfile

from models.tactic_predictor import TacticPredictor, Prediction
from predict_tactic import (static_predictors, loadPredictorByFile,
                            loadPredictorByName)
import coq_serapy as serapi_instance
import coq_serapy
from coq_serapy import (ProofContext, Obligation, SerapiInstance)

import data
from coq_serapy.contexts import (ScrapedTactic)
from util import (unwrap, eprint, escape_filename, escape_lemma_name,
                  mybarfmt, split_by_char_outside_matching, nostderr)
import search_report
from search_results import (SearchResult, SearchStatus, KilledException,
                            TacticInteraction)
from search_strategies import best_first_proof_search, bfs_beam_proof_search, dfs_proof_search_with_graph
import multi_project_report
import util
import tokenizer

from dataclasses import dataclass
from tqdm import tqdm
from pathlib_revised import Path2
from pathlib import Path
import torch

unnamed_goal_number: int = 0


def main(arg_list: List[str]) -> None:
    multiprocessing.set_start_method('spawn')
    sys.setrecursionlimit(100000)

    args, _, parser = parse_arguments(arg_list)
    # util.use_cuda = False
    # with util.silent():

    if not args.gpus and util.use_cuda:
        torch.cuda.set_device(f"cuda:{args.gpu}") # type: ignore
        util.cuda_device = f"cuda:{args.gpu}"

    predictor = get_predictor(parser, args)

    search_file_multithreaded(args, predictor)


def add_args_to_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prelude", default=".", type=Path)
    parser.add_argument("--output", "-o", dest="output_dir",
                        help="output data folder name",
                        default="search-report",
                        type=Path)
    parser.add_argument("--no-generate-report", dest="generate_report",
                        action="store_false")
    parser.add_argument("--verbose", "-v", help="verbose output",
                        action="count", default=0)
    parser.add_argument("--progress", "-P", help="show progress of files",
                        action='store_true')
    parser.add_argument("--read-progress", "-R",
                        help="show progress of reading the file",
                        action='store_true')
    parser.add_argument("--hardfail", "-f",
                        help="fail when hitting a coq anomaly",
                        action='store_true')
    parser.add_argument('--context-filter', dest="context_filter", type=str,
                        default=None)
    parser.add_argument('--weightsfile', default=None, type=Path)
    parser.add_argument('--predictor', choices=list(static_predictors.keys()),
                        default=None)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument("--gpus", default=None, type=str)
    parser.add_argument("--no-truncate_semicolons", dest="truncate_semicolons",
                        action='store_false')
    parser.add_argument("--search-width", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--search-depth", type=int, default=6)
    parser.add_argument("--astar-steps", type=int, default=1024)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--hard-depth-limit", dest="hard_depth_limit",
                        type=int, default=100)
    parser.add_argument("--max-subgoals", type=int, default=16)
    parser.add_argument("--no-resume", dest="resume", action='store_false')
    parser.add_argument("--overwrite-mismatch", dest="overwrite_mismatch",
                        action='store_true')
    parser.add_argument("--max-print-term", dest="max_print_term", type=int,
                        default=None)
    parser.add_argument("--max-print-hyps", dest="max_print_hyps", type=int,
                        default=None)
    parser.add_argument("--max-print-subgoals", dest="max_print_subgoals",
                        type=int, default=2)
    parser.add_argument("--max-proof-time", dest="max_proof_time",
                        type=float, default=300)
    parser.add_argument("--max-tactic-time", type=float, default=2)
    parser.add_argument("--linearize", action='store_true')
    parser.add_argument("--proof-times", default=None, type=Path)
    parser.add_argument('filenames', help="proof file name (*.v)",
                        nargs='+', type=Path)
    parser.add_argument("--splits-file", default=None, type=Path)
    parser.add_argument("--use-hammer",
                        help="Use Hammer tactic after every predicted tactic",
                        action='store_const', const=True, default=False)
    parser.add_argument("--include-proof-relevant", action="store_true")
    # parser.add_argument('--no-check-consistent', action='store_false',
    #                     dest='check_consistent')
    parser.add_argument('--show-failing-predictions', action='store_true')
    parser.add_argument('--count-failing-predictions', action='store_true',
                        dest="count_failing_predictions")
    parser.add_argument('--count-softfail-predictions', action='store_true',
                        dest="count_softfail_predictions")
    parser.add_argument("--careful", action='store_true')
    parser.add_argument("--relevant-lemmas", dest="relevant_lemmas",
                        choices=['local', 'hammer', 'searchabout'],
                        default='local')
    parser.add_argument("--command-limit", type=int, default=None)
    parser.add_argument("--search-type", choices=['dfs', 'beam-bfs', 'astar', 'best-first'], default='dfs')
    parser.add_argument("--scoring-function", choices=["lstd", "certainty", "pickled", "const", "norm-certainty"], default="certainty")
    parser.add_argument("--pickled-estimator", type=Path, default=None)
    proofsGroup = parser.add_mutually_exclusive_group()
    proofsGroup.add_argument("--proof", default=None)
    proofsGroup.add_argument("--proofs-file", default=None)
    parser.add_argument("--log-anomalies", type=Path, default=None)
    parser.add_argument("--log-hard-anomalies", type=Path, default=None)
    parser.add_argument("-j", "--num-threads", type=int, default=5)
    parser.add_argument("--max-term-length", type=int, default=256)
    parser.add_argument("--add-env-lemmas", type=Path, default=None)
    parser.add_argument("--add-axioms", type=Path, default=None)
    parser.add_argument("--max-search-time-per-lemma", default=None, type=float)
    parser.add_argument("--tactics-file", type=Path, default=Path("tactics.txt"))
    parser.add_argument("--tokens-file", type=Path, default=Path("tokens.txt"))
    parser.add_argument("--beta-file", type=Path, default=Path("beta.txt"))

def parse_arguments(args_list: List[str]) -> Tuple[argparse.Namespace,
                                                   List[str],
                                                   argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Produce an html report from attempting "
        "to complete proofs using Proverbot9001.")
    add_args_to_parser(parser)

    if __name__ == "__main__":
        known_args = parser.parse_args(args_list)
        unknown_args: List[str] = []
    else:
        known_args, unknown_args = parser.parse_known_args(args_list)
    if known_args.filenames[0].suffix == ".json":
        assert known_args.splits_file == None
        assert len(known_args.filenames) == 1
        known_args.splits_file = known_args.filenames[0]
        known_args.filenames = []
    return known_args, unknown_args, parser




def get_predictor(parser: argparse.ArgumentParser,
                  args: argparse.Namespace) -> TacticPredictor:
    predictor: TacticPredictor
    if args.weightsfile:
        predictor = loadPredictorByFile(args.weightsfile)
    elif args.predictor:
        predictor = loadPredictorByName(args.predictor)
    else:
        print("You must specify either --weightsfile or --predictor!")
        parser.print_help()
        sys.exit(1)
    return predictor


def search_file_worker_profiled(
        args: argparse.Namespace,
        predictor: TacticPredictor,
        predictor_lock: threading.Lock,
        jobs: 'multiprocessing.Queue[Tuple[str, str, str]]',
        done:
        'multiprocessing.Queue['
        '  Tuple[Tuple[str, str, str], SearchResult]]',
        worker_idx: int,
        device: str) -> None:
    cProfile.runctx('search_file_worker(args, predictor, '
                    'predictor_lock, jobs, done, worker_idx, device)',
                    globals(), locals(), 'searchstats-{}'.format(worker_idx))

class ReportJob(NamedTuple):
    project_dir: str
    filename: str
    module_prefix: str
    lemma_statement: str

class Worker:
    args: argparse.Namespace
    widx: int
    predictor: TacticPredictor
    coq: Optional[serapi_instance.SerapiInstance]
    switch_dict: Optional[Dict[str, str]]

    # File-local state
    cur_project: Optional[str]
    cur_file: Optional[str]
    last_program_statement: Optional[str]
    lemmas_encountered: List[ReportJob]
    remaining_commands: List[str]

    def __init__(self, args: argparse.Namespace, worker_idx: int,
                 predictor: TacticPredictor,
                 switch_dict: Optional[Dict[str, str]] = None) -> None:
        self.args = args
        self.widx = worker_idx
        self.predictor = predictor
        self.coq = None
        self.cur_file: Optional[str] = None
        self.cur_project: Optional[str] = None
        self.last_program_statement: Optional[str] = None
        self.lemmas_encountered: List[ReportJob] = []
        self.remaining_commands: List[str] = []
        self.switch_dict = switch_dict

    def __enter__(self) -> 'Worker':
        self.coq = serapi_instance.SerapiInstance(['sertop', '--implicit'],
                                  None, str(self.args.prelude),
                                  use_hammer=self.args.use_hammer)
        self.coq.quiet = True
        self.coq.verbose = self.args.verbose
        return self
    def __exit__(self, type, value, traceback) -> None:
        assert self.coq
        self.coq.kill()
        self.coq = None

    def set_switch_from_proj(self) -> None:
        assert self.cur_project
        try:
            with (self.args.prelude / self.cur_project / "switch.txt").open('r') as sf:
                switch = sf.read().strip()
        except FileNotFoundError:
            if self.switch_dict is not None:
                switch = self.switch_dict[self.cur_project]
            else:
                return
        env_string = subprocess.run(f"opam env --switch={switch} --set-switch",
                                    shell=True, stdout=subprocess.PIPE, text=True).stdout
        for env_line in env_string.splitlines():
            linematch = re.fullmatch(r"(\w*)='([^;]*)'; export (\w*);", env_line)
            assert linematch, env_line
            envvar = linematch.group(1)
            assert envvar == linematch.group(3)
            envval = linematch.group(2)
            os.environ[envvar] = envval

    def restart_coq(self) -> None:
        assert self.coq
        self.coq.kill()
        self.coq = serapi_instance.SerapiInstance(['sertop', '--implicit'],
                                  None, str(self.args.prelude / self.cur_project),
                                  use_hammer=self.args.use_hammer)
        self.coq.quiet = True
        self.coq.verbose = self.args.verbose

    def reset_file_state(self) -> None:
        self.last_program_statement = None
        self.lemmas_encountered = []
        self.remaining_commands = []

    def enter_file(self, filename: str) -> None:
        assert self.coq
        self.cur_file = filename
        module_name = serapi_instance.get_module_from_filename(filename)
        self.coq.run_stmt(f"Module {module_name}.")
        self.remaining_commands = serapi_instance.load_commands_preserve(
            self.args, 1, self.args.prelude / self.cur_project / filename)

    def run_into_job(self, job: ReportJob, restart: bool = True) -> None:
        assert self.coq
        assert job not in self.lemmas_encountered, "Jobs are out of order!"
        job_project, job_file, job_module, job_lemma = job
        # If we need to change projects, we'll have to reset the coq instance
        # to load new includes, and set the opam switch
        if job_project != self.cur_project:
            self.reset_file_state()
            self.cur_project = job_project
            self.set_switch_from_proj()
            self.restart_coq()
            self.enter_file(job_file)
        # If the job is in a different file load the jobs file from scratch.
        if job_file != self.cur_file:
            if self.cur_file:
                for sec_or_mod, _ in reversed(self.coq.sm_stack):
                    self.coq.run_stmt(f"End {sec_or_mod}.", timeout=240)
            self.enter_file(job_file)

        # This loop has three exit cases.  Either it will hit the correct job
        # and return, hit an error or assert before getting to the correct job,
        # or get to the end of the file and raise an assert.
        while True:
            try:
                if not self.coq.proof_context:
                    rest_commands, run_commands = \
                      unwrap(cast(Optional[Tuple[List[str], List[str]]],
                                  self.coq.run_into_next_proof(
                                      self.remaining_commands)))
                    assert rest_commands, f"Couldn't find lemma {job_lemma}"
            except serapi_instance.CoqAnomaly:
                if restart:
                    self.restart_coq()
                    self.reset_file_state()
                    self.enter_file(job_file)
                    eprint(f"Hit a coq anomaly! Restarting...",
                           guard=self.args.verbose >= 1)
                    self.run_into_job(job, restart=False)
                    return
                else:
                    assert False
            except serapi_instance.SerapiException:
                eprint(f"Failed getting to before: {job_lemma}")
                eprint(f"In file {job_file}")
                raise
            for command in run_commands:
                if re.match("\s*Program\s+.*",
                            serapi_instance.kill_comments(
                                command).strip()):
                    self.last_program_statement = command
            lemma_statement = run_commands[-1]
            if re.match(r"\s*Next\s+Obligation\s*\.\s*",
                        serapi_instance.kill_comments(
                            lemma_statement).strip()):
                assert self.last_program_statement
                obligation_num = 0
                while self.coq.local_lemmas[-(obligation_num+2)] == ":":
                    obligation_num += 1
                unique_lemma_statement = \
                    self.last_program_statement + \
                    f" Obligation {obligation_num}."
            else:
                unique_lemma_statement = lemma_statement
            self.remaining_commands = rest_commands
            if unique_lemma_statement == job_lemma and \
              self.coq.sm_prefix == job_module:
                return
            else:
                self.skip_proof(lemma_statement)
                self.lemmas_encountered.append(ReportJob(self.cur_project,
                                                         unwrap(self.cur_file),
                                                         self.coq.sm_prefix,
                                                         unique_lemma_statement))

    def skip_proof(self, lemma_statement: str) -> None:
        assert self.coq
        ending_command = None
        for cmd in self.remaining_commands:
            if serapi_instance.ending_proof(cmd):
                ending_command = cmd
                break
        assert ending_command
        proof_relevant = ending_command.strip() == "Defined." or \
            bool(re.match(
                r"\s*Derive",
                serapi_instance.kill_comments(lemma_statement))) or \
            bool(re.match(
                r"\s*Let",
                serapi_instance.kill_comments(lemma_statement))) or \
            bool(re.match(
                r"\s*Equations",
                serapi_instance.kill_comments(lemma_statement))) or \
            self.args.careful
        if proof_relevant:
            self.remaining_commands, _ = unwrap(self.coq.finish_proof(
                self.remaining_commands)) # type: ignore
        else:
            try:
                serapi_instance.admit_proof(self.coq, lemma_statement, ending_command)
            except serapi_instance.SerapiException:
                lemma_name = \
                  serapi_instance.lemma_name_from_statement(lemma_statement)
                eprint(f"{self.cur_file}: Failed to admit proof {lemma_name}")
                raise

            while not serapi_instance.ending_proof(self.remaining_commands[0]):
                self.remaining_commands.pop(0)
            # Pop the actual Qed/Defined/Save
            self.remaining_commands.pop(0)

    def run_job(self, job: ReportJob, restart: bool = True) -> SearchResult:
        assert self.coq
        self.run_into_job(job, restart=restart)
        job_project, job_file, job_module, job_lemma = job
        initial_context: ProofContext = unwrap(self.coq.proof_context)
        empty_context = ProofContext([], [], [], [])
        try:
            search_status, tactic_solution = \
              attempt_search(self.args, job_lemma,
                             self.coq.sm_prefix,
                             self.coq,
                             self.args.output_dir / self.cur_project,
                             self.widx, self.predictor)
        except KilledException:
            tactic_solution = None
            search_status = SearchStatus.INCOMPLETE
        except serapi_instance.CoqAnomaly:
            if self.args.hardfail:
                raise
            if self.args.log_anomalies:
                with self.args.log_anomalies.open('a') as f:
                    print(f"ANOMALY at {job_file}:{job_lemma}",
                          file=f)
                    traceback.print_exc(file=f)
            self.restart_coq()
            self.reset_file_state()
            if restart:
                self.enter_file(job_file)
                eprint("Hit an anomaly, restarting job", guard=self.args.verbose >= 2)
                return self.run_job(job, restart=False)
            else:
                if self.args.log_hard_anomalies:
                    with self.args.log_hard_anomalies.open('a') as f:
                        print(
                            f"HARD ANOMALY at "
                            f"{job_file}:{job_lemma}",
                            file=f)
                        traceback.print_exc(file=f)

                search_status = SearchStatus.CRASHED
                solution: List[TacticInteraction] = []
                eprint(f"Skipping job {job_file}:{coq_serapy.lemma_name_from_statement(job_lemma)} "
                       "due to multiple failures",
                       guard=self.args.verbose >= 1)
                return SearchResult(search_status, solution)
        except Exception:
            eprint(f"FAILED in file {job_file}, lemma {job_lemma}")
            raise
        if not tactic_solution:
            solution = [
                TacticInteraction("Proof.", initial_context),
                TacticInteraction("Admitted.", initial_context)]
        else:
            solution = (
                [TacticInteraction("Proof.", initial_context)]
                + tactic_solution +
                [TacticInteraction("Qed.", empty_context)])

        while not serapi_instance.ending_proof(self.remaining_commands[0]):
            self.remaining_commands.pop(0)
        # Pop the actual Qed/Defined/Save
        ending_command = self.remaining_commands.pop(0)
        serapi_instance.admit_proof(self.coq, job_lemma, ending_command)

        self.lemmas_encountered.append(job)
        return SearchResult(search_status, solution)

def search_file_worker(args: argparse.Namespace,
                       predictor: TacticPredictor,
                       predictor_lock: threading.Lock,
                       jobs: 'multiprocessing.Queue[ReportJob]',
                       done:
                       'multiprocessing.Queue['
                       '  Tuple[ReportJob, SearchResult]]',
                       worker_idx: int,
                       device: str) -> None:
    sys.setrecursionlimit(100000)
    # util.use_cuda = False
    if util.use_cuda:
        torch.cuda.set_device(device) # type: ignore
    util.cuda_device = device

    if args.splits_file:
        with args.splits_file.open('r') as f:
            project_dicts = json.loads(f.read())
        if any(["switch" in item for item in project_dicts]):
            switch_dict = {item["project_name"]: item["switch"]
                           for item in project_dicts}
        else:
            switch_dict = None
    else:
        switch_dict = None

    with Worker(args, worker_idx, predictor) as worker:
        while True:
            try:
                next_job = jobs.get_nowait()
            except queue.Empty:
                return
            solution = worker.run_job(next_job, restart=not args.hardfail)
            done.put((next_job, solution))

def project_dicts_from_args(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.splits_file:
        with args.splits_file.open('r') as f:
            project_dicts = json.loads(f.read())
    else:
        project_dicts = [{"project_name": ".",
                          "test_files": [str(filename) for filename in args.filenames]}]
    return project_dicts

def get_already_done_jobs(args: argparse.Namespace) -> List[ReportJob]:
    already_done_jobs: List[ReportJob] = []

    project_dicts = project_dicts_from_args(args)
    for project_dict in project_dicts:
        for filename in project_dict["test_files"]:
            proofs_file = (args.output_dir / project_dict["project_name"] /
                           (util.safe_abbrev(Path(filename),
                                             [Path(filename) for filename in
                                              project_dict["test_files"]])
                            + "-proofs.txt"))
            try:
                with proofs_file.open('r') as f:
                    for line in f:
                        (job_project, job_file, job_module, job_lemma), sol = json.loads(line)
                        already_done_jobs.append(ReportJob(job_project,
                                                           job_file,
                                                           job_module,
                                                           job_lemma))
            except FileNotFoundError:
                pass

    return already_done_jobs

def get_file_jobs(args: argparse.Namespace,
                  proj_filename_tuples: Iterable[Tuple[str, str]]) \
                  -> Iterator[ReportJob]:
    arg_proofs_names = None
    if args.proofs_file:
        with open(args.proofs_file, 'r') as f:
            arg_proofs_names = [line.strip() for line in f]
    elif args.proof:
        arg_proofs_names = [args.proof]

    for project, filename in proj_filename_tuples:
        cmds = serapi_instance.load_commands(args.prelude / project / filename)
        lemmas_in_file = serapi_instance.lemmas_in_file(filename, cmds,
                                                        args.include_proof_relevant)
        if arg_proofs_names:
            yield from (ReportJob(project, filename, module, stmt)
                        for (module, stmt) in lemmas_in_file
                        if serapi_instance.lemma_name_from_statement(stmt)
                        in arg_proofs_names)
        else:
            yield from (ReportJob(project, filename, module, stmt)
                        for (module, stmt) in lemmas_in_file)

def get_all_jobs(args: argparse.Namespace) -> List[ReportJob]:
    project_dicts = project_dicts_from_args(args)
    proj_filename_tuples = [(project_dict["project_name"], filename)
                            for project_dict in project_dicts
                            for filename in project_dict["test_files"]]
    return list(get_file_jobs(args, tqdm(proj_filename_tuples, desc="Getting jobs")))

def search_file_multithreaded(args: argparse.Namespace,
                              predictor: TacticPredictor) -> None:
    with multiprocessing.Manager() as manager:
        jobs: multiprocessing.Queue[ReportJob] = multiprocessing.Queue()
        done: multiprocessing.Queue[
            Tuple[ReportJob, SearchResult]
        ] = multiprocessing.Queue()
        solved_jobs = get_already_done_jobs(args)
        all_jobs = get_all_jobs(args)
        todo_jobs = [job for job in all_jobs if job not in solved_jobs]
        assert len(todo_jobs) == len(all_jobs) - len(solved_jobs),\
          f"{len(todo_jobs)} != {len(all_jobs)} - {len(solved_jobs)}"


        for job in todo_jobs:
            jobs.put(job)

        num_threads = min(args.num_threads,
                          len(todo_jobs))
        if util.use_cuda:
            if args.gpus:
                gpu_list = args.gpus.split(",")
            else:
                gpu_list = [args.gpu]
            worker_devices = [f"cuda:{gpu_idx}" for gpu_idx
                              in gpu_list[:min(len(gpu_list), num_threads)]]
        else:
            assert args.gpus is None, "Passed --gpus flag, but CUDA is not supported!"
            worker_devices = ["cpu"]
        worker_predictors = [copy.deepcopy(predictor)
                             for device in worker_devices]
        for predictor, device in zip(worker_predictors, worker_devices):
            predictor.to_device(device) # type: ignore
            predictor.share_memory() # type: ignore
        # This cast appears to be needed due to a buggy type stub on
        # multiprocessing.Manager()
        predictor_locks = [cast(multiprocessing.managers.SyncManager,
                                manager).Lock()
                           for predictor in worker_predictors]
        workers = [multiprocessing.Process(target=search_file_worker,
                                           args=(args,
                                                 worker_predictors[widx % len(worker_predictors)],
                                                 predictor_locks[widx % len(worker_predictors)],
                                                 jobs, done, widx,
                                                 worker_devices[widx % len(worker_predictors)]))
                   for widx in range(num_threads)]
        for worker in workers:
            worker.start()
        num_already_done = len(solved_jobs)
        with tqdm(total=len(todo_jobs) + num_already_done,
                  dynamic_ncols=True, desc="Searching proofs") as bar:
            bar.update(n=num_already_done)
            bar.refresh()
            for _ in range(len(todo_jobs)):
                (done_project, done_file, done_module, done_lemma), sol = done.get()
                if args.splits_file:
                    with args.splits_file.open('r') as splits_f:
                        project_dicts = json.loads(splits_f.read())
                    for project_dict in project_dicts:
                        if project_dict["project_name"] == done_project:
                            filenames = [Path(fname) for fname in project_dict["test_files"]]
                            break
                else:
                    filenames = args.filenames
                proofs_file = (args.output_dir / done_project /
                               (util.safe_abbrev(Path(done_file),
                                                 filenames)
                                + "-proofs.txt"))
                with proofs_file.open('a') as f:
                    f.write(json.dumps(((done_project, str(done_file), done_module, done_lemma),
                                        sol.to_dict())))
                    f.write("\n")
                bar.update()

        for worker in workers:
            worker.join()
    if args.generate_report:
        search_report.generate_report(args, predictor, project_dicts)

def try_run_prelude(args: argparse.Namespace, coq: SerapiInstance):
    if not args.weightsfile:
        eprint("No weightsfile")
        return
    prelude_path = args.weightsfile.with_suffix(".prelude.v")
    if not prelude_path.exists():
        eprint(f"Couldn't find prelude at {prelude_path}",
               guard=args.verbose >= 2)
        return
    eprint("Running prelude:", guard=args.verbose >= 2)
    prelude_commands = serapi_instance.load_commands_preserve(
        args, 0, prelude_path)
    for command in prelude_commands:
        eprint(f"Found command {command}",
               guard=args.verbose >= 2)
        coq.run_stmt(command)
    eprint("Finished with prelude",
           guard=args.verbose >= 2)


def get_lemma_declaration_from_name(coq: serapi_instance.SerapiInstance,
                                    lemma_name: str) -> str:
    return coq.check_term(lemma_name).replace("\n", "")

# The core of the search report

import _thread
import threading

# This method attempts to complete proofs using search.
def attempt_search(args: argparse.Namespace,
                   lemma_statement: str,
                   module_name: Optional[str],
                   coq: serapi_instance.SerapiInstance,
                   output_dir: Path,
                   bar_idx: int,
                   predictor: TacticPredictor) \
        -> SearchResult:
    global unnamed_goal_number
    if args.add_env_lemmas:
        with args.add_env_lemmas.open('r') as f:
            env_lemmas = [get_lemma_declaration_from_name(coq,
                                                          lemma_name.strip())
                          for lemma_name in f]
    else:
        env_lemmas = []
    if args.relevant_lemmas == "local":
        relevant_lemmas = coq.local_lemmas[:-1]
    elif args.relevant_lemmas == "hammer":
        relevant_lemmas = coq.get_hammer_premises()
    elif args.relevant_lemmas == "searchabout":
        relevant_lemmas = coq.get_lemmas_about_head()
    else:
        assert False, args.relevant_lemmas
    if module_name:
        module_prefix = escape_lemma_name(module_name)
    else:
        module_prefix = ""

    lemma_name = serapi_instance.lemma_name_from_statement(lemma_statement)
    if lemma_name == "":
        unnamed_goal_number += 1
        lemma_name = f"Obligation{unnamed_goal_number}"

    if args.max_search_time_per_lemma:
        timer = threading.Timer(args.max_search_time_per_lemma, _thread.interrupt_main)
        timer.start()
    try:
        if args.search_type == 'dfs':
            result = dfs_proof_search_with_graph(lemma_name, module_prefix,
                                                 env_lemmas + relevant_lemmas,
                                                 coq, output_dir,
                                                 args, bar_idx, predictor)
        elif args.search_type == 'beam-bfs':
            result = bfs_beam_proof_search(lemma_name, module_prefix,
                                           env_lemmas + relevant_lemmas, coq,
                                           args, bar_idx, predictor)
        elif args.search_type == 'astar' or args.search_type == 'best-first':
            result = best_first_proof_search(lemma_name, module_prefix,
                                             env_lemmas + relevant_lemmas, coq,
                                             args, bar_idx, predictor)
        else:
            assert False, args.search_type
    except KeyboardInterrupt:
        if args.max_search_time_per_lemma:
            raise KilledException("Lemma timeout")
        else:
            raise
    finally:
        if args.max_search_time_per_lemma:
            timer.cancel()
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
