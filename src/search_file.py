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
from datetime import datetime, timedelta
import time
import csv
import multiprocessing
import threading
import signal
import json
import json.decoder
import queue
import traceback
import subprocess
import cProfile
import copy
import functools
import signal
from typing import (List, Tuple, NamedTuple, Optional, Dict,
                    Union, Callable, cast, IO, TypeVar,
                    Any, Iterator, Iterable)

from models.tactic_predictor import TacticPredictor
import coq_serapy as serapi_instance

from util import eprint, FileLock
import search_report
from predict_tactic import static_predictors
from search_results import SearchResult
from search_worker import ReportJob, Worker, get_files_jobs, get_predictor, project_dicts_from_args
import util

from tqdm import tqdm
from pathlib import Path
import torch

start_time = datetime.now()
def main(arg_list: List[str]) -> None:
    multiprocessing.set_start_method('spawn')
    sys.setrecursionlimit(100000)

    args, _, parser = parse_arguments(arg_list)
    # util.use_cuda = False
    # with util.silent():

    if not args.gpus and util.use_cuda:
        torch.cuda.set_device(f"cuda:{args.gpu}") # type: ignore
        util.cuda_device = f"cuda:{args.gpu}"

    if not args.predictor and not args.weightsfile:
        print("You must specify a weightsfile or a predictor.")
        parser.print_help()
        sys.exit(1)


    search_file_multithreaded(args)


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
    parser.add_argument("--max-steps", type=int, default=None)
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
    parser.add_argument("--search-type", choices=['dfs', 'dfs-est', 'beam-bfs', 'astar', 'best-first'], default='dfs')
    parser.add_argument("--scoring-function", choices=["lstd", "certainty", "pickled", "const", "norm-certainty", "pickled-normcert"], default="certainty")
    parser.add_argument("--backend", choices=['serapi', 'lsp', 'auto'], default='auto')
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
    parser.add_argument("--just-print-jobs", action='store_true', help="Just print the jobs you *would* do, then exit")
    parser.add_argument("--features-json", action='store_true')
    parser.add_argument("--search-prefix", type=str, default=None)
    parser.add_argument("--no-set-switch", dest="set_switch", action='store_false')
    parser.add_argument("--blacklist-tactic", action="append", dest="blacklisted_tactics")
    parser.add_argument("--log-explored-states", type=Path, default=None)

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

def search_file_worker(args: argparse.Namespace,
                       jobs: 'multiprocessing.Queue[ReportJob]',
                       done:
                       'multiprocessing.Queue['
                       '  Tuple[ReportJob, SearchResult]]',
                       worker_idx: int,
                       device: str) -> None:
    sys.setrecursionlimit(100000)

    predictor = get_predictor(args)

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

    if args.log_explored_states is not None:
        with args.log_explored_states.open('w') as f:
            pass

    with Worker(args, worker_idx, predictor, switch_dict) as worker:
        while True:
            try:
                next_job = jobs.get_nowait()
            except queue.Empty:
                return
            solution = worker.run_job(next_job, restart=not args.hardfail)
            done.put((next_job, solution))

def get_already_done_jobs(args: argparse.Namespace) -> List[ReportJob]:
    already_done_jobs: List[ReportJob] = []

    project_dicts = project_dicts_from_args(args)
    for project_dict in project_dicts:
        for filename in project_dict["test_files"]:
            fixing_duplicates = False
            file_jobs: List[Tuple[ReportJob, Any]] = []
            proofs_file = (args.output_dir / project_dict["project_name"] /
                           (util.safe_abbrev(Path(filename),
                                             [Path(filename) for filename in
                                              project_dict["test_files"]])
                            + "-proofs.txt"))
            try:
                file_lines = None
                backoff_amount = 0.001
                while file_lines is None:
                    try:
                        with proofs_file.open('r') as f, FileLock(f, exclusive=False):
                            file_lines = list(f)
                    except FileNotFoundError:
                        raise
                    except OSError as e:
                        eprint("Error reading proofs files "
                               "(OSError, probably a temporary cluster filesystem problem). "
                               f"Trying again in {backoff_amount} seconds.\n{e}")
                        time.sleep(backoff_amount)
                        backoff_amount *= 2
                for idx, line in enumerate(file_lines):
                    try:
                        (job_project, job_file, job_module, job_lemma), sol = json.loads(line)
                    except json.decoder.JSONDecodeError:
                        print(f"On line {idx} in file {proofs_file}")
                        raise
                    assert Path(job_file) == Path(filename), f"Job found in file {filename} " \
                        f"doesn't match it's filename {filename}. {job_file}"
                    loaded_job = ReportJob(job_project, job_file, job_module, job_lemma)
                    if loaded_job in [job for job, sol in file_jobs]:
                        eprint(f"In project {project_dict['project_name']} "
                               f"file {filename} "
                               f"found duplicate job {loaded_job}. "
                               f"Automatically removing it...")
                        fixing_duplicates = True
                    else:
                        assert loaded_job not in already_done_jobs, \
                          f"Already found job {loaded_job} in another file!"
                        file_jobs.append((loaded_job, sol))
                already_done_jobs.extend([job for job, sol in file_jobs])
                if fixing_duplicates:
                    with proofs_file.open('w') as f, FileLock(f):
                        for job, sol in file_jobs:
                            print(json.dumps((job, sol)), file=f)
            except FileNotFoundError:
                pass

            if len(set(already_done_jobs)) < len(already_done_jobs):
                eprint(f"After adding file {filename} from project {project_dict['project_name']}, there are duplicates in the solution!")
                sys.exit(1)

    return already_done_jobs

def get_all_jobs(args: argparse.Namespace) -> List[ReportJob]:
    project_dicts = project_dicts_from_args(args)
    proj_filename_tuples = [(project_dict["project_name"], filename)
                            for project_dict in project_dicts
                            for filename in project_dict["test_files"]]
    return list(get_files_jobs(args, tqdm(proj_filename_tuples, desc="Getting jobs")))

def remove_already_done_jobs(args: argparse.Namespace) -> None:
    project_dicts = project_dicts_from_args(args)
    for project_dict in project_dicts:
        for filename in project_dict["test_files"]:
            proofs_file = (args.output_dir / project_dict["project_name"] /
                           (util.safe_abbrev(Path(filename),
                                             [Path(filename) for filename in
                                              project_dict["test_files"]])
                            + "-proofs.txt"))
            try:
                os.remove(proofs_file)
            except FileNotFoundError:
                pass

def search_file_multithreaded(args: argparse.Namespace) -> None:
    global start_time
    os.makedirs(str(args.output_dir), exist_ok=True)
    start_time = datetime.now()
    all_jobs = get_all_jobs(args)
    assert len(all_jobs) > 0, "No jobs found! Maybe you passed a bad proof parameter?"
    if args.resume:
        solved_jobs = [job for job in get_already_done_jobs(args)
                       if job in all_jobs]
        try:
            with open(args.output_dir / "time_so_far.txt", 'r') as f:
                time_taken = util.read_time_taken(f.read())
                start_time = datetime.now() - time_taken
        except FileNotFoundError:
            assert len(solved_jobs) == 0, "Trying to resume but can't find a time record!"
            pass
    else:
        remove_already_done_jobs(args)
        solved_jobs = []
    todo_jobs = [job for job in all_jobs if job not in solved_jobs]
    assert len(todo_jobs) == len(all_jobs) - len(solved_jobs),\
      f"{len(todo_jobs)} != {len(all_jobs)} - {len(solved_jobs)}"
    if args.just_print_jobs:
        for job in todo_jobs:
            print(job)
        sys.exit(0)
    with multiprocessing.Manager() as manager:
        jobs: multiprocessing.Queue[ReportJob] = multiprocessing.Queue()
        done: multiprocessing.Queue[
            Tuple[ReportJob, SearchResult]
        ] = multiprocessing.Queue()


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
        # This cast appears to be needed due to a buggy type stub on
        # multiprocessing.Manager()
        workers = [multiprocessing.Process(target=search_file_worker,
                                           args=(args,
                                                 jobs, done, widx,
                                                 worker_devices[widx % len(worker_devices)]))
                   for widx in range(num_threads)]
        for worker in workers:
            worker.start()
        num_already_done = len(solved_jobs)
        os.makedirs(args.output_dir, exist_ok=True)
        with util.sighandler_context(signal.SIGINT, functools.partial(exit_early, args)):
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
    time_taken = datetime.now() - start_time
    write_time(args)
    if args.generate_report:
        with open(args.output_dir / "args.json", 'w') as f:
            json.dump({k: f"\"{v}\"" if isinstance(v, (Path, str))
                       else str(v) for k, v in vars(args).items()}, f)
        predictor = get_predictor(args)
        search_report.generate_report(args, predictor, project_dicts_from_args(args),
                                      time_taken)

def format_arg_value(v: Any) -> str:
    if isinstance(v, (Path, str)):
        return f"\"{v}\""
    if isinstance(v, list):
        return "[" + ",".join([format_arg_value(item) for item in v]) + "]"
    return str(v)
def write_time(args: argparse.Namespace) -> None:
    global start_time
    with open(args.output_dir / "time_so_far.txt", 'w') as f:
        time_taken = datetime.now() - start_time
        print(str(time_taken), file=f)
def exit_early(args: argparse.Namespace, *rest) -> None:
    write_time(args)
    sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])
