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
import shutil
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
                    Any)

from models.tactic_predictor import TacticPredictor, Prediction
from predict_tactic import (static_predictors, loadPredictorByFile,
                            loadPredictorByName)
import coq_serapy as serapi_instance
from coq_serapy import (ProofContext, Obligation, SerapiInstance)

import data
from coq_serapy.contexts import (TacticContext, ScrapedTactic,
                                 truncate_tactic_context, FullContext)
from util import (unwrap, eprint, escape_filename, escape_lemma_name,
                  mybarfmt, split_by_char_outside_matching, nostderr)
import search_report
import util
import tokenizer
from dataclasses import dataclass

from tqdm import tqdm
from yattag import Doc
from pathlib import Path
from enum import Enum
import pygraphviz as pgv
import torch
Tag = Callable[..., Doc.Tag]
Text = Callable[..., None]
Line = Callable[..., None]

details_css = "details.css"
details_javascript = "search-details.js"


class SearchStatus(str, Enum):
    SUCCESS = 'SUCCESS'
    INCOMPLETE = 'INCOMPLETE'
    SKIPPED = 'SKIPPED'
    FAILURE = 'FAILURE'


class VernacBlock(NamedTuple):
    commands: List[str]


class TacticInteraction(NamedTuple):
    tactic: str
    context_before: ProofContext

    @classmethod
    def from_dict(cls, data):
        tactic = data['tactic']
        context_before = ProofContext.from_dict(data['context_before'])
        return cls(tactic, context_before)

    def to_dict(self):
        return {"tactic": self.tactic,
                "context_before": self.context_before.to_dict()}


class ProofBlock(NamedTuple):
    lemma_statement: str
    module: Optional[str]
    status: SearchStatus
    predicted_tactics: List[TacticInteraction]
    original_tactics: List[TacticInteraction]


class ArgsMismatchException(Exception):
    pass


class SourceChangedException(Exception):
    pass


class SearchResult(NamedTuple):
    status: SearchStatus
    commands: Optional[List[TacticInteraction]]

    @classmethod
    def from_dict(cls, data):
        status = SearchStatus(data['status'])
        if data['commands'] is None:
            commands = None
        else:
            commands = list(map(TacticInteraction.from_dict,
                                data['commands']))
        return cls(status, commands)

    def to_dict(self):
        assert self.commands
        return {'status': self.status.name,
                'commands': list(map(TacticInteraction.to_dict,
                                     self.commands))}


DocumentBlock = Union[VernacBlock, ProofBlock]

unnamed_goal_number: int


def main(arg_list: List[str]) -> None:
    multiprocessing.set_start_method('spawn')
    sys.setrecursionlimit(100000)

    args, parser = parse_arguments(arg_list)
    # util.use_cuda = False
    # with util.silent():

    if not args.gpus and util.use_cuda:
        torch.cuda.set_device(f"cuda:{args.gpu}")
        util.cuda_device = f"cuda:{args.gpu}"

    predictor = get_predictor(parser, args)
    base = Path(os.path.dirname(os.path.abspath(__file__)))

    if not args.output_dir.exists():
        os.makedirs(args.output_dir)

    for filename in [details_css, details_javascript]:
        destpath = args.output_dir / filename
        if not destpath.exists():
            srcpath = base.parent / 'reports' / filename
            shutil.copyfile(srcpath, destpath)

    search_file_multithreaded(args, predictor)


def parse_arguments(args_list: List[str]) -> Tuple[argparse.Namespace,
                                                   argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Produce an html report from attempting "
        "to complete proofs using Proverbot9001.")
    parser.add_argument("--prelude", default=".", type=Path)
    parser.add_argument("--output", "-o", dest="output_dir",
                        help="output data folder name",
                        default="search-report",
                        type=Path)
    parser.add_argument("--verbose", "-v", help="verbose output",
                        action="count", default=0)
    parser.add_argument("--progress", "-P", help="show progress of files",
                        action='store_true')
    parser.add_argument("--read-progress", "-p",
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
    parser.add_argument("--hard-depth-limit", dest="hard_depth_limit",
                        type=int, default=200)
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
    proofsGroup = parser.add_mutually_exclusive_group()
    proofsGroup.add_argument("--proof", default=None)
    proofsGroup.add_argument("--proofs-file", default=None)
    parser.add_argument("--log-anomalies", type=Path, default=None)
    parser.add_argument("--log-hard-anomalies", type=Path, default=None)
    parser.add_argument("-j", "--num-threads", type=int, default=5)
    parser.add_argument("--max-term-length", type=int, default=256)
    parser.add_argument("--add-env-lemmas", type=Path, default=None)
    parser.add_argument("--add-axioms", type=Path, default=None)
    if __name__ == "__main__":
        known_args = parser.parse_args(args_list)
    else:
        known_args, unknown_args = parser.parse_known_args(args_list)
    return known_args, parser


def produce_index(args: argparse.Namespace, predictor: TacticPredictor,
                  report_stats: List[search_report.ReportStats]) -> None:
    predictorOptions = predictor.getOptions()
    commit, date, weightshash = get_metadata(args)
    search_report.write_summary(args,
                                predictorOptions +
                                [("report type", "search"),
                                 ("search width", args.search_width),
                                 ("search depth", args.search_depth)],
                                predictor.unparsed_args,
                                commit, date, weightshash, report_stats)


def stats_from_blocks(blocks: List[DocumentBlock], vfilename: str) \
      -> search_report.ReportStats:
    num_proofs = 0
    num_proofs_failed = 0
    num_proofs_completed = 0
    for block in blocks:
        if isinstance(block, ProofBlock):
            if block.status != SearchStatus.SKIPPED:
                num_proofs += 1
            if block.status == SearchStatus.SUCCESS:
                num_proofs_completed += 1
            elif block.status == SearchStatus.FAILURE:
                num_proofs_failed += 1
    return search_report.ReportStats(vfilename, num_proofs,
                                     num_proofs_failed, num_proofs_completed)


def get_metadata(args: argparse.Namespace) -> Tuple[str, datetime.datetime, str]:
    cur_commit = subprocess.check_output(["git show --oneline | head -n 1"],
                                         shell=True).decode('utf-8').strip()
    cur_date = datetime.datetime.now()
    if args.weightsfile:
        weights_hash = str(subprocess.check_output(
            ["sha256sum", args.weightsfile]))
    else:
        weights_hash = ""
    return cur_commit, cur_date, weights_hash


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


# A few functions for profiling
def reset_times(args: argparse.Namespace):
    if args.proof_times:
        with args.proof_times.open('w'):
            pass


def append_time(args: argparse.Namespace, action: str, seconds: float):
    if args.proof_times:
        with args.proof_times.open('a') as f:
            f.write(f"{action}: {datetime.timedelta(seconds=seconds)}\n")


def search_file_worker_profiled(
        args: argparse.Namespace,
        predictor: TacticPredictor,
        predictor_lock: threading.Lock,
        jobs: 'multiprocessing.Queue[Tuple[str, str, str]]',
        done:
        'multiprocessing.Queue['
        '  Tuple[Tuple[str, str, str], SearchResult]]',
        worker_idx: int) -> None:
    cProfile.runctx('search_file_worker(args, predictor, '
                    'predictor_lock, jobs, done, worker_idx)',
                    globals(), locals(), 'searchstats-{}'.format(worker_idx))


def search_file_worker(args: argparse.Namespace,
                       predictor: TacticPredictor,
                       predictor_lock: threading.Lock,
                       jobs: 'multiprocessing.Queue[Tuple[str, str, str]]',
                       done:
                       'multiprocessing.Queue['
                       '  Tuple[Tuple[str, str, str], SearchResult]]',
                       worker_idx: int,
                       device: str) -> None:
    sys.setrecursionlimit(100000)
    # util.use_cuda = False
    if util.use_cuda:
        torch.cuda.set_device(device)
    util.cuda_device = device
    axioms_already_added = False
    last_program_statement = ""

    try:
        next_file, next_module, next_lemma = jobs.get_nowait()
    except queue.Empty:
        return
    with util.silent():
        all_commands = serapi_instance.load_commands_preserve(
            args, worker_idx + 1, args.prelude / next_file)

    rest_commands = all_commands
    while rest_commands:
        first_lemma_since_restart = next_lemma
        with serapi_instance.SerapiContext(["sertop", "--implicit"],
                                           serapi_instance.
                                           get_module_from_filename(next_file),
                                           str(args.prelude),
                                           use_hammer=args.use_hammer) as coq:
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
                                args, 0,
                                args.prelude / next_file)
                    rest_commands = all_commands
                    break
                except serapi_instance.SerapiException:
                    eprint(f"Failed getting to before: {next_lemma}")
                    eprint(f"In file {next_file}")
                    raise
                for command in run_commands:
                    if re.match("\s*Program\s+.*",
                                serapi_instance.kill_comments(
                                    command).strip()):
                        last_program_statement = command
                lemma_statement = run_commands[-1]
                if re.match(r"\s*Next\s+Obligation\s*\.\s*",
                            serapi_instance.kill_comments(
                                lemma_statement).strip()):
                    obligation_num = 0
                    while coq.local_lemmas[-(obligation_num+2)] == ":":
                        obligation_num += 1
                    unique_lemma_statement = \
                        last_program_statement + \
                        f" Obligation {obligation_num}."
                else:
                    unique_lemma_statement = lemma_statement

                if unique_lemma_statement == next_lemma and \
                        coq.sm_prefix == next_module:
                    if args.add_axioms and not axioms_already_added:
                        axioms_already_added = True
                        coq.cancel_last()
                        with args.add_axioms.open('r') as f:
                            for signature in f:
                                try:
                                    coq.run_stmt(signature)
                                    coq.run_stmt("Admitted.")
                                except serapi_instance.SerapyException:
                                    axiom_name = serapi_instance.lemma_name_from_statement(
                                        signature)
                                    eprint(f"Couldn't declare axiom {axiom_name} "
                                           f"at this point in the proof")
                        coq.run_stmt(lemma_statement)
                    initial_context = coq.proof_context
                    empty_context = ProofContext([], [], [], [])
                    try:
                        search_status, tactic_solution = \
                            attempt_search(args, lemma_statement,
                                           coq.sm_prefix,
                                           coq, worker_idx,
                                           predictor,
                                           predictor_lock)
                    except serapi_instance.CoqAnomaly:
                        if args.hardfail:
                            raise
                        if args.log_anomalies:
                            with args.log_anomalies.open('a') as f:
                                print(f"ANOMALY at {next_file}:{next_lemma}",
                                      file=f)
                                traceback.print_exc(file=f)
                        if first_lemma_since_restart == lemma_statement:
                            eprint("Hit the same anomaly twice! Skipping",
                                   guard=args.verbose >= 1)
                            if args.log_hard_anomalies:
                                with args.log_hard_anomalies.open('a') as f:
                                    print(
                                        f"HARD ANOMALY at "
                                        f"{next_file}:{next_lemma}",
                                        file=f)
                                    traceback.print_exc(file=f)
                            solution = [
                                TacticInteraction("Proof.", initial_context),
                                TacticInteraction("Admitted.", initial_context)
                            ]
                            done.put(((next_file, coq.module_prefix,
                                       next_lemma),
                                      SearchResult(SearchStatus.INCOMPLETE,
                                                   solution)))
                            try:
                                next_job = jobs.get_nowait()
                            except queue.Empty:
                                return
                            new_file, next_module, next_lemma = next_job
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
                                rest_commands = all_commands
                        else:
                            rest_commands = all_commands
                        break
                    except Exception:
                        eprint(f"FAILED in file {next_file}, lemma {next_lemma}")
                        raise
                    while not serapi_instance.ending_proof(rest_commands[0]):
                        rest_commands = rest_commands[1:]
                    serapi_instance.admit_proof(coq, lemma_statement,
                                                rest_commands[0])
                    if not tactic_solution:
                        solution = [
                            TacticInteraction("Proof.", initial_context),
                            TacticInteraction("Admitted.", initial_context)]
                    else:
                        solution = (
                            [TacticInteraction("Proof.", initial_context)]
                            + tactic_solution +
                            [TacticInteraction("Qed.", empty_context)])
                    rest_commands = rest_commands[1:]
                    done.put(((next_file, next_module, next_lemma),
                              SearchResult(search_status, solution)))
                    try:
                        next_job = jobs.get_nowait()
                    except queue.Empty:
                        return
                    new_file, next_module, next_lemma = next_job
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
                    ending_cmd = None
                    for cmd in rest_commands:
                        if serapi_instance.ending_proof(cmd):
                            if cmd.strip() == "Defined.":
                                proof_relevant = True
                            ending_cmd = cmd
                            break
                    assert ending_cmd is not None
                    proof_relevant = proof_relevant or \
                        bool(re.match(
                            r"\s*Derive",
                            serapi_instance.kill_comments(lemma_statement))) or \
                        bool(re.match(
                            r"\s*Let",
                            serapi_instance.kill_comments(lemma_statement))) or \
                        bool(re.match(
                            r"\s*Equations",
                            serapi_instance.kill_comments(lemma_statement))) or \
                        args.careful
                    if proof_relevant:
                        rest_commands, run_commands = coq.finish_proof(
                            rest_commands)
                    else:
                        try:
                            serapi_instance.admit_proof(coq, lemma_statement, ending_cmd)
                        except serapi_instance.SerapiException:
                            next_lemma_name = \
                                serapi_instance.lemma_name_from_statement(next_lemma)
                            eprint(f"{next_file}: Failed to admit proof {next_lemma_name}")
                            raise
                        while not serapi_instance.ending_proof(
                                rest_commands[0]):
                            rest_commands = rest_commands[1:]
                        rest_commands = rest_commands[1:]

                pass
    pass


def recover_sol(sol: Dict[str, Any]) -> SearchResult:
    return SearchResult.from_dict(sol)


def search_file_multithreaded(args: argparse.Namespace,
                              predictor: TacticPredictor) -> None:
    with multiprocessing.Manager() as manager:
        jobs: Queue[
            Tuple[str, str, str]] = multiprocessing.Queue()
        done: Queue[
            Tuple[Tuple[str, str, str], SearchResult]
        ] = multiprocessing.Queue()


        all_jobs: List[Tuple[str, str, str]] = []
        file_solutions: List[List[Tuple[Tuple[str, str, str], SearchResult]]
                             ] = [list() for _ in range(len(args.filenames))]

        for filename, solutions in zip(args.filenames, file_solutions):
            cmds = serapi_instance.load_commands_preserve(
                args, 0, args.prelude / filename)
            proofs_file = (args.output_dir /
                           (util.safe_abbrev(filename, args.filenames)
                            + "-proofs.txt"))
            all_lemma_statements = serapi_instance.lemmas_in_file(
                filename, cmds, args.include_proof_relevant)
            lemma_statements_todo = list(all_lemma_statements)

            if args.resume:
                try:
                    with proofs_file.open('r') as f:
                        eprint(f"Resuming from {str(proofs_file)}", guard=args.verbose >= 1)
                        for line in f:
                            (filename, module_prefix, done_lemma_stmt), sol = \
                                json.loads(line)
                            try:
                                lemma_statements_todo.remove((module_prefix,
                                                              done_lemma_stmt))
                            except ValueError:
                                possible_matches = [(prefix, stmt) for prefix, stmt in lemma_statements_todo if stmt == done_lemma_stmt]
                                if len(possible_matches) == 1:
                                    print(f"Couldn't find job [{filename}, {module_prefix}, {done_lemma_stmt}] from file, "
                                          f"but found [{filename}, {possible_matches[0][0]}, {done_lemma_stmt}]. Use it instead?")
                                    answer = input("(yes/no)")
                                    if answer.lower() in ["yes", "y"]:
                                        lemma_statements_todo.remove(possible_matches[0])
                                        continue
                                eprint(f"filename: {filename}, "
                                       f"module_prefix: {module_prefix}, "
                                       f"done_lemma_stmt: {done_lemma_stmt}")
                                raise
                            solutions.append(((filename, module_prefix,
                                               done_lemma_stmt),
                                              recover_sol(sol)))
                        pass
                except FileNotFoundError:
                    pass
            if args.proofs_file:
                with open(args.proofs_file, 'r') as f:
                    proof_names = [line.strip() for line in f]
                lemma_statements_todo = [
                    (module, stmt) for (module, stmt)
                    in lemma_statements_todo
                    if serapi_instance.lemma_name_from_statement(
                        stmt) in proof_names]
            elif args.proof:
                lemma_statements_todo = [
                    (module, stmt) for (module, stmt)
                    in lemma_statements_todo
                    if serapi_instance.lemma_name_from_statement(
                        stmt) == args.proof]

            for module_prefix, lemma_statement in lemma_statements_todo:
                all_jobs.append((str(filename), module_prefix,
                                 lemma_statement))
        for job in all_jobs:
            jobs.put(job)

        num_threads = min(args.num_threads,
                          len(all_jobs))
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
            predictor.to_device(device)
            predictor.share_memory()
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
        num_already_done = sum([len(solutions)
                                for solutions in file_solutions])
        with tqdm(total=len(all_jobs) + num_already_done,
                  dynamic_ncols=True) as bar:
            bar.update(n=num_already_done)
            bar.refresh()
            for _ in range(len(all_jobs)):
                (done_file, done_module, done_lemma), sol = done.get()
                proofs_file = (args.output_dir /
                               (util.safe_abbrev(Path(done_file),
                                                 args.filenames)
                                + "-proofs.txt"))
                with proofs_file.open('a') as f:
                    f.write(json.dumps(((done_file, done_module, done_lemma),
                                        sol.to_dict())))
                    f.write("\n")
                dict(zip(map(str, args.filenames),
                         file_solutions))[done_file].append(
                    ((done_file, done_module, done_lemma), sol))
                bar.update()

        for worker in workers:
            worker.join()

        model_name = dict(predictor.getOptions())["predictor"]
        stats: List[search_report.ReportStats] = []
        for filename, solutions in tqdm(zip(args.filenames, file_solutions),
                                        desc="Generating output",
                                        total=len(args.filenames)):
            blocks = blocks_from_scrape_and_sols(
                args.prelude / filename,
                [(lemma_stmt, module_name, sol)
                 for (filename, module_name, lemma_stmt), sol
                 in solutions])
            write_solution_vfile(args, filename, model_name, blocks)
            write_html(args, args.output_dir, filename, blocks)
            write_csv(args, filename, blocks)
            stats.append(stats_from_blocks(blocks, str(filename)))

        produce_index(args, predictor, stats)
    pass


def blocks_from_scrape_and_sols(
        src_filename: Path,
        lemma_statements_done: List[Tuple[str, str, SearchResult]]
        ) -> List[DocumentBlock]:

    interactions = data.read_all_text_data(
        src_filename.with_suffix(".v.scrape"))

    def lookup(module: str, lemma_stmt: str) -> Optional[SearchResult]:
        for lstmt, lmod, lresult in lemma_statements_done:
            if (lmod == module and
                    serapi_instance.kill_comments(lstmt).strip()
                    == serapi_instance.kill_comments(lemma_stmt).strip()):
                return lresult
        return None

    def generate():
        cur_lemma_stmt = ""
        unique_lemma_stmt = ""

        sm_stack = serapi_instance.initial_sm_stack(src_filename)

        tactics_interactions_batch: List[TacticInteraction] = []
        vernac_cmds_batch: List[str] = []

        in_proof = False
        obl_num = 0
        last_program_statement = ""

        def yield_proof():
            nonlocal sm_stack
            nonlocal tactics_interactions_batch
            nonlocal cur_lemma_stmt
            nonlocal unique_lemma_stmt
            nonlocal in_proof
            nonlocal obl_num
            nonlocal last_program_statement

            sm_prefix = serapi_instance.sm_prefix_from_stack(sm_stack)
            batch_without_brackets = [t for t in tactics_interactions_batch
                                      if t.tactic.strip() != "{" and
                                      t.tactic.strip() != "}"]
            result = lookup(sm_prefix, unique_lemma_stmt)
            if result is None:
                return ProofBlock(cur_lemma_stmt, sm_prefix,
                                  SearchStatus.SKIPPED, [],
                                  batch_without_brackets)
            else:
                return ProofBlock(cur_lemma_stmt, sm_prefix,
                                  result.status, result.commands,
                                  batch_without_brackets)
            tactics_interactions_batch = []

        for interaction in interactions:
            if in_proof and isinstance(interaction, str):
                in_proof = False
                yield yield_proof()
            elif in_proof and isinstance(interaction, ScrapedTactic):
                tactics_interactions_batch.append(
                    interaction_from_scraped(interaction))
            elif isinstance(interaction, ScrapedTactic):
                assert not in_proof
                cur_lemma_stmt = vernac_cmds_batch[-1]
                if re.match(r"\s*Next\s+Obligation\s*\.\s*",
                            serapi_instance.kill_comments(
                                cur_lemma_stmt).strip()):
                    unique_lemma_stmt = \
                      f"{last_program_statement} Obligation {obl_num}."
                    obl_num += 1
                else:
                    unique_lemma_stmt = cur_lemma_stmt
                yield VernacBlock(vernac_cmds_batch[:-1])
                vernac_cmds_batch = []
                tactics_interactions_batch = []
                tactics_interactions_batch.append(
                    interaction_from_scraped(interaction))
                in_proof = True
            if isinstance(interaction, str):
                sm_stack = serapi_instance.update_sm_stack(sm_stack, interaction)
                vernac_cmds_batch.append(interaction)
                if re.match(r"\s*Program\s+.*",
                            serapi_instance.kill_comments(interaction).strip()):
                    last_program_statement = interaction
                    obl_num = 0
        if in_proof:
            yield yield_proof()
        pass
    blocks = list(generate())
    return blocks


def interaction_from_scraped(s: ScrapedTactic) -> TacticInteraction:
    return TacticInteraction(s.tactic, s.context)


def write_solution_vfile(args: argparse.Namespace, filename: Path,
                         model_name: str,
                         doc_blocks: List[DocumentBlock]):
    with (args.output_dir / (util.safe_abbrev(filename, args.filenames)
                             + "-solution.v")
          ).open('w') as sfile:
        for k, v in [("search-width", args.search_width),
                     ("search-depth", args.search_depth),
                     ("model", model_name)]:
            print(f"(* {k}: {v} *)", file=sfile)

        for block in doc_blocks:
            if isinstance(block, VernacBlock):
                for cmd in block.commands:
                    print(cmd, file=sfile, end='')
            else:
                assert isinstance(block, ProofBlock)
                print(block.lemma_statement, end='', file=sfile)
                if block.predicted_tactics:
                    for tac in block.predicted_tactics:
                        print(tac.tactic, file=sfile)
                else:
                    for tac in block.original_tactics:
                        print(tac.tactic, file=sfile)

        pass


def html_header(tag: Tag, doc: Doc, text: Text, css: List[str],
                javascript: List[str], title: str) -> None:
    with tag('head'):
        for filename in css:
            doc.stag('link', href=filename, rel='stylesheet')
        for filename in javascript:
            with tag('script', type='text/javascript',
                     src=filename):
                pass
        with tag('title'):
            text(title)


def write_csv(args: argparse.Namespace, filename: str,
              doc_blocks: List[DocumentBlock]):
    with open("{}/{}.csv".format(args.output_dir,
                                 escape_filename(str(filename))),
              'w', newline='') as csvfile:
        for k, v in vars(args).items():
            csvfile.write("# {}: {}\n".format(k, v))

        rowwriter = csv.writer(csvfile, lineterminator=os.linesep)
        for block in doc_blocks:
            if isinstance(block, ProofBlock):
                rowwriter.writerow([block.lemma_statement.strip(),
                                    block.status,
                                    len(block.original_tactics)])


def write_html(args: argparse.Namespace,
               output_dir: str, filename: Path,
               doc_blocks: List[DocumentBlock]) -> None:
    global unnamed_goal_number
    unnamed_goal_number = 0
    doc, tag, text, line = Doc().ttl()
    with tag('html'):
        html_header(tag, doc, text, [details_css], [details_javascript],
                    "Proverbot Detailed Report for {}".format(str(filename)))
        with tag('body', onload='init()'), tag('pre'):
            for block_idx, block in enumerate(doc_blocks):
                if isinstance(block, VernacBlock):
                    write_commands(block.commands, tag, text, doc)
                else:
                    assert isinstance(block, ProofBlock)
                    status_klass = classFromSearchStatus(block.status)
                    write_lemma_button(block.lemma_statement, block.module,
                                       status_klass, tag, text)
                    with tag('div', klass='region'):
                        with tag('div', klass='predicted'):
                            write_tactics(args, block.predicted_tactics,
                                          block_idx,
                                          tag, text, doc)
                        with tag('div', klass='original'):
                            write_tactics(args, block.original_tactics,
                                          block_idx,
                                          tag, text, doc)
    with open("{}/{}.html".format(output_dir, escape_filename(str(filename))),
              'w') as fout:
        fout.write(doc.getvalue())


def write_lemma_button(lemma_statement: str, module: Optional[str],
                       status_klass: str, tag: Tag, text: Text):
    global unnamed_goal_number
    lemma_name = \
        serapi_instance.lemma_name_from_statement(lemma_statement)
    if module:
        module_prefix = escape_lemma_name(module)
    else:
        module_prefix = ""
    if lemma_name == "":
        unnamed_goal_number += 1
        fullname = module_prefix + lemma_name + str(unnamed_goal_number)
    else:
        fullname = module_prefix + lemma_name
    if status_klass != "skipped":
        with tag('button', klass='collapsible {}'.format(status_klass),
                 onmouseover="hoverLemma(\"{}\")".format(fullname),
                 onmouseout="unhoverLemma(\"{}\")".format(fullname)):
            with tag('code', klass='buttontext'):
                text(lemma_statement.strip())
    else:
        with tag('button', klass='collapsible {}'.format(status_klass)):
            with tag('code', klass='buttontext'):
                text(lemma_statement.strip())


def write_commands(commands: List[str], tag: Tag, text: Text, doc: Doc):
    for cmd in commands:
        with tag('code', klass='plaincommand'):
            text(cmd.strip("\n"))
        doc.stag('br')


def escape_quotes(term: str):
    return re.sub("\"", "\\\"", term)


def subgoal_to_string(args: argparse.Namespace, sg: Obligation) -> str:
    return "(\"" + escape_quotes(sg.goal[:args.max_print_term]) + "\", (\"" + \
        "\",\"".join([escape_quotes(hyp[:args.max_print_term]) for hyp in
                      sg.hypotheses[:args.max_print_hyps]]) + "\"))"


def write_tactics(args: argparse.Namespace,
                  tactics: List[TacticInteraction],
                  region_idx: int,
                  tag: Tag, text: Text, doc: Doc):
    for t_idx, t in enumerate(tactics):
        idStr = '{}-{}'.format(region_idx, t_idx)
        subgoals_str = "(" + ",".join([subgoal_to_string(args, subgoal)
                                       for subgoal in
                                       t.context_before.all_goals[
                                           :args.max_print_subgoals]]) + ")"
        with tag('span',
                 ('data-subgoals', subgoals_str),
                 id='command-{}'.format(idStr),
                 onmouseover='hoverTactic("{}")'.format(idStr),
                 onmouseout='unhoverTactic()'):
            with tag('code', klass='plaincommand'):
                text(t.tactic.strip())
            doc.stag('br')


def classFromSearchStatus(status: SearchStatus) -> str:
    if status == SearchStatus.SUCCESS:
        return 'good'
    elif status == SearchStatus.INCOMPLETE:
        return 'okay'
    elif status == SearchStatus.SKIPPED:
        return 'skipped'
    else:
        return 'bad'


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


# This method attempts to complete proofs using search.
def attempt_search(args: argparse.Namespace,
                   lemma_statement: str,
                   module_name: Optional[str],
                   coq: serapi_instance.SerapiInstance,
                   bar_idx: int,
                   predictor: TacticPredictor,
                   predictor_lock: threading.Lock) \
        -> SearchResult:
    if args.add_env_lemmas:
        with args.add_env_lemmas.open('r') as f:
            env_lemmas = [get_lemma_declaration_from_name(coq,
                                                          lemma_name.strip())
                          for lemma_name in f]
    else:
        env_lemmas = []
    result = dfs_proof_search_with_graph(lemma_statement, module_name,
                                         env_lemmas,
                                         coq,
                                         args, bar_idx, predictor,
                                         predictor_lock)
    return result


T = TypeVar('T')


def emap_lookup(emap: Dict[T, int], size: int, item: T):
    if item in emap:
        return emap[item]
    elif len(emap) < size - 1:
        emap[item] = len(emap) + 1
        return emap[item]
    else:
        return 0


class FeaturesExtractor:
    tactic_map: Dict[str, int]
    token_map: Dict[str, int]
    _num_tactics: int
    _num_tokens: int

    def __init__(self) -> None:
        self._num_tactics = 32
        self._num_tokens = 32
        self.tactic_map = {}
        self.token_map = {}

    def state_features(self, context: TacticContext) -> \
            Tuple[List[int], List[float]]:
        if len(context.prev_tactics) > 1:
            prev_tactic = serapi_instance.get_stem(context.prev_tactics[-1])
            prev_tactic_index = emap_lookup(self.tactic_map, self._num_tactics,
                                            prev_tactic)
        else:
            prev_tactic_index = 0

        if context.goal != "":
            goal_head_index = emap_lookup(self.token_map, self._num_tokens,
                                          tokenizer.get_words(context.goal)[0])
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
        stem, argument = serapi_instance.split_tactic(action)
        stem_idx = emap_lookup(self.tactic_map, self._num_tactics, stem)
        all_premises = context.hypotheses + context.relevant_lemmas
        stripped_arg = argument.strip(".").strip()
        if stripped_arg == "":
            arg_idx = 0
        else:
            index_hyp_vars = dict(serapi_instance.get_indexed_vars_in_hyps(
                all_premises))
            if stripped_arg in index_hyp_vars:
                hyp_varw, _, rest = all_premises[index_hyp_vars[stripped_arg]
                                                 ].partition(":")
                arg_idx = emap_lookup(self.token_map, self._num_tokens,
                                      tokenizer.get_words(rest)[0]) + 2
            else:
                goal_symbols = tokenizer.get_symbols(context.goal)
                if stripped_arg in goal_symbols:
                    arg_idx = emap_lookup(self.token_map, self._num_tokens,
                                          stripped_arg) + self._num_tokens + 2
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
    previous: Optional["LabeledNode"]
    children: List["LabeledNode"]


class SearchGraph:
    __graph: pgv.AGraph
    __next_node_id: int
    feature_extractor: FeaturesExtractor
    start_node: LabeledNode

    def __init__(self, lemma_name: str) -> None:
        self.__graph = pgv.AGraph(directed=True)
        self.__next_node_id = 0
        self.start_node = self.mkNode(Prediction(lemma_name, 1.0),
                                      FullContext(
                                          [], [], ProofContext([], [], [], [])),
                                      None)
        self.start_node.time_taken = 0.0
        self.feature_extractor = FeaturesExtractor()
        pass

    def mkNode(self, prediction: Prediction, context_before: FullContext,
               previous_node: Optional[LabeledNode],
               **kwargs) -> LabeledNode:
        self.__graph.add_node(self.__next_node_id,
                              label="{}\n({:.2f})".format(
                                  prediction.prediction,
                                  prediction.certainty),
                              **kwargs)
        self.__next_node_id += 1
        newNode = LabeledNode(prediction.prediction, prediction.certainty,
                              None, self.__next_node_id-1,
                              context_before, previous_node, [])
        if previous_node:
            self.__graph.add_edge(previous_node.node_id,
                                  newNode.node_id, **kwargs)
            previous_node.children.append(newNode)
        return newNode

    def mkQED(self, predictionNode: LabeledNode):
        self.mkNode(Prediction("QED", 1.0), FullContext(
            [], [], ProofContext([], [], [], [])),
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
        node_handle.attr["fillcolor"] = color
        node_handle.attr["style"] = "filled"

    def draw(self, filename: str) -> None:
        with nostderr():
            self.__graph.draw(filename, prog="dot")

    def write_feat_json(self, filename: str) -> None:
        def write_node(node: LabeledNode, f: IO[str]) -> None:
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
            write_node(self.start_node, f)


class SubSearchResult (NamedTuple):
    solution: Optional[List[TacticInteraction]]
    solved_subgoals: int


def contextInPath(full_context: ProofContext, path: List[LabeledNode]):
    return any([serapi_instance.contextSurjective(full_context,
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
                  coq: serapi_instance.SerapiInstance,
                  prediction: str,
                  previousNode: LabeledNode) \
                  -> Tuple[ProofContext, int, int, int,
                           Optional[Exception], float, bool]:
    coq.quiet = True
    time_left = max(args.max_proof_time - time_on_path(previousNode), 0)
    start_time = time.time()
    time_per_command = (coq.hammer_timeout + args.max_tactic_time
                        if coq.use_hammer else args.max_tactic_time)
    try:
        coq.run_stmt(prediction, timeout=min(time_left, time_per_command))
        error = None
    except (serapi_instance.TimeoutError, serapi_instance.ParseError,
            serapi_instance.CoqExn, serapi_instance.OverflowError,
            serapi_instance.ParseError,
            RecursionError,
            serapi_instance.UnrecognizedError) as e:
        return (unwrap(coq.proof_context), 0, 0, 0, e,
                time.time() - start_time, False)

    time_taken = time.time() - start_time
    num_stmts = 1
    subgoals_closed = 0
    unshelved = False
    if len(unwrap(coq.proof_context).fg_goals) == 0 and \
       len(unwrap(coq.proof_context).shelved_goals) > 0:
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


def contextIsBig(context: ProofContext):
    for obligation in context.all_goals:
        for hypothesis in obligation.hypotheses:
            if len(hypothesis) > goalBignessLimit:
                return True
        if len(obligation.goal) > goalBignessLimit:
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


def dfs_proof_search_with_graph(lemma_statement: str,
                                module_name: Optional[str],
                                extra_env_lemmas: List[str],
                                coq: serapi_instance.SerapiInstance,
                                args: argparse.Namespace,
                                bar_idx: int,
                                predictor: TacticPredictor,
                                predictor_lock: threading.Lock) \
                                -> SearchResult:
    global unnamed_goal_number
    unnamed_goal_number = 0
    lemma_name = serapi_instance.lemma_name_from_statement(lemma_statement)
    g = SearchGraph(lemma_name)

    if args.relevant_lemmas == "local":
        relevant_lemmas = coq.local_lemmas[:-1] + extra_env_lemmas
    elif args.relevant_lemmas == "hammer":
        relevant_lemmas = coq.get_hammer_premises() + extra_env_lemmas
    elif args.relevant_lemmas == "searchabout":
        relevant_lemmas = coq.get_lemmas_about_head() + extra_env_lemmas
    else:
        assert False, args.relevant_lemmas

    def cleanupSearch(num_stmts: int, msg: Optional[str] = None):
        if msg:
            eprint(f"Cancelling {num_stmts} statements "
                   f"because {msg}.", guard=args.verbose >= 2)
        for _ in range(num_stmts):
            coq.cancel_last()
    hasUnexploredNode = False

    def search(pbar: tqdm, current_path: List[LabeledNode],
               subgoal_distance_stack: List[int],
               extra_depth: int) -> SubSearchResult:
        nonlocal hasUnexploredNode
        nonlocal predictor_lock
        nonlocal relevant_lemmas
        global unnamed_goal_number
        full_context_before = FullContext(relevant_lemmas,
                                          coq.prev_tactics,
                                          coq.proof_context)
        with predictor_lock:
            predictions = predictor.predictKTactics(
                truncate_tactic_context(full_context_before.as_tcontext(),
                                        args.max_term_length),
                args.max_attempts)
            assert len(predictions) == args.max_attempts
        if coq.use_hammer:
            predictions = [Prediction(prediction.prediction[:-1] + "; try hammer.",
                                      prediction.certainty)
                           for prediction in predictions]
        num_successful_predictions = 0
        for prediction_idx, prediction in enumerate(predictions):
            if num_successful_predictions >= args.search_width:
                break
            try:
                context_after, num_stmts, \
                    subgoals_closed, subgoals_opened, \
                    error, time_taken, unshelved = \
                    tryPrediction(args, coq, prediction.prediction,
                                  current_path[-1])
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
                                          current_path[-1])
                predictionNode.time_taken = time_taken
                if unshelved:
                    predictionNode = g.mkNode(Prediction("Unshelve.", 1.0),
                                              full_context_before,
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
                    return SubSearchResult(solution, subgoals_closed)
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
                        and len(current_path) < args.hard_depth_limit:
                    if subgoals_closed > 0:
                        g.setNodeColor(predictionNode, "blue")
                    sub_search_result = search(pbar,
                                               current_path + [predictionNode],
                                               new_distance_stack,
                                               new_extra_depth)
                    cleanupSearch(num_stmts, "we finished subsearch")
                    if sub_search_result.solution or \
                       sub_search_result.solved_subgoals > subgoals_opened:
                        new_subgoals_closed = \
                            subgoals_closed + \
                            sub_search_result.solved_subgoals - \
                            subgoals_opened
                        return SubSearchResult(sub_search_result.solution,
                                               new_subgoals_closed)
                    if subgoals_closed > 0:
                        return SubSearchResult(None, subgoals_closed)
                else:
                    hasUnexploredNode = True
                    cleanupSearch(num_stmts, "we hit the depth limit")
                    if subgoals_closed > 0:
                        # depth = (args.search_depth + new_extra_depth + 1) \
                        #     - len(current_path)
                        return SubSearchResult(None, subgoals_closed)
            except serapi_instance.CoqAnomaly:
                predictionNode = g.mkNode(prediction,
                                          full_context_before,
                                          current_path[-1])
                g.setNodeColor(predictionNode, "grey25")
                if module_name:
                    module_prefix = escape_lemma_name(module_name)
                else:
                    module_prefix = ""
                if lemma_name == "":
                    unnamed_goal_number += 1
                    g.draw(f"{args.output_dir}/{module_prefix}"
                           f"{unnamed_goal_number}.svg")
                else:
                    g.write_feat_json(f"{args.output_dir}/{module_prefix}"
                                      f"{lemma_name}.json")
                    g.draw(f"{args.output_dir}/{module_prefix}"
                           f"{lemma_name}.svg")

                raise
        return SubSearchResult(None, 0)
    total_nodes = numNodesInTree(args.search_width,
                                 args.search_depth + 2) - 1
    desc_name = lemma_name
    if len(desc_name) > 25:
        desc_name = desc_name[:22] + "..."
    with TqdmSpy(total=total_nodes, unit="pred", file=sys.stdout,
                 desc=desc_name, disable=(not args.progress),
                 leave=False,
                 position=bar_idx + 1,
                 dynamic_ncols=True, bar_format=mybarfmt) as pbar:
        command_list, _ = search(pbar, [g.start_node], [], 0)
        pbar.clear()
    if module_name:
        module_prefix = escape_lemma_name(module_name)
    else:
        module_prefix = ""
    if lemma_name == "":
        unnamed_goal_number += 1
        g.draw(f"{args.output_dir}/{module_prefix}"
               f"{unnamed_goal_number}.svg")
    else:
        g.draw(f"{args.output_dir}/{module_prefix}{lemma_name}.svg")
        g.write_feat_json(f"{args.output_dir}/{module_prefix}"
                          f"{lemma_name}.json")
    if command_list:
        return SearchResult(SearchStatus.SUCCESS, command_list)
    elif hasUnexploredNode:
        return SearchResult(SearchStatus.INCOMPLETE, None)
    else:
        return SearchResult(SearchStatus.FAILURE, None)


def completed_proof(coq: serapi_instance.SerapiInstance) -> bool:
    if coq.proof_context:
        return len(coq.proof_context.all_goals) == 0 and \
            coq.tactic_history.curDepth() == 0
    else:
        return False


if __name__ == "__main__":
    main(sys.argv[1:])
