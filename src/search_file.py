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
from typing import (List, Tuple, NamedTuple, Optional, Dict,
                    Union, Callable, cast,
                    Any)

from models.tactic_predictor import TacticPredictor, Prediction
from predict_tactic import (static_predictors, loadPredictorByFile,
                            loadPredictorByName)
import serapi_instance
from serapi_instance import (ProofContext, Obligation, SerapiInstance)

import data
from format import TacticContext, ScrapedTactic
from util import (unwrap, eprint, escape_filename, escape_lemma_name,
                  mybarfmt, split_by_char_outside_matching, nostderr)
import search_report
import util
from dataclasses import dataclass

from tqdm import tqdm
from yattag import Doc
from pathlib_revised import Path2
from enum import Enum
import pygraphviz as pgv
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

predictor: TacticPredictor
unnamed_goal_number: int


def main(arg_list: List[str], bar_idx: int) -> None:
    sys.setrecursionlimit(100000)
    global predictor

    args, parser = parse_arguments(arg_list)
    util.use_cuda = False
    predictor = get_predictor(parser, args)
    base = Path2(os.path.dirname(os.path.abspath(__file__)))

    if not args.output_dir.exists():
        args.output_dir.makedirs()

    for filename in [details_css, details_javascript]:
        destpath = args.output_dir / filename
        if not destpath.exists():
            srcpath = base.parent / 'reports' / filename
            srcpath.copyfile(destpath)

    search_file_multithreaded(args, predictor)


def parse_arguments(args_list: List[str]) -> Tuple[argparse.Namespace,
                                                   argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Produce an html report from attempting "
        "to complete proofs using Proverbot9001.")
    parser.add_argument("--prelude", default=".", type=Path2)
    parser.add_argument("--output", "-o", dest="output_dir",
                        help="output data folder name",
                        default="search-report",
                        type=Path2)
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
    parser.add_argument('--weightsfile', default=None, type=Path2)
    parser.add_argument('--predictor', choices=list(static_predictors.keys()),
                        default=None)
    parser.add_argument("--no-truncate_semicolons", dest="truncate_semicolons",
                        action='store_false')
    parser.add_argument("--search-width", dest="search_width", type=int,
                        default=5)
    parser.add_argument("--max-attempts", dest="max_attempts", type=int,
                        default=10)
    parser.add_argument("--search-depth", dest="search_depth", type=int,
                        default=6)
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
    parser.add_argument("--proof-times", default=None, type=Path2)
    parser.add_argument('filenames', help="proof file name (*.v)",
                        nargs='+', type=Path2)
    parser.add_argument("--use-hammer",
                        help="Use Hammer tactic after every predicted tactic",
                        action='store_const', const=True, default=False)
    parser.add_argument('--no-check-consistent', action='store_false',
                        dest='check_consistent')
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
    parser.add_argument("--proof", default=None)
    parser.add_argument("--log-anomalies", type=Path2, default=None)
    parser.add_argument("--log-hard-anomalies", type=Path2, default=None)
    parser.add_argument("-j", "--num-threads", type=int, default=5)
    if __name__ == "__main__":
        known_args = parser.parse_args(args_list)
    else:
        known_args, unknown_args = parser.parse_known_args(args_list)
    return known_args, parser


def produce_index(args: argparse.Namespace, predictor: TacticPredictor,
                  report_stats: List[search_report.ReportStats]) -> None:
    predictorOptions = predictor.getOptions()
    commit, date = get_metadata()
    search_report.write_summary(args,
                                predictorOptions +
                                [("report type", "search"),
                                 ("search width", args.search_width),
                                 ("search depth", args.search_depth)],
                                predictor.unparsed_args,
                                commit, date, report_stats)


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
    pass


def get_metadata() -> Tuple[str, datetime.datetime]:
    cur_commit = subprocess.check_output(["git show --oneline | head -n 1"],
                                         shell=True).decode('utf-8').strip()
    cur_date = datetime.datetime.now()
    return cur_commit, cur_date


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


def admit_proof_cmds(lemma_statement: str) -> List[str]:
    let_match = re.match(r"\s*Let\s*(.*)\.$",
                         lemma_statement,
                         flags=re.DOTALL)
    if let_match and ":=" not in lemma_statement:
        split = split_by_char_outside_matching(r"\(", r"\)", ":=",
                                               let_match.group(1))
        assert not split
        name_and_type = let_match.group(1)
        admitted_defn = f"Hypothesis {name_and_type}."
        return ["Abort.", admitted_defn]
    else:
        return ["Admitted."]


def admit_proof(coq: serapi_instance.SerapiInstance,
                lemma_statement: str) -> List[str]:
    admit_cmds = admit_proof_cmds(lemma_statement)
    for cmd in admit_cmds:
        coq.run_stmt(cmd)
    return admit_cmds


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
                       worker_idx: int) -> None:
    util.use_cuda = False

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
                lemma_statement = run_commands[-1]
                if lemma_statement == next_lemma:
                    initial_context = coq.proof_context
                    empty_context = ProofContext([], [], [], [])
                    try:
                        search_status, tactic_solution = \
                            attempt_search(args, lemma_statement,
                                           coq.module_prefix,
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
                        if failing_lemma == lemma_statement:
                            eprint("Hit the same anomaly twice! Skipping")
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
                            failing_lemma = lemma_statement
                        break
                    except Exception:
                        eprint(f"FAILED in file {next_file}, lemma {next_lemma}")
                        raise
                    admit_proof(coq, lemma_statement)
                    if not tactic_solution:
                        solution = [
                            TacticInteraction("Proof.", initial_context),
                            TacticInteraction("Admitted.", initial_context)]
                    else:
                        solution = (
                            [TacticInteraction("Proof.", initial_context)]
                            + tactic_solution +
                            [TacticInteraction("Qed.", empty_context)])
                    while not serapi_instance.ending_proof(rest_commands[0]):
                        rest_commands = rest_commands[1:]
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
                            serapi_instance.kill_comments(lemma_statement))) or \
                        bool(re.match(
                            r"\s*Let",
                            serapi_instance.kill_comments(lemma_statement))) or \
                        args.careful
                    if proof_relevant:
                        rest_commands, run_commands = coq.finish_proof(
                            rest_commands)
                    else:
                        admit_proof(coq, lemma_statement)
                        while not serapi_instance.ending_proof(
                                rest_commands[0]):
                            rest_commands = rest_commands[1:]
                        rest_commands = rest_commands[1:]

                pass

    pass


def lemmas_in_file(filename: str, cmds: List[str]) \
        -> List[Tuple[str, str]]:
    lemmas = []
    proof_relevant = False
    in_proof = False
    for cmd_idx, cmd in reversed(list(enumerate(cmds))):
        if in_proof and serapi_instance.possibly_starting_proof(cmd):
            in_proof = False
            if not proof_relevant and not re.match(r"\s*Derive",  cmd):
                lemmas.append((cmd_idx, cmd))
        if serapi_instance.ending_proof(cmd):
            in_proof = True
            if cmd.strip() == "Defined.":
                proof_relevant = True
            else:
                proof_relevant = False
    module_stack = [serapi_instance.get_module_from_filename(filename)]
    section_stack: List[str] = []
    full_lemmas = []
    for cmd_idx, cmd in enumerate(cmds):
        # Module stuff
        stripped_cmd = serapi_instance.kill_comments(
            cmd).strip()
        module_start_match = re.match(
                      r"Module\s+(?:(?:Import|Export)\s+)?(?:Type\s+)?([\w']*)",
                      stripped_cmd)
        if stripped_cmd.count(":=") > stripped_cmd.count("with"):
            module_start_match = None
        section_start_match = re.match(r"Section\s+([\w']*)(?!.*:=)",
                                       stripped_cmd)
        end_match = re.match(r"End ([\w']*)\.", stripped_cmd)
        if module_start_match:
            module_stack.append(module_start_match.group(1))
        elif section_start_match:
            section_stack.append(section_start_match.group(1))
        elif end_match:
            if module_stack and \
               module_stack[-1] == end_match.group(1):
                module_stack.pop()
            elif section_stack and section_stack[-1] == end_match.group(1):
                section_stack.pop()
            else:
                assert False, \
                    f"Unrecognized End \"{cmd}\", " \
                    f"top of module stack is {module_stack[-1]}"
            # Done
        if (cmd_idx, cmd) in lemmas:
            full_lemmas.append(("".join([module + "."
                                         for module
                                         in module_stack]),
                                cmd))
    return full_lemmas


def recover_sol(sol: Dict[str, Any]) -> SearchResult:
    return SearchResult.from_dict(sol)


def search_file_multithreaded(args: argparse.Namespace,
                              predictor: TacticPredictor) -> None:
    with multiprocessing.Manager() as manager:
        jobs: multiprocessing.Queue[
            Tuple[str, str, str]] = multiprocessing.Queue()
        done: multiprocessing.Queue[
            Tuple[Tuple[str, str, str], SearchResult]
        ] = multiprocessing.Queue()

        # This cast appears to be needed due to a buggy type stub on
        # multiprocessing.Manager()
        predictor_lock = cast(multiprocessing.managers.SyncManager,
                              manager).Lock()

        all_jobs: List[Tuple[str, str, str]] = []
        file_solutions: List[List[Tuple[Tuple[str, str, str], SearchResult]]
                             ] = [list() for _ in range(len(args.filenames))]

        for filename, solutions in zip(args.filenames, file_solutions):
            cmds = serapi_instance.load_commands_preserve(
                args, 0, args.prelude / filename)
            proofs_file = (args.output_dir / (safe_abbrev(filename, args.filenames) + "-proofs.txt"))
            all_lemma_statements = lemmas_in_file(filename, cmds)
            lemma_statements_todo = list(all_lemma_statements)

            if args.resume:
                try:
                    with proofs_file.open('r') as f:
                        for line in f:
                            (filename, module_prefix, done_lemma_stmt), sol = \
                                json.loads(line)
                            try:
                                lemma_statements_todo.remove((module_prefix,
                                                              done_lemma_stmt))
                            except ValueError:
                                eprint(f"module_prefix: {module_prefix}, "
                                       f"done_lemma_stmt: {done_lemma_stmt}")
                                raise
                            solutions.append(((filename, module_prefix,
                                               done_lemma_stmt),
                                              recover_sol(sol)))
                        eprint(f"Resumed from {str(proofs_file)}")
                        pass
                except FileNotFoundError:
                    pass
            for module_prefix, lemma_statement in lemma_statements_todo:
                if not args.proof or \
                   args.proof == serapi_instance.lemma_name_from_statement(
                       lemma_statement):
                    all_jobs.append((str(filename), module_prefix,
                                     lemma_statement))
        for job in all_jobs:
            jobs.put(job)
        workers = [multiprocessing.Process(target=search_file_worker,
                                           args=(args, predictor,
                                                 predictor_lock,
                                                 jobs, done, widx))
                   for widx in range(min(args.num_threads,
                                         len(all_jobs)))]
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
                               (safe_abbrev(Path2(done_file), args.filenames) + "-proofs.txt"))
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
        for filename, solutions in zip(args.filenames, file_solutions):
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
        src_filename: Path2,
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

        module_stack = [serapi_instance.get_module_from_filename(src_filename)]
        section_stack: List[str] = []

        tactics_interactions_batch: List[TacticInteraction] = []
        vernac_cmds_batch: List[str] = []

        in_proof = False
        for interaction in interactions:

            if in_proof and isinstance(interaction, str):
                module_prefix = "".join([module + "." for module
                                         in module_stack])
                result = lookup(module_prefix, cur_lemma_stmt)
                batch_without_brackets = [t for t in tactics_interactions_batch
                                          if t.tactic.strip() != "{" and
                                          t.tactic.strip() != "}"]
                if result is None:
                    yield ProofBlock(cur_lemma_stmt, module_prefix,
                                     SearchStatus.SKIPPED, [],
                                     batch_without_brackets)
                else:
                    yield ProofBlock(cur_lemma_stmt, module_prefix,
                                     result.status, result.commands,
                                     batch_without_brackets)
                tactics_interactions_batch = []
                in_proof = False
            elif in_proof and isinstance(interaction, ScrapedTactic):
                tactics_interactions_batch.append(
                    interaction_from_scraped(interaction))
            elif isinstance(interaction, ScrapedTactic):
                assert not in_proof and isinstance(interaction, ScrapedTactic)
                cur_lemma_stmt = vernac_cmds_batch[-1]
                yield VernacBlock(vernac_cmds_batch[:-1])
                vernac_cmds_batch = []
                tactics_interactions_batch = []
                tactics_interactions_batch.append(
                    interaction_from_scraped(interaction))
                in_proof = True
            if isinstance(interaction, str):
                # Module stuff
                stripped_cmd = serapi_instance.kill_comments(
                    interaction).strip()
                module_start_match = re.match(
                    r"Module\s+(?:Import\s+)?(?:Type\s+)?([\w']*)",
                    stripped_cmd)
                if stripped_cmd.count(":=") > stripped_cmd.count("with"):
                    module_start_match = None
                section_start_match = re.match(r"Section\s+([\w']*)(?!.*:=)",
                                               stripped_cmd)
                end_match = re.match(r"End ([\w']*)\.", stripped_cmd)
                if module_start_match:
                    module_stack.append(module_start_match.group(1))
                elif section_start_match:
                    section_stack.append(section_start_match.group(1))
                elif end_match:
                    if module_stack and \
                       module_stack[-1] == end_match.group(1):
                        module_stack.pop()
                    elif section_stack and \
                            section_stack[-1] == end_match.group(1):
                        section_stack.pop()
                    else:
                        assert False, \
                            f"Unrecognized End \"{interaction}\", " \
                            f"top of module stack is {module_stack[-1]}"
                vernac_cmds_batch.append(interaction)
        pass
    blocks = list(generate())
    return blocks


def interaction_from_scraped(s: ScrapedTactic) -> TacticInteraction:
    return TacticInteraction(s.tactic, s.context)


def write_solution_vfile(args: argparse.Namespace, filename: Path2,
                         model_name: str,
                         doc_blocks: List[DocumentBlock]):
    with (args.output_dir / (safe_abbrev(filename) + "-solution.v")
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
               output_dir: str, filename: Path2,
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

def safe_abbrev(filename: Path2, all_files: Path2) -> str:
    if filename.stem in [f.stem for f in all_files if f != filename]:
        return escape_filename(str(filename))
    else:
        return filename.stem



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
    result = dfs_proof_search_with_graph(lemma_statement, module_name, coq,
                                         args, bar_idx, predictor,
                                         predictor_lock)
    return result


@dataclass(init=True)
class LabeledNode:
    prediction: str
    certainty: float
    time_taken: Optional[float]
    node_id: int
    context_before: ProofContext
    previous: Optional["LabeledNode"]


class SearchGraph:
    __graph: pgv.AGraph
    __next_node_id: int
    start_node: LabeledNode

    def __init__(self, lemma_name: str) -> None:
        self.__graph = pgv.AGraph(directed=True)
        self.__next_node_id = 0
        self.start_node = self.mkNode(Prediction(lemma_name, 1.0),
                                      ProofContext([], [], [], []),
                                      None)
        self.start_node.time_taken = 0.0
        pass

    def addPredictions(self, src: LabeledNode, context_before: ProofContext,
                       predictions: List[Prediction]) -> List[LabeledNode]:
        return [self.mkNode(pred, context_before, src) for pred in predictions]

    def mkNode(self, prediction: Prediction, context_before: ProofContext,
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
                              context_before, previous_node)
        if previous_node:
            self.__graph.add_edge(previous_node.node_id,
                                  newNode.node_id, **kwargs)
        return newNode

    def mkQED(self, predictionNode: LabeledNode):
        self.mkNode(Prediction("QED", 1.0), ProofContext([], [], [], []),
                    predictionNode,
                    fillcolor="green", style="filled")
        cur_node = predictionNode
        cur_path = []
        while cur_node != self.start_node:
            self.setNodeColor(cur_node, "palegreen1")
            cur_path.append(cur_node)
            assert cur_node.previous
            cur_node = cur_node.previous
        return [TacticInteraction(n.prediction, n.context_before)
                for n in reversed(cur_path)]
        pass

    def setNodeColor(self, node: LabeledNode, color: str) -> None:
        node_handle = self.__graph.get_node(node.node_id)
        node_handle.attr["fillcolor"] = color
        node_handle.attr["style"] = "filled"

    def draw(self, filename: str) -> None:
        with nostderr():
            self.__graph.draw(filename, prog="dot")


class SubSearchResult (NamedTuple):
    solution: Optional[List[TacticInteraction]]
    solved_subgoals: int


def contextInPath(full_context: ProofContext, path: List[LabeledNode]):
    return any([serapi_instance.contextSurjective(full_context,
                                                  n.context_before)
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
    time_per_command = 60 if coq.use_hammer else args.max_tactic_time
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
        relevant_lemmas = coq.local_lemmas[:-1]
    elif args.relevant_lemmas == "hammer":
        relevant_lemmas = coq.get_hammer_premises()
    elif args.relevant_lemmas == "searchabout":
        relevant_lemmas = coq.get_lemmas_about_head()
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
        tactic_context_before = TacticContext(relevant_lemmas,
                                              coq.prev_tactics,
                                              coq.hypotheses,
                                              coq.goals)
        with predictor_lock:
            predictions = predictor.predictKTactics(tactic_context_before,
                                                    args.max_attempts)
            assert len(predictions) == args.max_attempts
        proof_context_before = coq.proof_context
        if coq.use_hammer:
            predictions = [Prediction(prediction.prediction + "; try hammer.",
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
                                                  unwrap(proof_context_before),
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
                                          unwrap(proof_context_before),
                                          current_path[-1])
                predictionNode.time_taken = time_taken
                if unshelved:
                    predictionNode = g.mkNode(Prediction("Unshelve.", 1.0),
                                              unwrap(proof_context_before),
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
                                          unwrap(proof_context_before),
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
        g.draw(f"{args.output_dir}/{module_prefix}{lemma_name}"
               f"{unnamed_goal_number}.svg")
    else:
        g.draw(f"{args.output_dir}/{module_prefix}{lemma_name}.svg")
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
    multiprocessing.set_start_method('spawn')
    main(sys.argv[1:], 0)
