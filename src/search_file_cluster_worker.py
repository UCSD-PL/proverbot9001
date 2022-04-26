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
#    Copyright 2022 Alex Sanchez-Stern
#
##########################################################################

import fcntl
import time
import util
import traceback
import argparse
import json
import sys
import multiprocessing
import re
from os import environ
from typing import List, Optional

from pathlib_revised import Path2
import torch

from search_file import (add_args_to_parser, get_predictor, attempt_search,
                         Job, SearchResult, KilledException, TacticInteraction,
                         SearchStatus)
import coq_serapy
from coq_serapy.contexts import ProofContext
from models.tactic_predictor import TacticPredictor
from util import eprint, unwrap

def main(arg_list: List[str]) -> None:
    multiprocessing.set_start_method('spawn')
    arg_parser = argparse.ArgumentParser()

    add_args_to_parser(arg_parser)
    arg_parser.add_argument("--num-workers", default=32, type=int)
    arg_parser.add_argument("--workers-output-dir", default=Path2("output"),
                            type=Path2)
    arg_parser.add_argument("--worker-timeout", default="6:00:00")
    arg_parser.add_argument("-p", "--partition", default="defq")
    args = arg_parser.parse_args(arg_list)
    
    if 'SLURM_ARRAY_TASK_ID' in environ:
        workerid = int(environ['SLURM_ARRAY_TASK_ID'])
    else:
        assert False, 'SLURM_ARRAY_TASK_ID must be set'

    sys.setrecursionlimit(100000)
    if util.use_cuda:
        torch.cuda.set_device("cuda:0")
        util.cuda_device = "cuda:0"

    predictor = get_predictor(arg_parser, args)
    with (args.output_dir / "workers_scheduled.txt").open('a') as f, FileLock(f):
        print(workerid, file=f)
    workers = [multiprocessing.Process(target=run_worker,
                                       args=(args, widx,
                                             predictor))
               for widx in range(args.num_threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    eprint(f"Finished worker {workerid}")

def run_worker(args: argparse.Namespace, workerid: int,
               predictor: TacticPredictor) -> None:
    with (args.output_dir / "jobs.txt").open('r') as f:
        all_jobs = [json.loads(line) for line in f]

    with Worker(args, predictor) as worker:
        while True:
            eprint("Locking taken file")
            with (args.output_dir / "taken.txt").open('r+') as f, FileLock(f):
                taken_jobs = [json.loads(line) for line in f]
                eprint(f"Found {len(taken_jobs)} taken jobs")
                remaining_jobs = [job for job in all_jobs if job not in taken_jobs]
                eprint(f"Found {len(remaining_jobs)} remaining jobs")
                if len(remaining_jobs) > 0:
                    current_job = remaining_jobs[0]
                    print(json.dumps(current_job), file=f)
                    eprint(f"Starting job {current_job}")
                else:
                    break
            solution = worker.run_job(current_job)
            job_file, _, _ = current_job
            with (args.output_dir /
                  (util.safe_abbrev(Path2(job_file), args.filenames) + "-proofs.txt")
                  ).open('a') as f, FileLock(f):
                eprint(f"Finished job {current_job}")
                print(json.dumps((current_job, solution.to_dict())), file=f)

class Worker:
    args: argparse.Namespace
    predictor: TacticPredictor
    coq: Optional[coq_serapy.SerapiInstance]
    
    # File-local state
    cur_file: Optional[str]
    last_program_statement: Optional[str]
    lemmas_encountered: List[str]
    remaining_commands: List[str]

    def __init__(self, args: argparse.Namespace, predictor: TacticPredictor) -> None:
        self.args = args
        self.predictor = predictor
        self.coq = None
        self.cur_file: Optional[str] = None
        self.last_program_statement: Optional[str] = None
        self.lemmas_encountered: List[str] = []
        self.remaining_commands: List[str] = []
    def __enter__(self) -> 'Worker':
        self.coq = coq_serapy.SerapiInstance(['sertop', '--implicit'],
                                  None, str(self.args.prelude),
                                  use_hammer=self.args.use_hammer)
        self.coq.quiet = True
        self.coq.verbose = self.args.verbose
        return self
    def __exit__(self, type, value, traceback) -> None:
        assert self.coq
        self.coq.kill()
        self.coq = None

    def restart_coq(self) -> None:
        assert self.coq
        self.coq.kill()
        self.coq = coq_serapy.SerapiInstance(['sertop', '--implicit'],
                                  None, self.args.prelude,
                                  use_hammer=self.args.use_hammer)

    def reset_file_state(self) -> None:
        self.cur_file = None
        self.last_program_statement = None
        self.lemmas_encountered = []
        self.remaining_commands = []

    def run_into_job(self, job: Job, restart: bool = True) -> None:
        assert self.coq
        job_file, job_module, job_lemma = job
        # If the job is in a different file, or earlier in this same file, load
        # the jobs file from scratch.
        if job_file != self.cur_file or job_lemma in self.lemmas_encountered:
            if self.cur_file:
                old_module_name = coq_serapy.get_module_from_filename(self.cur_file)
                self.coq.run_stmt(f"End {old_module_name}.")
            module_name = coq_serapy.get_module_from_filename(job_file)
            self.coq.run_stmt(f"Module {module_name}.")
            self.reset_file_state()
            self.cur_file = job_file
            self.remaining_commands = coq_serapy.load_commands_preserve(
                self.args, 0, self.args.prelude / job_file)

        # This loop has three exit cases.  Either it will hit the correct job
        # and return, hit an error or assert before getting to the correct job,
        # or get to the end of the file and raise an assert.
        while True:
            try:
                rest_commands, run_commands = unwrap(self.coq.run_into_next_proof(
                    self.remaining_commands))
                assert rest_commands, f"Couldn't find lemma {job_lemma}"
            except coq_serapy.CoqAnomaly:
                if restart:
                    self.restart_coq()
                    self.reset_file_state()
                    self.run_into_job(job, restart=False)
                else:
                    assert False
            except coq_serapy.SerapiException:
                eprint(f"Failed getting to before: {job_lemma}")
                eprint(f"In file {job_file}")
                raise
            for command in run_commands:
                if re.match("\s*Program\s+.*",
                            coq_serapy.kill_comments(
                                command).strip()):
                    self.last_program_statement = command
            lemma_statement = run_commands[-1]
            if re.match(r"\s*Next\s+Obligation\s*\.\s*",
                        coq_serapy.kill_comments(
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

    def skip_proof(self, lemma_statement: str) -> None:
        assert self.coq
        proof_relevant = False
        for cmd in self.remaining_commands:
            if coq_serapy.ending_proof(cmd):
                if cmd.strip() == "Defined.":
                    proof_relevant = True
                break
        proof_relevant = proof_relevant or \
            bool(re.match(
                r"\s*Derive",
                coq_serapy.kill_comments(lemma_statement))) or \
            bool(re.match(
                r"\s*Let",
                coq_serapy.kill_comments(lemma_statement))) or \
            bool(re.match(
                r"\s*Equations",
                coq_serapy.kill_comments(lemma_statement))) or \
            self.args.careful
        if proof_relevant:
            self.remaining_commands, _ = unwrap(self.coq.finish_proof(
                self.remaining_commands))
        else:
            try:
                coq_serapy.admit_proof(self.coq, lemma_statement)
            except coq_serapy.SerapiException:
                lemma_name = \
                  coq_serapy.lemma_name_from_statement(lemma_statement)
                eprint(f"{self.cur_file}: Failed to admit proof {lemma_name}")
                raise
                
            while not coq_serapy.ending_proof(self.remaining_commands[0]):
                self.remaining_commands.pop(0)
            # Pop the actual Qed/Defined/Save
            self.remaining_commands.pop(0)

    def run_job(self, job: Job, restart: bool = True) -> SearchResult:
        assert self.coq
        job_file, job_module, job_lemma = job
        self.run_into_job(job, restart=restart)
        initial_context = unwrap(self.coq.proof_context)
        empty_context = ProofContext([], [], [], [])
        try:
            search_status, tactic_solution = \
              attempt_search(self.args, job_lemma,
                             self.coq.sm_prefix,
                             self.coq, 0, self.predictor)
        except KilledException:
            solution = [
                TacticInteraction("Proof.", initial_context),
                TacticInteraction("Admitted.", initial_context)
                ]

            return SearchResult(SearchStatus.INCOMPLETE,
                                solution)
        except coq_serapy.CoqAnomaly:
            if self.args.hardfail:
                raise
            if self.args.log_anomalies:
                with self.args.log_anomalies.open('a') as f:
                    print(f"ANOMALY at {job_file}:{job_lemma}",
                          file=f)
                    traceback.print_exc(file=f)
            if restart:
                self.restart_coq()
                self.reset_file_state()
                return self.run_job(job, restart=False)
            else:
                if self.args.log_hard_anomalies:
                    with self.args.log_hard_anomalies.open('a') as f:
                        print(
                            f"HARD ANOMALY at "
                            f"{job_file}:{job_lemma}",
                            file=f)
                        traceback.print_exc(file=f)
                solution = [
                    TacticInteraction("Proof.", initial_context),
                    TacticInteraction("Admitted.", initial_context)
                    ]

                return SearchResult(SearchStatus.SKIPPED,
                                    solution)
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

        coq_serapy.admit_proof(self.coq, job_lemma)
        while not coq_serapy.ending_proof(self.remaining_commands[0]):
            self.remaining_commands.pop(0)
        # Pop the actual Qed/Defined/Save
        self.remaining_commands.pop(0)

        return SearchResult(search_status, solution)
    
class FileLock:
    def __init__(self, file_handle):
        self.file_handle = file_handle

    def __enter__(self):
        while True:
            try:
                fcntl.flock(self.file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
               time.sleep(0.01)
        return self

    def __exit__(self, type, value, traceback):
        fcntl.flock(self.file_handle, fcntl.LOCK_UN)

if __name__ == "__main__":
    main(sys.argv[1:])
