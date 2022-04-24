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

from search_file import (parse_arguments, get_predictor, attempt_search,
                         Job, SearchResult, KilledException)
import coq_serapy

def main(arg_list: List[str]) -> None:
    args, _, parser = parse_arguments(arg_list)
    
    if 'SLURM_ARRAY_TASK_ID' in environ:
        workerid = int(environ['SLURM_ARRAY_TASK_ID'])
    else:
        assert False, 'SLURM_ARRAY_TASK_ID must be set'

    sys.setrecursionlimit(100000)
    if util.use_cuda:
        torch.cuda.set_device(device)
    util.cuda_device = device

    workers = [multiprocessing.Process(target=search_file_worker,
                                       args=(args, workerid * args.num_threads + widx))
               for widx in range(args.num_threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

def run_worker(args: argparse.Namespace, workerid: int) -> None:
    with (args.output / "workers_scheduled.txt").open('a') as f, FileLock(f):
        print(args.workerid, file=f)
    with (args.output / "jobs.txt").open('r') as f:
        all_jobs = [json.loads(line) for line in f]

    predictor = get_predictor(parser, args)
    with Worker(args, predictor) as worker:
        while True:
            with (args.output / "taken.txt").open('r+') as f, FileLock(f):
                taken_jobs = [json.loads(line) for line in f]
            remaining_jobs = [job for job in all_jobs in job not in taken_jobs]
            if len(remainint_jobs) > 0:
                current_job = remaining_jobs[0]
                print(json.dumps(current_job), file=f)
            else:
                break
            solution = worker.run_job(current_job)
            job_file, _, _ = solution
            with (args.output /
                  (util.safe_abbrev(job_file, args.filenames) + "-proofs.txt")
                  ).open('a') as f, FileLock(f):
                print(json.dumps(job, solution.to_dict()), file=f)

class Worker:
    args: argparse.Namespace
    predictor: TacticPredictor
    coq: Optional[SerapiInstance]
    
    # File-local state
    cur_file: Optional[str]
    last_program_statement: Optional[str]
    lemmas_encountered: List[str]
    remaining_commands: List[str]

    def __init__(self, args: argparse.Namespace, predictor: TacticPredictor) -> None:
        self.args = args
        self.predictor = predictor
        self.coq = None
        self.cur_file = None
        self.last_program_statement = None
        self.lemmas_encountered = []
        self.remaining_commands = []
    def __enter__(self) -> 'Worker':
        self.coq = SerapiInstance(['sertop', '--implicit'],
                                  None, self.args.prelude,
                                  use_hammer=self.args.use_hammer)
        self.coq.quiet = True
        self.coq.verbose = self.args.verbose
        return self
    def __exit__(self) -> None:
        self.coq.kill()
        self.coq = None

    def restart_coq(self) -> None:
        assert self.coq
        self.__exit__()
        self.__enter__()

    def reset_file_state(self) -> None:
        self.cur_file = None
        self.last_program_statement = None
        self.lemmas_encountered = []
        self.remaining_commands = []

    def run_into_job(job: Job, restart: bool = True) -> None:
        assert self.coq
        job_file, job_module, job_lemma = job
        # If the job is in a different file, or earlier in this same file, load
        # the jobs file from scratch.
        if job_file != self.cur_file or job_lemma in self.lemmas_encountered:
            self.reset_file_state()
            self.cur_file = job_file
            self.remaining_commands = coq_serapy.load_commands_preserve(
                args, worker_idx + 1, args.preludde / next_file)

        # This loop has three exit cases.  Either it will hit the correct job
        # and return, hit an error or assert before getting to the correct job,
        # or get to the end of the file and raise an assert.
        while True:
            try:
                rest_commands, run_commands = coq.run_into_next_proof(
                    self.remaining_commands)
                assert rest_commands, f"Couldn't find lemma {next_lemma}"
            except coq_serapy.CoqAnomaly:
                if restart:
                    self.restart_coq()
                    self.reset_file_state()
                    self.run_into_job(job, restart=False)
                else:
                    assert False
            except coq_serapy.SerapiException:
                eprint(f"Failed getting to before: {next_lemma}")
                eprint(f"In file {next_file}")
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
                obligation_num = 0
                while coq.local_lemmas[-(obligation_num+2)] == ":":
                    obligation_num += 1
                unique_lemma_statement = \
                    last_program_statement + \
                    f" Obligation {obligation_num}."
            else:
                unique_lemma_statement = lemma_statement
            self.remaining_commands = rest_commands
            if unique_lemma_statement == next_lemma and \
              coq.sm_prefix == next_module:
                return
            else:
                self.skip_proof()

    def skip_proof() -> None:
        proof_relevant = False
        for cmd in self.remaining_commands:
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
            bool(re.match(
                r"\s*Equations",
                serapi_instance.kill_comments(lemma_statement))) or \
            args.careful
        if proof_relevant:
            self.remaining_commands, _ = coq.finish_proof(self.remaining_commands)
        else:
            try:
                coq_serapy.admit_proof(self.coq, lemma_statement)
            except serapi_instance.SerapiException:
                next_lemma_name = \
                  coq_serapy.lemma_name_from_statement(next_lemma)
                eprint(f"{next_file}: Failed to admit proof {next_lemma_name}")
                raise
                
            while not coq_serapy.ending_proof(self.remaining_commands[0]):
                self.remaining_commands.pop(0)
            # Pop the actual Qed/Defined/Save
            self.remaining_commands.pop(0)

    def run_job(job: Job, restart: bool = True) -> SearchResult:
        _, _, job_lemma = job
        self.run_into_job(job, restart=restart)
        initial_context = coq.proof_context
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
            if args.hardfail:
                raise
            if args.log_anomalies:
                with args.log_anomalies.open('a') as f:
                    print(f"ANOMALY at {next_file}:{next_lemma}",
                          file=f)
                    traceback.print_exc(file=f)
            if restart:
                self.restart_coq()
                self.reset_file_state()
                self.run_job(job, restart=False)
            else:
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

                return SearchResult(SearchStatus.SKIPPED,
                                    solution)
        except Exception:
            eprint(f"FAILED in file {next_file}, lemma {next_lemma}")
            raise
        serapi_instance.admit_proof(coq, lemma_statement)
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

        return solution
    
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
