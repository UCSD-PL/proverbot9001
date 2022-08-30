#!/usr/bin/env python3

import argparse
import subprocess
import re
import os
import traceback
from typing import NamedTuple, Optional, Dict, List, cast, Tuple, Iterable, Iterator
from pathlib import Path

import coq_serapy
from coq_serapy.contexts import ProofContext
from models.tactic_predictor import TacticPredictor
from search_results import SearchResult, KilledException, SearchStatus, TacticInteraction
from search_strategies import best_first_proof_search, bfs_beam_proof_search, dfs_proof_search_with_graph

from util import unwrap, eprint, escape_lemma_name

unnamed_goal_number: int = 0

class ReportJob(NamedTuple):
    project_dir: str
    filename: str
    module_prefix: str
    lemma_statement: str

class Worker:
    args: argparse.Namespace
    widx: int
    predictor: TacticPredictor
    coq: Optional[coq_serapy.SerapiInstance]
    switch_dict: Optional[Dict[str, str]]

    # File-local state
    cur_project: Optional[str]
    cur_file: Optional[str]
    last_program_statement: Optional[str]
    lemmas_encountered: List[ReportJob]
    remaining_commands: List[str]
    axioms_already_added: bool

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
        self.axioms_already_added = False

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

    def set_switch_from_proj(self) -> None:
        assert self.cur_project
        assert self.coq
        try:
            with (self.args.prelude / self.cur_project / "switch.txt").open('r') as sf:
                switch = sf.read().strip()
        except FileNotFoundError:
            if self.switch_dict is not None:
                switch = self.switch_dict[self.cur_project]
            else:
                return
        self.coq.set_switch(switch)

    def restart_coq(self) -> None:
        assert self.coq
        self.coq.kill()
        self.coq = coq_serapy.SerapiInstance(['sertop', '--implicit'],
                                    None, str(self.args.prelude / self.cur_project),
                                    use_hammer=self.args.use_hammer)
        self.coq.quiet = True
        self.coq.verbose = self.args.verbose

    def reset_file_state(self) -> None:
        self.last_program_statement = None
        self.lemmas_encountered = []
        self.remaining_commands = []
        self.axioms_already_added = False

    def enter_file(self, filename: str) -> None:
        assert self.coq
        self.cur_file = filename
        module_name = coq_serapy.get_module_from_filename(filename)
        self.coq.run_stmt(f"Module {module_name}.")
        self.remaining_commands = coq_serapy.load_commands_preserve(
            self.args, 1, self.args.prelude / self.cur_project / filename)
        self.axioms_already_added = False

    def exit_cur_file(self) -> None:
        for sec_or_mod, _ in reversed(self.coq.sm_stack):
            self.coq.run_stmt(f"Reset {sec_or_mod}.")

    def run_into_job(self, job: ReportJob, restart_anomaly: bool, careful: bool) -> None:
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
                self.exit_cur_file()
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
            except coq_serapy.CoqAnomaly:
                if restart:
                    self.restart_coq()
                    self.reset_file_state()
                    self.enter_file(job_file)
                    eprint(f"Hit a coq anomaly! Restarting...",
                           guard=self.args.verbose >= 1)
                    self.run_into_job(job, False, careful)
                    return
                else:
                    assert False
            except coq_serapy.SerapiException:
                if not careful:
                    eprint(f"Hit a problem, possibly due to admitting proofs! Restarting file with --careful...",
                           guard=self.args.verbose >= 1)
                    self.reset_file_state()
                    self.exit_cur_file()
                    self.enter_file(job_file)
                    self.run_into_job(job, restart_anomaly, True)
                    return
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
                self.skip_proof(lemma_statement, careful)
                self.lemmas_encountered.append(ReportJob(self.cur_project,
                                                         unwrap(self.cur_file),
                                                         self.coq.sm_prefix,
                                                         unique_lemma_statement))

    def skip_proof(self, lemma_statement: str, careful: bool) -> None:
        assert self.coq
        ending_command = None
        for cmd in self.remaining_commands:
            if coq_serapy.ending_proof(cmd):
                ending_command = cmd
                break
        assert ending_command
        proof_relevant = ending_command.strip() == "Defined." or \
            bool(re.match(
                r"\s*Derive",
                coq_serapy.kill_comments(lemma_statement))) or \
            bool(re.match(
                r"\s*Let",
                coq_serapy.kill_comments(lemma_statement))) or \
            bool(re.match(
                r"\s*Equations",
                coq_serapy.kill_comments(lemma_statement))) or \
            careful
        if proof_relevant:
            self.remaining_commands, _ = unwrap(self.coq.finish_proof(
                self.remaining_commands)) # type: ignore
        else:
            try:
                coq_serapy.admit_proof(self.coq, lemma_statement, ending_command)
            except coq_serapy.SerapiException:
                lemma_name = \
                  coq_serapy.lemma_name_from_statement(lemma_statement)
                eprint(f"{self.cur_file}: Failed to admit proof {lemma_name}")
                raise

            while not coq_serapy.ending_proof(self.remaining_commands[0]):
                self.remaining_commands.pop(0)
            # Pop the actual Qed/Defined/Save
            self.remaining_commands.pop(0)

    def run_job(self, job: ReportJob, restart: bool = True) -> SearchResult:
        assert self.coq
        self.run_into_job(job, restart, self.args.careful)
        job_project, job_file, job_module, job_lemma = job
        initial_context: ProofContext = unwrap(self.coq.proof_context)
        if self.args.add_axioms and not self.axioms_already_added:
            self.axioms_already_added = True
            # Cancel the lemma statement so we can run the axiom
            self.coq.cancel_last()
            with self.args.add_axioms.open('r') as f:
                for signature in f:
                    try:
                        self.coq.run_stmt(signature)
                        self.coq.run_stmt("Admitted.")
                    except coq_serapy.CoqExn:
                        axiom_name = coq_serapy.lemma_name_from_statement(
                            signature)
                        eprint(f"Couldn't declare axiom {axiom_name} "
                               f"at this point in the proof")
            self.coq.run_stmt(job_lemma)
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
        except coq_serapy.CoqAnomaly:
            if self.args.hardfail:
                raise
            if self.args.log_anomalies:
                with self.args.log_anomalies.open('a') as f:
                    print(f"ANOMALY at {job_file}:{job_lemma}",
                          file=f)
                    traceback.print_exc(file=f)
            self.restart_coq()
            self.reset_file_state()
            self.enter_file(job_file)
            if restart:
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

        while not coq_serapy.ending_proof(self.remaining_commands[0]):
            self.remaining_commands.pop(0)
        # Pop the actual Qed/Defined/Save
        ending_command = self.remaining_commands.pop(0)
        coq_serapy.admit_proof(self.coq, job_lemma, ending_command)

        self.lemmas_encountered.append(job)
        return SearchResult(search_status, solution)

def get_lemma_declaration_from_name(coq: coq_serapy.SerapiInstance,
                                    lemma_name: str) -> str:
    return coq.check_term(lemma_name).replace("\n", "")

import _thread
import threading

# This method attempts to complete proofs using search.
def attempt_search(args: argparse.Namespace,
                   lemma_statement: str,
                   module_name: Optional[str],
                   coq: coq_serapy.SerapiInstance,
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

    lemma_name = coq_serapy.lemma_name_from_statement(lemma_statement)
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

def get_file_jobs(args: argparse.Namespace,
                  project: str, filename: str) -> List[ReportJob]:
    arg_proofs_names = None
    if args.proofs_file:
        with open(args.proofs_file, 'r') as f:
            arg_proofs_names = [line.strip() for line in f]
    elif args.proof:
        arg_proofs_names = [args.proof]
    cmds = coq_serapy.load_commands(args.prelude / project / filename)
    lemmas_in_file = coq_serapy.lemmas_in_file(filename, cmds,
                                                    args.include_proof_relevant)
    if arg_proofs_names:
        return [ReportJob(project, filename, module, stmt)
                for (module, stmt) in lemmas_in_file
                if coq_serapy.lemma_name_from_statement(stmt)
                in arg_proofs_names]
    else:
        return [ReportJob(project, filename, module, stmt)
                for (module, stmt) in lemmas_in_file]


def get_files_jobs(args: argparse.Namespace,
                   proj_filename_tuples: Iterable[Tuple[str, str]]) \
                   -> Iterator[ReportJob]:
    for project, filename in proj_filename_tuples:
        yield from get_file_jobs(args, project, filename)
