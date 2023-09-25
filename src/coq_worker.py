#!/usr/bin/env python3

import argparse
import subprocess
import re
import os
import traceback
import json
from typing import NamedTuple, Optional, Dict, List, cast, Tuple, Iterable, Iterator, Any, TypeVar
from pathlib import Path

import coq_serapy
from coq_serapy.contexts import ProofContext
from linearize_semicolons import get_linearized
from models.tactic_predictor import TacticPredictor
from coq_serapy.contexts import (FullContext, truncate_tactic_context,
                                 Obligation, TacticContext, ProofContext)
from predict_tactic import (loadPredictorByFile,
                            loadPredictorByName)
from util import unwrap, eprint, escape_lemma_name, split_by_char_outside_matching

class ReportJob(NamedTuple):
    project_dir: str
    filename: str
    module_prefix: str
    lemma_statement: str

T = TypeVar('T', bound='Worker')

class Worker:
    args: argparse.Namespace
    coq: Optional[coq_serapy.CoqAgent]
    switch_dict: Optional[Dict[str, str]]

    # File-local state
    cur_project: Optional[str]
    cur_file: Optional[str]
    last_program_statement: Optional[str]
    lemmas_encountered: Dict[ReportJob, int]
    remaining_commands: List[str]
    original_commands: List[str]
    obligation_num: int

    def __init__(self, args: argparse.Namespace,
                 switch_dict: Optional[Dict[str, str]] = None) -> None:
        self.args = args
        self.coq = None
        self.cur_file: Optional[str] = None
        self.cur_project: Optional[str] = None
        self.last_program_statement: Optional[str] = None
        self.lemmas_encountered: Dict[ReportJob, int] = {}
        self.remaining_commands: List[str] = []
        self.obligation_num = 0
        self.switch_dict = switch_dict

    def enter_instance(self) -> None:
        if self.args.backend == 'auto':
            coq_serapy.setup_opam_env()
            version_string = subprocess.run(["sertop", "--version"],
                                            stdout=subprocess.PIPE,
                                            text=True, check=True).stdout
            version_match = re.fullmatch(r"\d+\.(\d+).*", version_string,
                                         flags=re.DOTALL)
            assert version_match, f"Can't match version string \"{version_string}\""
            minor_version = int(version_match.group(1))
            assert minor_version >= 10, \
                    "Versions of Coq before 8.10 are not supported! "\
                    f"Currently installed coq is {version_string}"
            if minor_version >= 16:
                backend = 'lsp'
            else:
                backend = 'serapi'
        else:
            backend = self.args.backend

        if backend == 'lsp':
            backend = coq_serapy.CoqLSPyInstance(
                "coq-lsp", root_dir=str(self.args.prelude),
                verbosity=self.args.verbose)
        if backend == 'serapi':
            backend = coq_serapy.CoqSeraPyInstance(
                ["sertop"], root_dir=str(self.args.prelude))
        self.coq = coq_serapy.CoqAgent(backend, str(self.args.prelude),
                                       verbosity=self.args.verbose)

    def __enter__(self: T) -> T:
        self.enter_instance()
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
        coq_serapy.set_switch(switch)

    def restart_coq(self) -> None:
        assert self.coq
        self.coq.kill()
        self.enter_instance()

    def reset_file_state(self) -> None:
        self.last_program_statement = None
        self.lemmas_encountered = {}
        self.remaining_commands = []
        self.obligation_num = 0

    def enter_file(self, filename: str) -> None:
        assert self.coq
        self.cur_file = filename
        self.coq.enter_file(filename)
        self.original_commands = get_linearized(
            self.args, ["sertop"], 1,
            str(Path(self.cur_project) / filename))
        self.remaining_commands = self.original_commands

    def exit_cur_file(self) -> None:
        assert self.coq
        self.coq.reset()

    def run_backwards_into_job(self, job: ReportJob, restart_anomaly: bool = True) -> None:
        assert self.coq
        assert not self.coq.proof_context, "Already in a proof!"
        job_project, job_file, job_module, job_lemma = job
        lemma_name = coq_serapy.lemma_name_from_statement(job_lemma)
        for i in range(len(self.coq._file_state.local_lemmas)):
            ll_sm_stack, ll_lemma_hyp, ll_is_sec_local = self.coq._file_state.local_lemmas[-1]
            ll_sm_prefix = ".".join([mod for mod, is_section in ll_sm_stack]) + "."
            ll_lemma_name = coq_serapy.get_var_term_in_hyp(ll_lemma_hyp)
            if ll_lemma_name == lemma_name and ll_sm_prefix == job_module:
                break
            self.coq._file_state.local_lemmas.pop()
        assert len(self.coq._file_state.local_lemmas) > 0, "Couldn't find lemma!"

        # Set self.remaining_commands to the commands after the lemma we're
        # cancelling into.
        all_file_commands = self.original_commands
        commands_after_lemma_start = list(all_file_commands)
        sm_stack = coq_serapy.initial_sm_stack(job_file)
        while (coq_serapy.sm_prefix_from_stack(sm_stack) != job_module or
               coq_serapy.kill_comments(commands_after_lemma_start[0]).strip() !=
               coq_serapy.kill_comments(job.lemma_statement).strip()):
            next_cmd = commands_after_lemma_start.pop(0)
            sm_stack = coq_serapy.update_sm_stack(sm_stack, next_cmd)
        self.remaining_commands = commands_after_lemma_start
        # Reset the sm stack in Coq to the one from the command we're
        # cancelling to.
        self.coq._file_state.sm_stack = sm_stack

        # Get the state number from before the lemma from our dict.
        checkjob = ReportJob(job_project, job_file, job_module, coq_serapy.kill_comments(job_lemma).strip())
        state_before_lemma = self.lemmas_encountered[checkjob]
        # Filter lemmas out of lemmas_encountered that occur after the target
        # lemma.
        self.lemmas_encountered = \
                {lemma: state
                 for lemma, state in self.lemmas_encountered.items()
                 if state <= state_before_lemma}
        try:
            # Reset to the state number before the target lemma
            self.coq.run_stmt(f"BackTo {state_before_lemma}.")
            # Finally run the lemma statement
            self.coq.run_stmt(job_lemma)
        except coq_serapy.CoqAnomaly as e:
            if restart_anomaly:
                self.restart_coq()
                self.reset_file_state()
                self.enter_file(job_file)
                eprint("Hit a coq anomaly! Restarting...",
                    guard=self.args.verbose >= 1)
                self.run_into_job(job, True, False)
                return
            eprint(e)
            assert False

    def run_into_job(self, job: ReportJob, restart_anomaly: bool, careful: bool) -> None:
        assert self.coq
        job_project, job_file, job_module, job_lemma = job
        # If we need to change projects, we'll have to reset the coq instance
        # to load new includes, and set the opam switch
        if job_project != self.cur_project:
            if self.cur_project is not None:
                self.reset_file_state()
                self.restart_coq()
            self.cur_project = job_project
            if self.args.set_switch:
                self.set_switch_from_proj()
            self.enter_file(job_file)
        # Strip comments for comparison with lemmas encountered
        checkjob = ReportJob(job_project, job_file, job_module, coq_serapy.kill_comments(job_lemma).strip())
        if checkjob in self.lemmas_encountered:
            self.run_backwards_into_job(job)
            return

        # If the job is in a different file load the jobs file from scratch.
        if job_file != self.cur_file:
            self.reset_file_state()
            if self.cur_file:
                self.exit_cur_file()
            self.enter_file(job_file)

        # This loop has three exit cases.  Either it will hit the correct job
        # and return, hit an error or assert before getting to the correct job,
        # or get to the end of the file and raise an assert.
        while True:
            try:
                assert not self.coq.proof_context, \
                    "Currently in a proof! Back up to before the current proof, "\
                    "or use coq.finish_proof(cmds) or " \
                    "admit_proof(coq, lemma_statement, ending_stmt)"
                rest_commands, run_commands, state_before_proof = \
                    self.coq.run_into_next_proof(self.remaining_commands)
                assert rest_commands, f"Couldn't find lemma {job_lemma}"
            except coq_serapy.CoqAnomaly:
                if restart_anomaly:
                    self.restart_coq()
                    self.reset_file_state()
                    self.enter_file(job_file)
                    eprint("Hit a coq anomaly! Restarting...",
                           guard=self.args.verbose >= 1)
                    self.run_into_job(job, False, careful)
                    return
                assert False
            except coq_serapy.SerapiException:
                raise
                if not careful:
                    eprint("Hit a problem, possibly due to admitting proofs! "
                           "Restarting file with --careful...",
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
                    self.obligation_num = 0
            lemma_statement = run_commands[-1]
            if re.match(r"\s*Next\s+Obligation\s*\.\s*",
                        coq_serapy.kill_comments(
                            lemma_statement).strip()):
                assert self.last_program_statement
                unique_lemma_statement = \
                    self.last_program_statement + \
                    f" Obligation {self.obligation_num}."
                self.obligation_num += 1
            else:
                unique_lemma_statement = lemma_statement
            self.remaining_commands = rest_commands
            norm_job = ReportJob(self.cur_project,
                                 unwrap(self.cur_file),
                                 self.coq.sm_prefix,
                                 coq_serapy.kill_comments(unique_lemma_statement).strip())
            self.lemmas_encountered[norm_job] = state_before_proof
            if coq_serapy.kill_comments(unique_lemma_statement).strip() == \
               coq_serapy.kill_comments(job_lemma).strip() and \
              self.coq.sm_prefix == job_module:
                return
            self.skip_proof(careful)

    def skip_proof(self, careful: bool) -> None:
        assert self.coq
        lemma_statement = coq_serapy.kill_comments(self.coq.prev_tactics[0]).strip()
        ending_command = None
        important_vernac_cmds = []
        for cmd in self.remaining_commands:
            if re.match("\s*(?:Local\s+|Global\s+)?(?:Opaque|Transparent)(\s+[\w']+)+\.\s*", cmd):
                important_vernac_cmds.append(cmd)
            if coq_serapy.ending_proof(cmd):
                ending_command = cmd
                break
        assert ending_command
        proof_relevant = ending_command.strip() == "Defined." or \
            bool(re.match(r"\s*Derive", lemma_statement)) or \
            bool(re.match(r"\s*Let", lemma_statement)) or \
            bool(re.match(r"\s*Equations", lemma_statement)) or \
            bool(re.match(r"\s*Next\s+Obligation", lemma_statement)) or \
            bool(re.match(r".*\s+with\s+.*", lemma_statement, flags=re.DOTALL)) or \
            careful
        if proof_relevant:
            while len(self.coq.prev_tactics) > 1:
                self.coq.cancel_last()
            self.remaining_commands, _ = unwrap(self.coq.finish_proof(
               self.remaining_commands)) # type: ignore
        else:
            lemma_name = \
                coq_serapy.lemma_name_from_statement(lemma_statement)
            try:
                starting_command = coq_serapy.kill_comments(self.remaining_commands[0]).strip()
                if starting_command.startswith("Proof"):
                    self.coq.run_stmt(starting_command)
                for cmd in important_vernac_cmds:
                    self.coq.run_stmt(cmd)
                if not coq_serapy.ending_proof(starting_command):
                    coq_serapy.admit_proof(self.coq, lemma_statement, ending_command)
            except coq_serapy.SerapiException:
                eprint(f"{self.cur_file}: Failed to admit proof {lemma_name}")
                raise

            while not coq_serapy.ending_proof(self.remaining_commands[0]):
                self.remaining_commands.pop(0)
            # Pop the actual Qed/Defined/Save
            self.remaining_commands.pop(0)

def completed_proof(coq: coq_serapy.SerapiInstance) -> bool:
    if coq.proof_context:
        return len(coq.proof_context.all_goals) == 0 and \
            coq.tactic_history.curDepth() == 0
    return False

def get_all_jobs(args: argparse.Namespace, partition: str = "test_files") -> List[ReportJob]:
    project_dicts = project_dicts_from_args(args)
    proj_filename_tuples = [(project_dict["project_name"], filename)
                            for project_dict in project_dicts
                            for filename in project_dict[partition]]
    jobs = list(get_files_jobs(args, tqdm(proj_filename_tuples, desc="Getting jobs")))
    if args.proofs_file is not None:
        found_job_lines = [sm_prefix + coq_serapy.lemma_name_from_statement(stmt)
                           for project, filename, sm_prefix, stmt, done_stmts in jobs]
        with open(args.proofs_file, 'r') as f:
            jobs_lines = list(f)
        for job_line in jobs_lines:
            assert in_qualified_proofs_list(job_line.strip(), found_job_lines), \
                f"Couldn't find job {job_line.strip()}, found jobs {found_job_lines}"
        assert len(jobs) == len(jobs_lines), \
            f"There are {len(jobs_lines)} lines in the jobs file but only {len(jobs)} found jobs!"
    elif args.proof:
        assert len(jobs) == 1
    return jobs

def get_predictor(args: argparse.Namespace, allow_static_predictor: bool = True) -> TacticPredictor:
    predictor: TacticPredictor
    if args.weightsfile:
        predictor = loadPredictorByFile(args.weightsfile)
    elif allow_static_predictor and args.predictor:
        predictor = loadPredictorByName(args.predictor)
    else:
        raise ValueError("Can't load a predictor from given args!")
    return predictor

