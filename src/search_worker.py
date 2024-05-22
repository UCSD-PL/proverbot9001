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
from models.tactic_predictor import TacticPredictor
from search_results import SearchResult, KilledException, SearchStatus, TacticInteraction
from search_strategies import best_first_proof_search, bfs_beam_proof_search, dfs_proof_search_with_graph, dfs_estimated
from predict_tactic import (loadPredictorByFile,
                            loadPredictorByName)
from linearize_semicolons import get_linearized

from util import unwrap, eprint, escape_lemma_name, split_by_char_outside_matching, print_time

obl_num: int = 0

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
    unnamed_goal_num: int

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
        self.unnamed_goal_num = 0
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
            coq_backend = coq_serapy.CoqLSPyInstance(
                "coq-lsp", root_dir=str(self.args.prelude),
                verbosity=self.args.verbose)
        if backend == 'serapi':
            coq_backend = coq_serapy.CoqSeraPyInstance(
                ["sertop"], timeout=60)
            coq_backend.verbosity = self.args.verbose
        self.coq = coq_serapy.CoqAgent(coq_backend, str(self.args.prelude),
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
        self.unnamed_goal_num = 0

    def reset_project_state(self) -> None:
        self.reset_file_state()
        self.cur_project = None

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
        with print_time("Resetting command state"):
            try:
                self.coq.reset()
            except coq_serapy.CoqAnomaly as e:
                eprint(f"Got anomaly {e}")
                if e.msg == "Timing Out":
                    self.enter_instance()
                else:
                    raise
            except coq_serapy.CoqTimeoutError as e:
                self.restart_coq()
                self.coq.backend.enterDirectory(str(self.cur_project))

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
        last_program_statement = ""
        obl_num = 0
        while True:
            if coq_serapy.possibly_starting_proof(commands_after_lemma_start[0]):
                unique_lemma_stmt, _, self.obligation_num, self.unnamed_goal_num = \
                  unique_lemma_stmt_and_name(
                    commands_after_lemma_start[0],
                    commands_after_lemma_start[1:],
                    last_program_statement if last_program_statement != "" else None,
                    self.obligation_num, self.unnamed_goal_num)
                if (coq_serapy.sm_prefix_from_stack(sm_stack) == job_module
                    and unique_lemma_stmt.strip() ==
                    coq_serapy.kill_comments(job_lemma).strip()):
                    break

            next_cmd = commands_after_lemma_start.pop(0)
            sm_stack = coq_serapy.update_sm_stack(sm_stack, next_cmd)
            if re.match(r"\s*(?:(?:Local|Global)\s+)?Program\s+.*",
                        coq_serapy.kill_comments(
                          commands_after_lemma_start[0]).strip(),
                        re.DOTALL):
                last_program_statement = commands_after_lemma_start[0]
                obl_num = 0
        self.remaining_commands = commands_after_lemma_start
        # Reset the sm stack in Coq to the one from the command we're
        # cancelling to.
        self.coq._file_state.sm_stack = sm_stack
        self.last_program_statement = last_program_statement
        self.obligation_num = obl_num

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
            self.coq.backend.backToState(state_before_lemma)
            # Finally run the lemma statement
            self.coq.run_stmt(commands_after_lemma_start[0])
        except coq_serapy.CoqAnomaly as e:
            if restart_anomaly:
                self.restart_coq()
                self.enter_file(job_file)
                self.reset_project_state()
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
        assert job_project != None, "The job project is NONE!"
        if job_project != self.cur_project:
            first_project = self.cur_project is None
            self.cur_project = job_project
            if self.args.set_switch:
                self.set_switch_from_proj()
            self.reset_file_state()
            self.restart_coq()
            self.coq.backend.enterDirectory(str(self.cur_project))
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
                assert self.remaining_commands, f"Couldn't find lemma {job_lemma}"
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
                    self.enter_file(job_file)
                    self.reset_project_state()
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
            unique_lemma_statement, lemma_name, self.obligation_num, self.unnamed_goal_num = \
                unique_lemma_stmt_and_name(lemma_statement, rest_commands,
                                           self.last_program_statement, self.obligation_num,
                                           self.unnamed_goal_num)
            self.remaining_commands = rest_commands
            assert rest_commands is not None
            norm_job = ReportJob(self.cur_project,
                                 unwrap(self.cur_file),
                                 self.coq.sm_prefix,
                                 coq_serapy.kill_comments(unique_lemma_statement).strip())
            self.lemmas_encountered[norm_job] = state_before_proof
            if coq_serapy.kill_comments(unique_lemma_statement).strip() == \
               coq_serapy.kill_comments(job_lemma).strip() and \
              self.coq.sm_prefix == job_module:
                return
            try:
                self.skip_proof(careful)
            except coq_serapy.CoqAnomaly:
                if restart_anomaly:
                    self.restart_coq()
                    self.enter_file(job_file)
                    self.reset_project_state()
                    eprint("Hit a coq anomaly! Restarting...",
                           guard=self.args.verbose >= 1)
                    self.run_into_job(job, False, careful)
                    return
                assert False
            except coq_serapy.CoqExn:
                eprint("Got an error when trying to skip proof of "
                       f"{self.coq.sm_prefix}{lemma_name}")
                eprint("Maybe one of your 'Proof using' declarations is wrong?")
                raise

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
            while len(self.coq.tactic_history.getFullHistory()) > 1:
                self.coq.cancel_last()
            self.remaining_commands, _ = unwrap(self.coq.finish_proof(
               self.remaining_commands)) # type: ignore
        else:
            lemma_name = \
                coq_serapy.lemma_name_from_statement(lemma_statement)
            try:
                starting_command = coq_serapy.kill_comments(self.remaining_commands[0]).strip()
                if starting_command.startswith("Proof") or coq_serapy.ending_proof(starting_command):
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

class SearchWorker(Worker):
    widx: int
    predictor: TacticPredictor
    axioms_already_added: bool
    def __init__(self, args: argparse.Namespace, worker_idx: int,
                 predictor: TacticPredictor,
                 switch_dict: Optional[Dict[str, str]] = None) -> None:
        super().__init__(args, switch_dict)
        self.widx = worker_idx
        self.predictor = predictor
        self.axioms_already_added = False

    def enter_file(self, filename: str) -> None:
        super().enter_file(filename)
        self.axioms_already_added = False

    def reset_file_state(self) -> None:
        super().reset_file_state()
        self.axioms_already_added = False

    def run_job(self, job: ReportJob, restart: bool = True) -> SearchResult:
        assert self.coq
        self.run_into_job(job, restart, self.args.careful)
        job_project, job_file, job_module, job_lemma = job
        initial_context: ProofContext = unwrap(self.coq.proof_context)
        # In certain rare cases (uses of "Goal") this can be different from the job_lemma
        original_lemma_statement = self.coq.prev_tactics[-1]
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
            self.coq.run_stmt(original_lemma_statement)
        empty_context = ProofContext([], [], [], [])
        context_lemmas = context_lemmas_from_args(self.args, self.coq)
        try:
            search_status, _, tactic_solution, steps_taken = \
              attempt_search(self.args, job_lemma,
                             self.coq.sm_prefix,
                             context_lemmas,
                             self.coq,
                             self.args.output_dir / self.cur_project,
                             self.widx, self.predictor)
            while len(self.coq.tactic_history.getFullHistory()) > 1:
                self.coq.cancel_last()
            self.skip_proof(False)
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
            self.enter_file(job_file)
            self.reset_project_state()
            if restart:
                eprint("Hit an anomaly, restarting job", guard=self.args.verbose >= 2)
                return self.run_job(job, restart=False)
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
            return SearchResult(search_status, context_lemmas, solution, 0)
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

        #while not coq_serapy.ending_proof(self.remaining_commands[0]):
        #    self.remaining_commands.pop(0)
        ## Pop the actual Qed/Defined/Save
        #ending_command = self.remaining_commands.pop(0)
        return SearchResult(search_status, context_lemmas, solution, steps_taken)

def get_lemma_declaration_from_name(coq: coq_serapy.SerapiInstance,
                                    lemma_name: str) -> str:
    return coq.check_term(lemma_name).replace("\n", "")

def context_lemmas_from_args(args: argparse.Namespace,
                             coq: coq_serapy.SerapiInstance) -> List[str]:
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
        assert False, f"Unrecognized relevant lemmas option {args.relevant_lemmas}"
    return env_lemmas + relevant_lemmas

import _thread
import threading

# This method attempts to complete proofs using search.
def attempt_search(args: argparse.Namespace,
                   lemma_statement: str,
                   module_name: Optional[str],
                   context_lemmas: List[str],
                   coq: coq_serapy.SerapiInstance,
                   output_dir: Path,
                   bar_idx: int,
                   predictor: TacticPredictor) \
        -> SearchResult:
    global obl_num
    if "Proof" not in coq.prev_tactics[-1]:
        coq.run_stmt("Proof.")
    if module_name:
        module_prefix = escape_lemma_name(module_name)
    else:
        module_prefix = ""

    lemma_name = coq_serapy.lemma_name_from_statement(lemma_statement)
    if lemma_name == "":
        obl_num  += 1
        lemma_name = f"Obligation {obl_num}"

    if args.max_search_time_per_lemma:
        timer = threading.Timer(args.max_search_time_per_lemma, _thread.interrupt_main)
        timer.start()
    try:
        if args.search_type == 'dfs':
            result = dfs_proof_search_with_graph(lemma_name, module_prefix,
                                                 context_lemmas,
                                                 coq, output_dir,
                                                 args, bar_idx, predictor)
        elif args.search_type == 'dfs-est':
            result = dfs_estimated(lemma_name, module_prefix,
                                   context_lemmas,
                                   coq, output_dir,
                                   args, bar_idx, predictor)
        elif args.search_type == 'beam-bfs':
            result = bfs_beam_proof_search(lemma_name, module_prefix,
                                           context_lemmas, coq,
                                           output_dir,
                                           args, bar_idx, predictor)
        elif args.search_type == 'astar' or args.search_type == 'best-first':
            result = best_first_proof_search(lemma_name, module_prefix,
                                             context_lemmas, coq,
                                             output_dir,
                                             args, bar_idx, predictor)
        else:
            assert False, args.search_type
    except KeyboardInterrupt as exc:
        if args.max_search_time_per_lemma:
            raise KilledException("Lemma timeout") from exc
        raise
    finally:
        if args.max_search_time_per_lemma:
            timer.cancel()
    return result

def in_proofs_list(module: str, stmt: str, proofs_list: List[str]) -> bool:
    match_string = module + coq_serapy.lemma_name_from_statement(stmt)
    return in_qualified_proofs_list(match_string, proofs_list)

def in_qualified_proofs_list(job_line: str, proofs_list: List[str]) -> bool:
    for qualified_ident in proofs_list:
        if job_line.endswith("." + qualified_ident) or\
           qualified_ident == job_line:
            return True
    return False

def get_file_jobs(args: argparse.Namespace,
                  project: str, filename: str) -> List[ReportJob]:
    # eprint(f"Looking at file {filename}")
    arg_proofs_names = None
    if args.proofs_file:
        with open(args.proofs_file, 'r') as f:
            arg_proofs_names = [line.strip() for line in f]
    elif args.proof:
        arg_proofs_names = [args.proof]
    cmds = coq_serapy.load_commands(args.prelude / project / filename)
    lemmas_in_file = coq_serapy.lemmas_in_file(filename, cmds,
                                               args.include_proof_relevant,
                                               disambiguate_goal_stmts = True)
    # for (module, stmt) in lemmas_in_file:
    #     if in_proofs_list(module, stmt, arg_proofs_names):
    #         eprint(f"{(module, stmt)} found in proofs list")
    #     else:
    #         eprint(f"{(module, stmt)} not found in proofs list")
    if arg_proofs_names:
        return [ReportJob(project, filename, module, stmt)
                for (module, stmt) in lemmas_in_file
                if in_proofs_list(module, stmt, arg_proofs_names)]
    return [ReportJob(project, filename, module, stmt)
            for (module, stmt) in lemmas_in_file]


def get_files_jobs(args: argparse.Namespace,
                   proj_filename_tuples: Iterable[Tuple[str, str]]) \
                   -> Iterator[ReportJob]:
    for project, filename in proj_filename_tuples:
        yield from get_file_jobs(args, project, filename)

def get_predictor(args: argparse.Namespace, allow_static_predictor: bool = True,
                  device: Optional[str] = None) -> TacticPredictor:
    predictor: TacticPredictor
    if args.weightsfile:
        predictor = loadPredictorByFile(args.weightsfile, device)
    elif allow_static_predictor and args.predictor:
        predictor = loadPredictorByName(args.predictor, device)
    else:
        raise ValueError("Can't load a predictor from given args!")
    return predictor

def project_dicts_from_args(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.splits_file:
        with Path(args.splits_file).open('r') as f:
            project_dicts = json.loads(f.read())
    else:
        project_dicts = [{"project_name": ".",
                          "test_files": [str(filename) for filename in args.filenames]}]
    return project_dicts
def files_of_dict(args: argparse.Namespace,
                  project_dict: Dict[str, Any]) -> List[str]:
    if args.include_train_set:
        return project_dict["train_files"] + project_dict["test_files"]
    else:
        return project_dict["test_files"]

def job_summary(job: ReportJob) -> str:
    obl_match = re.match("(.*\.)\s+Obligation (\d+)\.",
                         job.lemma_statement,
                         re.DOTALL)
    if obl_match:
        lname = coq_serapy.lemma_name_from_statement(obl_match.group(1))
        return f"{job.module_prefix}{lname}, "\
               f"Obligation {obl_match.group(2)}"
    else:
        lname = coq_serapy.lemma_name_from_statement(job.lemma_statement)
        return f"{job.module_prefix}{lname}"

# This mostly replicates functionality in
# coq_serapy.coq_util.lemmas_in_file, consider merging.
def unique_lemma_stmt_and_name(orig_lemma_statement: str, rest_commands: List[str],
                               last_program_statement: Optional[str],
                               obligation_num: int,
                               unnamed_goal_num: int) -> Tuple[str, str, int, int]:
    slemma = coq_serapy.kill_comments(orig_lemma_statement).strip()
    next_obl_match = re.match(r"Next\s+Obligation\s*\.", slemma)
    goal_match = re.match(r"\s*Goal\s+(.*)\.$", slemma)
    if next_obl_match:
        assert last_program_statement
        unique_stmt = last_program_statement + f" Obligation {obligation_num}."
        unique_name = coq_serapy.lemma_name_from_statement(
            last_program_statement) + f" Obligation {obligation_num}."
        return unique_stmt, unique_name, obligation_num + 1, unnamed_goal_num
    elif goal_match:
        first_ending_command = None
        for cmd in rest_commands:
            if coq_serapy.ending_proof(cmd):
                 first_ending_command = cmd
                 break
        assert first_ending_command is not None,\
            "Couldn't find an ending command after `Goal`."
        named_ending_match = re.match(r"(?:Save|Defined)\s+(\w+)\.",
                                      first_ending_command.strip())
        if named_ending_match:
            lemma_name = named_ending_match.group(1)
            unique_stmt = \
                f"Theorem {lemma_name}: {goal_match.group(1)}."
            return unique_stmt, lemma_name, obligation_num, unnamed_goal_num
        else:
            if unnamed_goal_num == 0:
                postfix = ""
            else:
                postfix = str(unnamed_goal_num - 1)
            lemma_name = f"Unnamed_thm{postfix}"
            unique_stmt = f"Theorem {lemma_name}: {goal_match.group(1)}."
            return unique_stmt, lemma_name, obligation_num, unnamed_goal_num + 1
    else:
        return slemma, coq_serapy.lemma_name_from_statement(slemma), \
            obligation_num, unnamed_goal_num
