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

from util import unwrap, eprint, escape_lemma_name, split_by_char_outside_matching

from coq_worker import ReportJob, Worker, get_predictor

unnamed_goal_number: int = 0

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
        context_lemmas = context_lemmas_from_args(self.args, self.coq)
        try:
            search_status, _, tactic_solution, steps_taken = \
              attempt_search(self.args, job_lemma,
                             self.coq.sm_prefix,
                             context_lemmas,
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

        while not coq_serapy.ending_proof(self.remaining_commands[0]):
            self.remaining_commands.pop(0)
        # Pop the actual Qed/Defined/Save
        ending_command = self.remaining_commands.pop(0)
        coq_serapy.admit_proof(self.coq, job_lemma, ending_command)

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
    global unnamed_goal_number
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
    for proof_ident in proofs_list:
        if (module + coq_serapy.lemma_name_from_statement(stmt)).endswith("." + proof_ident):
            return True
    return False

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
                if in_proofs_list(module, stmt, arg_proofs_names)]
    return [ReportJob(project, filename, module, stmt)
            for (module, stmt) in lemmas_in_file]


def get_files_jobs(args: argparse.Namespace,
                   proj_filename_tuples: Iterable[Tuple[str, str]]) \
                   -> Iterator[ReportJob]:
    for project, filename in proj_filename_tuples:
        yield from get_file_jobs(args, project, filename)

def project_dicts_from_args(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.splits_file:
        with Path(args.splits_file).open('r') as f:
            project_dicts = json.loads(f.read())
    else:
        project_dicts = [{"project_name": ".",
                          "test_files": [str(filename) for filename in args.filenames]}]
    return project_dicts
