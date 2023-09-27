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
import multiprocessing
import functools
import sys
import contextlib
import shutil
import json
import re

import linearize_semicolons
import coq_serapy


from util import eprint, mybarfmt, unwrap

from coq_serapy import CoqAnomaly, CoqAgent
from coq_serapy.contexts import TacticContext, FullContext, ProofContext, truncate_tactic_context
from models.tactic_predictor import Prediction, TacticPredictor
from search_file import get_predictor

from typing import NamedTuple, TextIO, List, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
# from tqdm.contrib.concurrent import process_map
from worker import Worker

# class ErrorScrapeJob(NamedTuple):
#     file_idx: int
#     filename: str

# class ErrorScrapeWorker(Worker):
#     widx: int
#     predictor: TacticPredictor
#     def __init__(self, args: argparse.Namespace, worker_idx: int,
#                  predictor: TacticPredictor) -> None:
#         super().__init__(args)
#         self.widx = worker_idx
#         self.predictor = predictor

#     # maybe change this to return the error scrape results
#     def run_job(self, job: ErrorScrapeJob, restart: bool = True) -> None:
#         assert self.coq
#         self.run_into_job(job, restart, self.args.careful)
#         job_idx, job_file = job
#         try:
#             # call error scrape function here
            
#             pass
#         except coq_serapy.CoqAnomaly:
#             if self.args.hardfail:
#                 raise
#             self.restart_coq()
#             self.reset_file_state()
#             self.enter_file(job_file)
#             if restart:
#                 eprint("Hit an anomaly, restarting job", guard=self.args.verbose >= 2)
#                 return self.run_job(job, restart=False)
#         except Exception:
#             eprint(f"FAILED in file {job_file}")
#             raise
#         # Pop the actual Qed/Defined/Save
#         ending_command = self.remaining_commands.pop(0)
#     def scrape_error(self) -> None:
#         sys.setrecursionlimit(4500)
#         # run_stmts = []
#         # predictor = get_predictor(args)
#         file_idx, filename = file_tuple
#         full_filename = args.prelude + "/" + filename
#         result_file = full_filename + ".error.scrape"
#         temp_file = full_filename + ".error.scrape.partial"
#         if args.cont:
#             with contextlib.suppress(FileNotFoundError):
#                 with open(result_file, 'r') as f:
#                     if args.verbose:
#                         eprint(f"Found existing scrape at {result_file}! Using it")
#                     return result_file
#         try:
#             if args.linearize:
#                 commands = linearize_semicolons.get_linearized(args, coqargs, file_idx, filename)
#             else:
#                 commands = coq_serapy.load_commands_preserve(
#                     args, file_idx, str(full_filename))
#             if args.switch:
#                 coq_serapy.set_switch(args.switch)
#             if True:
#                 coq = ImprovedSerapiInstance(
#                     coqargs,
#                     coq_serapy.get_module_from_filename(filename),
#                     args.prelude, use_hammer=args.relevant_lemmas == "hammer")
#                 coq.verbose = args.verbose
#                 try:
#                     with open(temp_file, 'w') as f:
#                         # while True:
#                         #     commands, cmds_run = coq.run_into_next_proof(commands)
#                         #     if "Lemma invert_expr_context" in cmds_run[-1]:
#                         #         break
#                         #     commands, cmds_run = coq.finish_proof(commands)

#                         # for command in commands:
#                         for command in tqdm(commands, file=sys.stdout,
#                                             disable=(not args.progress),
#                                             position=file_idx * 2,
#                                             desc="Scraping error file", leave=False,
#                                             dynamic_ncols=True,
#                                             bar_format=mybarfmt):
#                             process_statement(args, coq, command, f, predictor)
#                     shutil.move(temp_file, result_file)
#                     return result_file
#                 except coq_serapy.TimeoutError:
#                     eprint("Command in {} timed out.".format(filename))
#                     return temp_file
#         except Exception as e:
#             eprint("FAILED: In file {}:".format(filename))
#             eprint(e)
#             if args.hardfail or len(args.inputs) == 1 or args.hardfail_scrape:
#                 raise e
#         return None


class ImprovedSerapiInstance():
    coq: Optional[coq_serapy.CoqAgent]
    coqargs: List[str]
    module: str
    prelude: str
    use_hammer: bool
    current_gallina = List[str]
    current_proof = List[str]

    def __init__(self, coqargs, module, prelude, use_hammer=False):
        self.coqargs = coqargs
        self.module = module
        self.prelude = prelude
        self.use_hammer = use_hammer
        self.coq = coq_serapy.SerapiInstance(coqargs, module, prelude, use_hammer=use_hammer)
        self.current_gallina = []
        self.current_proof = []

    def run_stmt(self, stmt, timeout=60, store=True):
        in_proof_before = self.coq.proof_context is not None
        
        try:
            self.coq.run_stmt(stmt, timeout=timeout)
        except CoqAnomaly as e:
            print(e)
            self._restart_coq()
            self.coq.run_stmt(stmt, timeout=timeout)

        in_proof_after = self.coq.proof_context is not None

        if not store:
            return
        if not in_proof_before:
            # print("Command: ", command)
            self.current_gallina.append(stmt)
        elif "Proof" in stmt:
            self.current_gallina.append(stmt)
        elif in_proof_before and in_proof_after:
            self.current_proof.append(stmt)
        elif in_proof_before and not in_proof_after:
            # print("Admitting...")
            self.current_gallina.append("Admitted.")
            self.current_proof = []

    def _restart_coq(self):
        self.coq.kill()
        self.coq = None
        self.coq = coq_serapy.SerapiInstance(self.coqargs, self.module, self.prelude, use_hammer=self.use_hammer)
        for stmt in tqdm(self.current_gallina + self.current_proof, file=sys.stdout,
                            desc="Restarting coq", leave=False,
                            dynamic_ncols=True,
                            bar_format=mybarfmt):
            self.coq.run_stmt(stmt)    
    
    


def main():
    multiprocessing.set_start_method('spawn')
    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description="scrape a proof")
    parser.add_argument('-o', '--output', help="output data file name",
                        default=None)
    parser.add_argument('-j', '--threads', default=1, type=int)
    parser.add_argument('-c', '--continue', dest='cont', default=False,
                        const=True, action='store_const')
    parser.add_argument('--hardfail', default=False, const=True,
                        action='store_const')
    parser.add_argument('--hardfail-scrape', action='store_true')
    parser.add_argument('--prelude', default=".")
    parser.add_argument('--weightsfile', default=None, type=Path)
    parser.add_argument("--max-term-length", type=int, default=256)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument("--progress", "-P", help="show progress of files",
                        action='store_const', const=True, default=False)
    parser.add_argument('--skip-nochange-tac', default=False, const=True,
                        action='store_const',
                        dest='skip_nochange_tac')
    parser.add_argument("--relevant-lemmas", dest="relevant_lemmas",
                        default='local',
                        choices=['local', 'hammer', 'searchabout'])
    parser.add_argument("--no-linearize", dest="linearize",
                        action='store_false')
    parser.add_argument("--ignore-lin-hash", action='store_true')
    parser.add_argument("--linearizer-timeout", type=int,
                        default=(60 * 60))
    parser.add_argument("-s", "--switch", default=None, type=str)
    parser.add_argument('inputs', nargs="+", help="proof file name(s) (*.v)")
    args = parser.parse_args()

    # Set up the command which runs sertop.
    coqargs = ["sertop", "--implicit"]
    tasks = [(idx % args.threads, filename) for (idx, filename)
             in enumerate(args.inputs)]
    # worker_idx = 0
    # predictor = get_predictor(args)

    # with ErrorScrapeWorker(args, worker_idx, predictor) as worker:
    #     while tasks:
    #         next_task = tasks.pop(0)
    #     worker.run_job(next_task, restart=True)

    # print(tasks)
    # exit()
    for idx,task in enumerate(tasks):
        scrape_result_file = scrape_file(coqargs, args, task)
        with (open(args.output, 'w') if args.output
              else contextlib.nullcontext(sys.stdout)) as out:
            if scrape_result_file is None:
                eprint("Failed file {} of {}"
                       .format(idx, len(args.inputs)))
            with open(scrape_result_file, 'r') as f:
                for line in f:
                    out.write(line)
    
    # scrape_result_files = process_map(functools.partial(scrape_file, coqargs, args), tasks, max_workers=args.threads)
    # with (open(args.output, 'w') if args.output
    #         else contextlib.nullcontext(sys.stdout)) as out:
    #     with multiprocessing.Pool(args.threads) as pool:
    #         with tqdm(total=len(tasks), bar_format=mybarfmt) as pbar:
    #             for scrape_result_file in pool.imap_unordered(
    #             functools.partial(scrape_file, coqargs, args),
    #             tasks):
    #                 pbar.update()
    #                 # with (open(args.output, 'w') if args.output
    #                 #         else contextlib.nullcontext(sys.stdout)) as out:
    #                 # for idx, scrape_result_file in enumerate(scrape_result_files,
    #                 #                                             start=1):
    #                 if scrape_result_file is None:
    #                     eprint("Failed file {}".format(scrape_result_file))
    #                     # eprint("Failed file {} of {}"
    #                     #         .format(idx, len(args.inputs)))
    #                 else:
    #                     if args.verbose:
    #                         eprint("Finished file {}".format(scrape_result_file))
    #                         # eprint("Finished file {} of {}"
    #                         #         .format(idx, len(args.inputs)))
    #                     with open(scrape_result_file, 'r') as f:
    #                         for line in f:
    #                             out.write(line)


def scrape_file(coqargs: List[str], args: argparse.Namespace, 
                file_tuple: Tuple[int, str]) -> Optional[str]:
    sys.setrecursionlimit(4500)
    run_stmts = []
    predictor = get_predictor(args)
    file_idx, filename = file_tuple
    full_filename = args.prelude + "/" + filename
    result_file = full_filename + ".error.scrape"
    temp_file = full_filename + ".error.scrape.partial"
    if args.cont:
        with contextlib.suppress(FileNotFoundError):
            with open(result_file, 'r') as f:
                if args.verbose:
                    eprint(f"Found existing scrape at {result_file}! Using it")
                return result_file
    try:
        if args.linearize:
            commands = linearize_semicolons.get_linearized(args, coqargs, file_idx, filename)
        else:
            commands = coq_serapy.load_commands_preserve(
                args, file_idx, str(full_filename))
        if args.switch:
            coq_serapy.set_switch(args.switch)
        if True:
            coq = ImprovedSerapiInstance(
                coqargs,
                coq_serapy.get_module_from_filename(filename),
                args.prelude, use_hammer=args.relevant_lemmas == "hammer")
            coq.verbose = args.verbose
            try:
                with open(temp_file, 'w') as f:
                    # while True:
                    #     commands, cmds_run = coq.run_into_next_proof(commands)
                    #     if "Lemma invert_expr_context" in cmds_run[-1]:
                    #         break
                    #     commands, cmds_run = coq.finish_proof(commands)

                    # for command in commands:
                    for command in tqdm(commands, file=sys.stdout,
                                        disable=(not args.progress),
                                        position=file_idx * 2,
                                        desc="Scraping error file", leave=False,
                                        dynamic_ncols=True,
                                        bar_format=mybarfmt):
                        process_statement(args, coq, command, f, predictor)
                shutil.move(temp_file, result_file)
                return result_file
            except coq_serapy.CoqTimeoutError:
                eprint("Command in {} timed out.".format(filename))
                return temp_file
    except Exception as e:
        eprint("FAILED: In file {}:".format(filename))
        eprint(e)
        if args.hardfail or len(args.inputs) == 1 or args.hardfail_scrape:
            raise e
    return None


def process_statement(args: argparse.Namespace,
                      coq: ImprovedSerapiInstance, command: str,
                      result_file: TextIO, predictor: TacticPredictor):
    if coq.coq.proof_context:
        prev_tactics = coq.coq.prev_tactics
        context = coq.coq.proof_context
        if args.relevant_lemmas == "local":
            relevant_lemmas = [re.sub("\n", " ", lemma)
                               for lemma in coq.coq.local_lemmas[:-1]]
        elif args.relevant_lemmas == "hammer":
            relevant_lemmas = coq.coq.get_hammer_premises()
        elif args.relevant_lemmas == "searchabout":
            relevant_lemmas = coq.coq.get_lemmas_about_head()
        else:
            assert False, args.relevant_lemmas
        result_file.write(json.dumps({"relevant_lemmas": relevant_lemmas,
                                      "prev_tactics": prev_tactics,
                                      "context": context.to_dict(),
                                      "tactic": command,
                                      "prev_tried_tactic": prev_tactics[-1],
                                      "error": "no_error"}))

        full_context_before = FullContext(relevant_lemmas,
                                          coq.coq.prev_tactics,
                                          unwrap(coq.coq.proof_context))
        predictions = predictor.predictKTactics(
            truncate_tactic_context(full_context_before.as_tcontext(),
                                    args.max_term_length),
                    args.max_attempts)

        # print(predictions)
        for prediction in predictions:
            try:
                coq.run_stmt(prediction.prediction, timeout=600, store=False)
                error = None
                coq.coq.cancel_last()
            except Exception as e:
                error = str(e)
            if error:
                result_file.write(json.dumps({"relevant_lemmas": relevant_lemmas,
                                      "prev_tactics": prev_tactics,
                                      "context": context.to_dict(),
                                      "tactic": command,
                                      "prev_tried_tactic": prediction,
                                      "error": error}))

    else:
        result_file.write(json.dumps(command))
    result_file.write("\n")

    # print("Running command: ", command)
    try:
        coq.run_stmt(command, timeout=600, store=True)
    except coq_serapy.coq_backend.CoqExn as e:
        print("CoqExn: ", e)
        print("Command: ", command)
        with open("error.log", 'a') as f:
            f.write("".join(coq.coq.tactic_history.getFullHistory()))

    
        raise e
    # in_proof_before = coq.proof_context is not None
    # try:
    #     # coq = restart_coq(coqargs, args, coq, filename, run_stmts)
    #     # print("after return")
    #     # print("after run_stmt")
    # except CoqAnomaly as e:
    #     print("anomaly")
    #     print(e)
    #     coq.restart()
    #     # coq = restart_coq(coqargs, args, coq, filename, run_stmts)
    #     coq.run_stmt(command, timeout=600, store=True)
    # in_proof_after = coq.proof_context is not None
    
    # global current_proof
    # if not in_proof_before:
    #     # print("Command: ", command)
    #     run_stmts.append(command)
    # if in_proof_before:
    #     if in_proof_after:
    #         current_proof.append(command)
    #     else:
    #         # print("Admitting...")
    #         run_stmts.append("Admitted.")
    #         current_proof = []
    # return coq

# run_stmts = []
# current_proof = []
# def restart_coq(coqargs, args, coq, filename, run_stmts):
#     coq.kill()
#     coq = None
#     coq = serapi_instance.SerapiInstance(
#         coqargs,
#         serapi_instance.get_module_from_filename(filename),
#         args.prelude)
#     for stmt in tqdm(run_stmts, file=sys.stdout,
#                         disable=(not args.progress),
#                         desc="Restarting coq", leave=False,
#                         dynamic_ncols=True,
#                         bar_format=mybarfmt):
#         coq.run_stmt(stmt)
#     for stmt in tqdm(current_proof, file=sys.stdout,
#                         disable=(not args.progress),
#                         desc="Restarting coq (current proof)", leave=False,
#                         dynamic_ncols=True,
#                         bar_format=mybarfmt):
#         coq.run_stmt(stmt)
#     return coq
    



if __name__ == "__main__":
    main()

