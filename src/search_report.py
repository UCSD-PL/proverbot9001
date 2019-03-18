#!/usr/bin/env python3.7

import argparse
import subprocess
import os
import sys
import multiprocessing
import re
import datetime
import time
import functools

from models.tactic_predictor import TacticPredictor
from predict_tactic import (static_predictors, loadPredictorByFile,
                            loadPredictorByName)
import serapi_instance
import helper

from typing import List, Tuple, NamedTuple, Optional

predictor : TacticPredictor
coqargs : List[str]
includes : str
prelude : str

def main(arg_list : List[str]) -> None:
    global predictor
    global coqargs
    global includes
    global prelude

    args, parser = parse_arguments(arg_list)
    commit, date = get_metadata()
    predictor = get_predictor(parser, args)
    base = os.path.dirname(os.path.abspath(__file__)) + "/.."
    coqargs = ["{}/coq-serapi/sertop.native".format(base),
               "--prelude={}/coq".format(base)]
    prelude = args.prelude
    with open(prelude + "/_CoqProject", 'r') as includesfile:
        includes = includesfile.read()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    context_filter = args.context_filter or dict(predictor.getOptions())["context_filter"]

    with multiprocessing.pool.ThreadPool(args.threads) as pool:
        file_results = [stats for stats in
                        pool.imap_unordered(
                            functools.partial(report_file, args, context_filter),
                            args.filenames)
                        if stats]

    write_summary(args, predictor.getOptions() +
                  [("report type", "static"), ("predictor", args.predictor)],
                  commit, date, file_results)

class ReportStats(NamedTuple):
    num_proofs : int
    num_proofs_completed : int

def report_file(args : argparse.Namespace,
                context_filter_spec : str,
                filename : str) -> Optional[ReportStats]:
    num_proofs = 0
    num_proofs_completed = 0
    commands_in = get_commands(filename)
    commands_out = []
    with serapi_instance.SerapiContext(coqargs, includes, prelude) as coq:
        while len(commands_in) > 0:
            while not coq.proof_context:
                next_in_command = commands_in.pop(0)
                coq.run_stmt(next_in_command)
                commands_out.append(next_in_command)
            lemma_statement = next_in_command
            tactic_solution = attempt_search(lemma_statement, coq)
            if tactic_solution:
                commands_out += tactic_solution
                commands_out.append("Qed.")
                coq.run_stmt("Qed.")
            else:
                commands_out.append("Admitted.")
                coq.run_stmt("Admitted.")
    write_html(commands_out)
    return ReportStats(num_proofs, num_proofs_completed)

def get_commands(filename : str) -> List[str]:
    local_filename = prelude + "/" + filename
    loaded_commands = helper.try_load_lin(local_filename)
    if loaded_commands is None:
        fresh_commands = helper.lift_and_linearize(
            helper.load_commands_preserve(prelude + "/" + filename),
            coqargs, includes, prelude,
            filename, False)
        helper.save_lin(fresh_commands, local_filename)
        return fresh_commands
    else:
        return loaded_commands

def parse_arguments(args_list : List[str]) -> Tuple[argparse.Namespace,
                                                    argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description=
        "Produce an html report from attempting to complete proofs using Proverbot9001.")
    parser.add_argument("-j", "--threads", default=16, type=int)
    parser.add_argument("--prelude", default=".")
    parser.add_argument("--output", "-o", help="output data folder name",
                        default="search-report")
    parser.add_argument('--context-filter', dest="context_filter", type=str,
                        default=None)
    parser.add_argument('--weightsfile', default=None)
    parser.add_argument('--predictor', choices=list(static_predictors.keys()),
                        default=None)
    parser.add_argument("--search-width", dest="search_width", type=int, default=3)
    parser.add_argument("--search-depth", dest="search_depth", type=int, default=10)
    parser.add_argument('filenames', nargs="+", help="proof file name (*.v)")
    return parser.parse_args(args_list), parser

def get_metadata() -> Tuple[str, str]:
    cur_commit = subprocess.check_output(["git show --oneline | head -n 1"],
                                         shell=True).decode('utf-8').strip()
    cur_date = str(datetime.datetime.now())
    return cur_commit, cur_date

def get_predictor(parser : argparse.ArgumentParser,
                  args : argparse.Namespace) -> TacticPredictor:
    predictor : TacticPredictor
    if args.weightsfile:
        predictor = loadPredictorByFile(args.weightsfile)
    elif args.predictor:
        predictor = loadPredictorByName(args.predictor)
    else:
        print("You must specify either --weightsfile or --predictor!")
        parser.print_help()
        sys.exit(1)
    return predictor
