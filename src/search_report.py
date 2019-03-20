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
import shutil

from models.tactic_predictor import TacticPredictor, TacticContext
from predict_tactic import (static_predictors, loadPredictorByFile,
                            loadPredictorByName)
import serapi_instance
import helper
import syntax
from util import *

from typing import List, Tuple, NamedTuple, Optional, Sequence

predictor : TacticPredictor
coqargs : List[str]
includes : str
prelude : str

details_css = ["details.css"]
details_javascript = ["search-details.js"]
report_css = ["report.css"]
report_js = ["report.js"]
extra_files = details_css + details_javascript + report_css + report_js + ["logo.png"]

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

    files_done = 0

    def print_done(stats : ReportStats):
        nonlocal files_done
        files_done += 1
        print("Finished output for file {} ({} of {})"
              .format(stats.filename, files_done, len(args.filenames)))
        return stats

    with multiprocessing.pool.ThreadPool(args.threads) as pool:
        file_results = [print_done(stats) for stats in
                        pool.imap_unordered(
                            functools.partial(report_file, args, context_filter),
                            args.filenames)
                        if stats]

    print("Writing summary with {} file outputs.".format(len(file_results)))
    write_summary(args, predictor.getOptions() +
                  [("report type", "search"), ("predictor", args.predictor)],
                  commit, date, file_results)

class ReportStats(NamedTuple):
    filename : str
    num_proofs : int
    num_proofs_completed : int

def report_file(args : argparse.Namespace,
                context_filter_spec : str,
                filename : str) -> Optional[ReportStats]:
    num_proofs = 0
    num_proofs_completed = 0
    commands_in = get_commands(filename, args.verbose or args.debug)
    print("Loaded {} commands for file {}".format(len(commands_in), filename))
    commands_out = []
    with serapi_instance.SerapiContext(coqargs, includes, prelude) as coq:
        coq.debug = args.debug
        while len(commands_in) > 0:
            while not coq.full_context and len(commands_in) > 0:
                next_in_command = commands_in.pop(0)
                coq.run_stmt(next_in_command)
                commands_out.append(next_in_command)
            if len(commands_in) == 0:
                break
            num_proofs += 1
            lemma_statement = commands_out.pop()
            tactic_solution = attempt_search(args, lemma_statement, coq)
            if tactic_solution:
                commands_out.append("PROOF_START_GOOD")
                commands_out.append(lemma_statement)

                num_proofs_completed += 1
                commands_out.append("Proof.")
                commands_out += tactic_solution
                commands_out.append("Qed.")
                coq.run_stmt("Qed.")
            else:
                commands_out.append("PROOF_START_BAD")
                commands_out.append(lemma_statement)

                commands_out.append("Proof.")
                commands_out.append("Admitted.")
                coq.run_stmt("Admitted.")
            commands_out.append("PROOF_END")

            coq.cancel_last()
            while coq.full_context != None:
                coq.cancel_last()
            coq.run_stmt(lemma_statement)
            while coq.full_context != None:
                next_in_command = commands_in.pop(0)
                coq.run_stmt(next_in_command)
    write_html(args.output, filename, commands_out)
    return ReportStats(filename, num_proofs, num_proofs_completed)

def get_commands(filename : str, verbose : bool) -> List[str]:
    local_filename = prelude + "/" + filename
    loaded_commands = helper.try_load_lin(local_filename, verbose=verbose)
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
    parser.add_argument("--debug", "-vv", help="debug output",
                        action='store_const', const=True, default=False)
    parser.add_argument("--verbose", "-v", help="verbose output",
                        action='store_const', const=True, default=False)
    parser.add_argument('--context-filter', dest="context_filter", type=str,
                        default=None)
    parser.add_argument('--weightsfile', default=None)
    parser.add_argument('--predictor', choices=list(static_predictors.keys()),
                        default=None)
    parser.add_argument("--search-width", dest="search_width", type=int, default=3)
    parser.add_argument("--search-depth", dest="search_depth", type=int, default=10)
    parser.add_argument('filenames', nargs="+", help="proof file name (*.v)")
    return parser.parse_args(args_list), parser

def get_metadata() -> Tuple[str, datetime.datetime]:
    cur_commit = subprocess.check_output(["git show --oneline | head -n 1"],
                                         shell=True).decode('utf-8').strip()
    cur_date = datetime.datetime.now()
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

from yattag import Doc
Tag = Callable[..., Doc.Tag]
Text = Callable[..., None]
Line = Callable[..., None]

def html_header(tag : Tag, doc : Doc, text : Text, css : List[str],
                javascript : List[str], title : str) -> None:
    with tag('head'):
        for filename in css:
            doc.stag('link', href=filename, rel='stylesheet')
        for filename in javascript:
            with tag('script', type='text/javascript',
                     src=filename):
                pass
        with tag('title'):
            text(title)

def write_summary(args : argparse.Namespace, options : Sequence[Tuple[str, str]],
                  cur_commit : str, cur_date : datetime.datetime,
                  individual_stats : List[ReportStats]) -> None:
    def report_header(tag : Any, doc : Doc, text : Text) -> None:
        html_header(tag, doc, text,report_css, report_js,
                    "Proverbot Report")
    combined_stats = combine_file_results(individual_stats)
    doc, tag, text, line = Doc().ttl()
    with tag('html'):
        report_header(tag, doc, text)
        with tag('body'):
            with tag('h4'):
                text("{} files processed".format(len(args.filenames)))
            with tag('h5'):
                text("Commit: {}".format(cur_commit))
            with tag('h5'):
                text("Run on {}".format(cur_date.strftime("%Y-%m-%d %H:%M:%S.%f")))
            with tag('img',
                     ('src', 'logo.png'),
                     ('id', 'logo')):
                pass
            with tag('h2'):
                text("Proofs Completed: {}% ({}/{})"
                     .format(stringified_percent(combined_stats.num_proofs_completed,
                                                 combined_stats.num_proofs),
                             combined_stats.num_proofs_completed,
                             combined_stats.num_proofs))
            with tag('ul'):
                for k, v in options:
                    if k == 'filenames':
                        continue
                    elif not v:
                        continue
                    with tag('li'):
                        text("{}: {}".format(k, v))

            with tag('table'):
                with tag('tr', klass="header"):
                    line('th', 'Filename')
                    line('th', 'Number of Proofs in File')
                    line('th', '% Proofs Completed')
                    line('th', 'Details')
                sorted_rows = sorted(individual_stats,
                                     key=lambda fresult:fresult.num_proofs,
                                     reverse=True)
                for fresult in sorted_rows:
                    if fresult.num_proofs == 0:
                        continue
                    with tag('tr'):
                        line('td', fresult.filename)
                        line('td', str(fresult.num_proofs))
                        line('td', stringified_percent(fresult.num_proofs_completed,
                                                       fresult.num_proofs))
                        with tag('td'):
                            with tag('a',
                                     href=escape_filename(fresult.filename) + ".html"):
                                text("Details")
                with tag('tr'):
                    line('td', "Total");
                    line('td', str(combined_stats.num_proofs))
                    line('td', stringified_percent(combined_stats.num_proofs_completed,
                                                   combined_stats.num_proofs))
    for filename in extra_files:
        shutil.copy(os.path.dirname(os.path.abspath(__file__)) + "/../reports/" + filename,
                    args.output + "/" + filename)
    with open("{}/report.html".format(args.output), "w") as fout:
        fout.write(doc.getvalue())

def write_html(output_dir : str, filename : str, commands_out : List[str]) -> None:
    def details_header(tag : Any, doc : Doc, text : Text, filename : str) -> None:
        html_header(tag, doc, text, details_css, details_javascript,
                    "Proverbot Detailed Report for {}".format(filename))
    doc, tag, text, line = Doc().ttl()
    region_idx = 0
    with tag('html'):
        details_header(tag, doc, text, filename)
        with tag('body', onload='init()'), tag('pre'):
            while len(commands_out) > 0:
                command = commands_out.pop(0)
                if command == "PROOF_START_GOOD" or command == "PROOF_START_BAD":
                    if command == "PROOF_START_GOOD":
                        k = 'good'
                    else:
                        k = 'bad'
                    doc.stag('br')
                    with tag('button', klass='collapsible ' + k,
                             id='collapsible-{}'.format(region_idx)):
                        with tag('code', klass='buttontext'):
                            text(commands_out.pop(0))
                    with tag('div', klass='region'):
                        command = commands_out.pop(0)
                        while(command != "PROOF_END"):
                            with tag('code', 'plaincommand'):
                                text(command)
                            doc.stag('br')
                            command = commands_out.pop(0)
                else:
                    with tag('code', klass='plaincommand'):
                        text(command)
    with open("{}/{}.html".format(output_dir, escape_filename(filename)), 'w') as fout:
        fout.write(syntax.syntax_highlight(doc.getvalue()))

def combine_file_results(stats : List[ReportStats]) -> ReportStats:
    return ReportStats("",
                       sum([s.num_proofs for s in stats]),
                       sum([s.num_proofs_completed for s in stats]))

# The core of the search report

# This method attempts to complete proofs using search.
def attempt_search(args : argparse.Namespace,
                   lemma_statement : str,
                   coq : serapi_instance.SerapiInstance) \
    -> Optional[List[str]]:
    commands = dfs_proof_search_with_graph(lemma_statement, coq, args)
    return commands

def dfs_proof_search(lemma_statement : str, coq : serapi_instance.SerapiInstance,
                     args : argparse.Namespace) -> Optional[List[str]]:
    def get_context() -> TacticContext:
        return TacticContext(coq.prev_tactics, coq.get_hypothesis(),
                             coq.get_goals())
    def predictions() -> List[str]:
        return [pred.prediction for pred in
                predictor.predictKTactics(get_context(), args.search_width)]
    def search(current_path : List[str]) -> Optional[List[str]]:
        for prediction in predictions():
            try:
                coq.quiet = True
                coq.run_stmt(prediction)
                if completed_proof(coq):
                    return current_path + [prediction]
                elif len(current_path) + 1 < args.search_depth:
                    sub_search_result = search(current_path + [prediction])
                    if sub_search_result:
                        return sub_search_result
                coq.cancel_last()
            except (serapi_instance.CoqExn, serapi_instance.TimeoutError):
                continue
        return None
    return search([])

def bfs_proof_search(lemma_statement : str, coq : serapi_instance.SerapiInstance,
                     args : argparse.Namespace) -> Optional[List[str]]:
    states_to_explore = [[lemma_statement]]
    coq.cancel_last()

    while len(states_to_explore) > 0:
        next_state_to_explore = states_to_explore.pop(0)
        try:
            coq.quiet = True
            for tactic in next_state_to_explore:
                coq.run_stmt(tactic)
        except (serapi_instance.CoqExn, serapi_instance.TimeoutError):
            while coq.full_context:
                coq.cancel_last()
            continue
        finally:
            coq.quiet = False
        if completed_proof(coq):
            return next_state_to_explore[1:]
        state_context = TacticContext(coq.prev_tactics,
                                      coq.get_hypothesis(),
                                      coq.get_goals())
        if len(next_state_to_explore) < args.search_depth:
            predictions = predictor.predictKTactics(state_context, args.search_width)
            for prediction in predictions:
                states_to_explore.append(next_state_to_explore + [prediction.prediction])
        while coq.full_context:
            coq.cancel_last()
    coq.run_stmt(lemma_statement)
    return None

def completed_proof(coq : serapi_instance.SerapiInstance) -> bool:
    coq.run_stmt("Unshelve.")
    return coq.full_context == ""
