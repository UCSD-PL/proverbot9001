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
from serapi_instance import FullContext, Subgoal
import linearize_semicolons
import syntax
from format import format_goal
from util import *

from typing import List, Tuple, NamedTuple, Optional, Sequence, Dict

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
    try:
        with open(prelude + "/_CoqProject", 'r') as includesfile:
            includes = includesfile.read()
    except FileNotFoundError:
        print("Didn't find a _CoqProject file in prelude dir")
        includes = ""

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

    with multiprocessing.pool.ThreadPool(args.num_threads) as pool:
        file_results = [print_done(stats) for stats in
                        pool.imap_unordered(
                            functools.partial(report_file, args, context_filter),
                            args.filenames)
                        if stats]

    print("Writing summary with {} file outputs.".format(len(file_results)))
    write_summary(args, predictor.getOptions() +
                  [("report type", "search"),
                   ("search width", args.search_width),
                   ("search depth", args.search_depth)],
                  commit, date, file_results)

class ReportStats(NamedTuple):
    filename : str
    num_proofs : int
    num_proofs_failed : int
    num_proofs_completed : int

from enum import Enum, auto
from typing import Union
class SearchStatus(Enum):
    SUCCESS = auto()
    INCOMPLETE = auto()
    FAILURE = auto()

class VernacBlock(NamedTuple):
    commands : List[str]

class TacticInteraction(NamedTuple):
    tactic : str
    context_before : FullContext

class ProofBlock(NamedTuple):
    lemma_statement : str
    status : SearchStatus
    predicted_tactics : List[TacticInteraction]
    original_tactics : List[TacticInteraction]

DocumentBlock = Union[VernacBlock, ProofBlock]

class ArgsMismatchException(Exception):
    pass

def report_file(args : argparse.Namespace,
                context_filter_spec : str,
                filename : str) -> Optional[ReportStats]:
    if args.resume:
        try:
            stats = read_stats_from_csv(args, filename)
            print(f"Resumed {filename} from existing state")
            return stats
        except FileNotFoundError:
            pass
        except ArgsMismatchException as e:
            print(f"Arguments in report for {filename} didn't match current arguments! {e} Overwriting (interrupt to cancel).")

    num_proofs = 0
    num_proofs_failed = 0
    num_proofs_completed = 0
    commands_in = get_commands(filename, args.verbose or args.debug)
    commands_run : List[str] = []
    num_commands_total = len(commands_in)
    def show_progress(tag:str=""):
        if args.progress and args.num_threads == 1:
            print("\r{:.2f}% done ({} of {} commands processed) {}".format(
                100 * (1 - (len(commands_in) / num_commands_total)),
                num_commands_total - len(commands_in), num_commands_total,
                tag),
                  end="")
            sys.stdout.flush()
    # Run vernacular until the next proof (or end of file)
    def run_to_next_proof(coq : serapi_instance.SerapiInstance) -> str:
        nonlocal commands_run
        nonlocal commands_in
        nonlocal blocks_out
        vernacs : List[str] = []
        assert not coq.full_context
        while not coq.full_context and len(commands_in) > 0:
            next_in_command = commands_in.pop(0)
            coq.run_stmt(next_in_command)
            if not coq.full_context:
                vernacs.append(next_in_command)
            show_progress()
        if len(vernacs) > 0:
            blocks_out.append(VernacBlock(vernacs))
            commands_run += vernacs
        return next_in_command

    def run_to_next_vernac(coq : serapi_instance.SerapiInstance,
                           initial_full_context : FullContext,
                           lemma_statement : str):
        nonlocal commands_run
        nonlocal commands_in
        nonlocal num_proofs_failed
        nonlocal num_proofs_completed
        nonlocal blocks_out
        coq.run_stmt(lemma_statement)
        commands_run.append(lemma_statement)
        original_tactics : List[TacticInteraction] = []
        while coq.full_context != "":
            next_in_command = commands_in.pop(0)
            context_before = coq.getAllGoals()
            coq.run_stmt(next_in_command)
            commands_run.append(next_in_command)
            original_tactics.append(TacticInteraction(next_in_command, context_before))
            show_progress()
        empty_context = FullContext([])
        # Append the proof data
        if not tactic_solution:
            if search_status == SearchStatus.FAILURE:
                num_proofs_failed += 1
            blocks_out.append(ProofBlock(lemma_statement, search_status,
                                         [TacticInteraction("Proof.",
                                                            initial_full_context),
                                          TacticInteraction("Admitted.",
                                                            initial_full_context)],
                                         original_tactics))
        else:
            num_proofs_completed += 1
            blocks_out.append(ProofBlock(lemma_statement, search_status,
                                         [TacticInteraction("Proof",
                                                            initial_full_context)] +
                                         tactic_solution +
                                         [TacticInteraction("Qed.", empty_context)],
                                         original_tactics))

    print("Loaded {} commands for file {}".format(len(commands_in), filename))
    blocks_out : List[DocumentBlock] = []
    if args.progress and args.num_threads == 1:
        print("0.00% done (0 of {} commands processed)".format(num_commands_total),
              end="")
    while len(commands_in) > 0:
        try:
            # print("Starting a coq instance...")
            with serapi_instance.SerapiContext(coqargs, includes, prelude) as coq:
                for command in commands_run:
                    coq.run_stmt(command)
                if len(commands_run) > 0 and args.verbose and args.num_threads == 1:
                    print("Caught up with commands:\n{}\n...\n{}".format(commands_run[0].strip(), commands_run[-1].strip()))
                coq.debug = args.debug
                while len(commands_in) > 0:
                    lemma_statement = run_to_next_proof(coq)
                    if len(commands_in) == 0:
                        break
                    # Get beginning of next proof
                    num_proofs += 1
                    initial_context = coq.getAllGoals()
                    # Try to search
                    show_progress()
                    search_status, tactic_solution = \
                        attempt_search(args, lemma_statement, coq)
                    # Cancel until before the proof
                    try:
                        while coq.full_context != "":
                            coq.cancel_last()
                    except serapi_instance.CoqExn as e:
                        commands_in.insert(0, lemma_statement)
                        raise serapi_instance.CoqAnomaly(f"While cancelling: {e}")
                    # Run the original proof
                    run_to_next_vernac(coq, initial_context, lemma_statement)
                if args.progress:
                    print("\r")
        except serapi_instance.CoqAnomaly as e:
            if args.verbose:
                print(f"Hit a coq anomaly {e.msg}! Restarting coq instance.")
            if args.hardfail:
                raise e
        except:
            print(f"FAILED: in file {filename}")
            raise
    write_html(args.output, filename, blocks_out)
    write_csv(args, filename, blocks_out)
    return ReportStats(filename, num_proofs, num_proofs_failed, num_proofs_completed)

def get_commands(filename : str, verbose : bool) -> List[str]:
    local_filename = prelude + "/" + filename
    loaded_commands = serapi_instance.try_load_lin(local_filename, verbose=verbose)
    if loaded_commands is None:
        fresh_commands = linearize_semicolons.preprocess_file_commands(
            serapi_instance.load_commands_preserve(prelude + "/" + filename),
            coqargs, includes, prelude,
            local_filename, filename, False)
        serapi_instance.save_lin(fresh_commands, local_filename)
        return fresh_commands
    else:
        return loaded_commands

def parse_arguments(args_list : List[str]) -> Tuple[argparse.Namespace,
                                                    argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description=
        "Produce an html report from attempting to complete proofs using Proverbot9001.")
    parser.add_argument("-j", "--threads", dest="num_threads", default=16, type=int)
    parser.add_argument("--prelude", default=".")
    parser.add_argument("--output", "-o", help="output data folder name",
                        default="search-report")
    parser.add_argument("--debug", "-vv", help="debug output",
                        action='store_const', const=True, default=False)
    parser.add_argument("--verbose", "-v", help="verbose output",
                        action='store_const', const=True, default=False)
    parser.add_argument("--progress", "-P", help="show progress of files",
                        action='store_const', const=True, default=False)
    parser.add_argument("--hardfail", "-f", help="fail when hitting a coq anomaly",
                        action='store_const', const=True, default=False)
    parser.add_argument('--context-filter', dest="context_filter", type=str,
                        default=None)
    parser.add_argument('--weightsfile', default=None)
    parser.add_argument('--predictor', choices=list(static_predictors.keys()),
                        default=None)
    parser.add_argument("--search-width", dest="search_width", type=int, default=3)
    parser.add_argument("--search-depth", dest="search_depth", type=int, default=10)
    parser.add_argument("--no-resume", dest="resume",
                        const=False, default=True, action='store_const')
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

def write_summary_html(filename : str,
                       options : Sequence[Tuple[str, str]],
                       cur_commit : str, cur_date : datetime.datetime,
                       individual_stats : List[ReportStats],
                       combined_stats : ReportStats) -> None:
    def report_header(tag : Any, doc : Doc, text : Text) -> None:
        html_header(tag, doc, text,report_css, report_js,
                    "Proverbot Report")
    doc, tag, text, line = Doc().ttl()
    with tag('html'):
        report_header(tag, doc, text)
        with tag('body'):
            with tag('h4'):
                text("{} files processed".format(len(individual_stats)))
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
                    line('th', '% Proofs Incomplete')
                    line('th', '% Proofs Failed')
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
                        line('td', stringified_percent(fresult.num_proofs -
                                                       (fresult.num_proofs_completed +
                                                        fresult.num_proofs_failed),
                                                       fresult.num_proofs))
                        line('td', stringified_percent(fresult.num_proofs_failed,
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
                    line('td', stringified_percent(combined_stats.num_proofs -
                                                   (combined_stats.num_proofs_completed +
                                                    combined_stats.num_proofs_failed),
                                                   combined_stats.num_proofs))
                    line('td', stringified_percent(combined_stats.num_proofs_failed,
                                                   combined_stats.num_proofs))
    with open(filename, "w") as fout:
        fout.write(doc.getvalue())

import csv
def write_summary_csv(filename : str, combined_stats : ReportStats,
                      options : Sequence[Tuple[str, str]]):
    with open(filename, 'w', newline='') as csvfile:
        for k, v in options:
            csvfile.write("# {}: {}\n".format(k, v))
        rowwriter = csv.writer(csvfile, lineterminator=os.linesep)
        rowwriter.writerow([combined_stats.num_proofs,
                            combined_stats.num_proofs_failed,
                            combined_stats.num_proofs_completed])

def write_summary(args : argparse.Namespace, options : Sequence[Tuple[str, str]],
                  cur_commit : str, cur_date : datetime.datetime,
                  individual_stats : List[ReportStats]) -> None:
    combined_stats = combine_file_results(individual_stats)
    write_summary_html("{}/report.html".format(args.output),
                       options, cur_commit, cur_date, individual_stats, combined_stats)
    write_summary_csv("{}/report.csv".format(args.output), combined_stats, options)
    write_proof_csv(args.output, [s.filename for s in individual_stats])
    for filename in extra_files:
        shutil.copy(os.path.dirname(os.path.abspath(__file__)) + "/../reports/" + filename,
                    args.output + "/" + filename)
def write_proof_csv(output_dir : str, filenames : List[str]):
    with open('{}/proofs.csv'.format(output_dir), 'w') as fout:
        fout.write("lemma, status, prooflength\n")
        for filename in filenames:
            with open("{}/{}.csv".format(output_dir, escape_filename(filename)), 'r') \
                 as fin:
                fout.writelines(fin)

def write_csv(args : argparse.Namespace, filename : str, doc_blocks : List[DocumentBlock]):
    with open("{}/{}.csv".format(args.output, escape_filename(filename)),
              'w', newline='') as csvfile:
        for k, v in vars(args).items():
            csvfile.write("# {}: {}\n".format(k, v))

        rowwriter = csv.writer(csvfile, lineterminator=os.linesep)
        for block in doc_blocks:
            if isinstance(block, ProofBlock):
                rowwriter.writerow([block.lemma_statement.strip(),
                                    block.status,
                                    len(block.original_tactics)])
def read_csv_options(f : Iterable[str]) -> Tuple[argparse.Namespace, Iterable[str]]:
    params : Dict[str, str] = {}
    f_iter = iter(f)
    final_line = ""
    for line in f_iter:
        param_match = re.match("# (.*): (.*)", line)
        if param_match:
            params[param_match.group(1)] = param_match.group(2)
        else:
            final_line = line
            break
    rest_iter : Iterable[str]
    if final_line == "":
        rest_iter = iter([])
    else:
        rest_iter = itertools.chain([final_line], f_iter)
    return argparse.Namespace(**params), rest_iter

important_args = ["prelude", "context_filter", "weightsfile", "predictor", "search_width", "search_depth"]

def read_stats_from_csv(args : argparse.Namespace, vfilename : str) -> ReportStats:
    num_proofs = 0
    num_proofs_failed = 0
    num_proofs_completed = 0
    with open("{}/{}.csv".format(args.output, escape_filename(vfilename)),
              'r', newline='') as csvfile:
        saved_args, rest_iter = read_csv_options(csvfile)
        for arg in important_args:
            try:
                oldval = str(vars(saved_args)[arg])
                newval = str(vars(args)[arg])
                if oldval != newval:
                    raise ArgsMismatchException(f"Old value of {arg} is {oldval}, "
                                                f"new value is {newval}")
            except KeyError:
                raise ArgsMismatchException(f"No old value for arg {arg} found.")
        rowreader = csv.reader(rest_iter, lineterminator=os.linesep)
        for row in rowreader:
            num_proofs += 1
            if row[1] == "SearchStatus.SUCCESS":
                num_proofs_completed += 1
            elif row[1] == "SearchStatus.FAILURE":
                num_proofs_failed += 1
            else:
                assert row[1] == "SearchStatus.INCOMPLETE"
    return ReportStats(vfilename, num_proofs, num_proofs_failed, num_proofs_completed)

def write_html(output_dir : str, filename : str,
               doc_blocks : List[DocumentBlock]) -> None:
    doc, tag, text, line = Doc().ttl()
    with tag('html'):
        html_header(tag, doc, text, details_css, details_javascript,
                    "Proverbot Detailed Report for {}".format(filename))
        with tag('body', onload='init()'), tag('pre'):
            for block_idx, block in enumerate(doc_blocks):
                if isinstance(block, VernacBlock):
                    write_commands(block.commands, tag, text, doc)
                else:
                    assert isinstance(block, ProofBlock)
                    status_klass = classFromSearchStatus(block.status)
                    write_lemma_button(block.lemma_statement, status_klass, tag, text)
                    with tag('div', klass='region'):
                        with tag('div', klass='predicted'):
                            write_tactics(block.predicted_tactics, block_idx,
                                          tag, text, doc)
                        with tag('div', klass='original'):
                            write_tactics(block.original_tactics, block_idx,
                                          tag, text, doc)
    with open("{}/{}.html".format(output_dir, escape_filename(filename)), 'w') as fout:
        fout.write(syntax.syntax_highlight(doc.getvalue()))

def combine_file_results(stats : List[ReportStats]) -> ReportStats:
    return ReportStats("",
                       sum([s.num_proofs for s in stats]),
                       sum([s.num_proofs_failed for s in stats]),
                       sum([s.num_proofs_completed for s in stats]))

def write_lemma_button(lemma_statement : str, status_klass : str, tag : Tag, text : Text):
    lemma_name = \
        serapi_instance.lemma_name_from_statement(lemma_statement)
    with tag('button', klass='collapsible {}'.format(status_klass),
             onmouseover="hoverLemma(\"{}\")".format(lemma_name),
             onmouseout="unhoverLemma(\"{}\")".format(lemma_name)):
        with tag('code', klass='buttontext'):
            text(lemma_statement.strip())
def write_commands(commands : List[str], tag : Tag, text : Text, doc : Doc):
    for cmd in commands:
        with tag('code', klass='plaincommand'):
            text(cmd.strip("\n"))
        doc.stag('br')

def subgoal_to_string(sg : Subgoal) -> str:
    return "(\"" + sg.goal + "\", (\"" + "\",\"".join(sg.hypotheses) + "\"))"

def write_tactics(tactics : List[TacticInteraction],
                  region_idx : int,
                  tag : Tag, text : Text, doc : Doc):
    for t_idx, t in enumerate(tactics):
        idStr = '{}-{}'.format(region_idx, t_idx)
        with tag('span',
                 ('data-subgoals',
                  "(" + ",".join([subgoal_to_string(subgoal)
                                  for subgoal in t.context_before.subgoals]) + ")"),
                 id='command-{}'.format(idStr),
                 onmouseover='hoverTactic("{}")'.format(idStr),
                 onmouseout='unhoverTactic()'):
            with tag('code', klass='plaincommand'):
                text(t.tactic.strip())
            doc.stag('br')

def classFromSearchStatus(status : SearchStatus) -> str:
    if status == SearchStatus.SUCCESS:
        return 'good'
    elif status == SearchStatus.INCOMPLETE:
        return 'okay'
    else:
        return 'bad'


# The core of the search report

class SearchResult(NamedTuple):
    status : SearchStatus
    commands : Optional[List[TacticInteraction]]

# This method attempts to complete proofs using search.
def attempt_search(args : argparse.Namespace,
                   lemma_statement : str,
                   coq : serapi_instance.SerapiInstance) \
    -> SearchResult:
    result = dfs_proof_search_with_graph(lemma_statement, coq, args)
    return result

# This implementation is here for reference/documentation
# def dfs_proof_search(lemma_statement : str, coq : serapi_instance.SerapiInstance,
#                      args : argparse.Namespace) -> Optional[List[str]]:
#     def get_context() -> TacticContext:
#         return TacticContext(coq.prev_tactics(), coq.get_hypothesis(),
#                              coq.get_goals())
#     def predictions() -> List[str]:
#         return [pred.prediction for pred in
#                 predictor.predictKTactics(get_context(), args.search_width)]
#     def search(current_path : List[str]) -> Optional[List[str]]:
#         for prediction in predictions():
#             try:
#                 coq.quiet = True
#                 coq.run_stmt(prediction)
#                 if completed_proof(coq):
#                     return current_path + [prediction]
#                 elif len(current_path) + 1 < args.search_depth:
#                     sub_search_result = search(current_path + [prediction])
#                     if sub_search_result:
#                         return sub_search_result
#                 coq.cancel_last()
#             except (serapi_instance.CoqExn, serapi_instance.TimeoutError):
#                 continue
#         return None
#     return search([])

import pygraphviz as pgv
# from graphviz import Digraph

class LabeledNode(NamedTuple):
    prediction : str
    node_id : int
    context_before : FullContext
class SubSearchResult (NamedTuple):
    solution : Optional[List[TacticInteraction]]
    solved_subgoals : int

def dfs_proof_search_with_graph(lemma_statement : str,
                                coq : serapi_instance.SerapiInstance,
                                args : argparse.Namespace) \
                                -> SearchResult:
    search_graph = pgv.AGraph(directed=True)
    next_node_id = 0
    lemma_name = serapi_instance.lemma_name_from_statement(lemma_statement)
    def mkNode(prediction : str, context_before : FullContext, **kwargs) -> LabeledNode:
        nonlocal next_node_id
        search_graph.add_node(next_node_id, label=prediction, **kwargs)
        node_obj = LabeledNode(prediction, next_node_id, context_before)
        next_node_id += 1
        return node_obj
    def mkEdge(src : LabeledNode, dest : LabeledNode, **kwargs) -> None:
        search_graph.add_edge(src.node_id, dest.node_id, **kwargs)
    def setNodeColor(node : LabeledNode, color : str) -> None:
        node_handle = search_graph.get_node(node.node_id)
        node_handle.attr["fillcolor"] = color
        node_handle.attr["style"] = "filled"

    start_node = mkNode(serapi_instance.lemma_name_from_statement(lemma_statement),
                        FullContext([]))
    def edgeToPrev(prediction : LabeledNode, current_path : List[LabeledNode]) -> None:
        if len(current_path) == 0:
            mkEdge(start_node, prediction)
        else:
            mkEdge(current_path[-1], prediction)
    def get_prediction_context() -> TacticContext:
        context = TacticContext(coq.prev_tactics, coq.hypothesis,
                                coq.goals)
        return context
    def get_fullcontext() -> FullContext:
        return coq.getAllGoals()
    def make_predictions() -> List[str]:
        return [pred.prediction for pred in
                predictor.predictKTactics(get_prediction_context(), args.search_width)]
    def subgoalSurjective(newsub : serapi_instance.Subgoal,
                          oldsub : serapi_instance.Subgoal) -> bool:
        oldhyp_terms = [serapi_instance.get_hyp_type(hyp) for hyp in oldsub.hypotheses]
        for newhyp_term in [serapi_instance.get_hyp_type(hyp)
                            for hyp in newsub.hypotheses]:
            if newhyp_term not in oldhyp_terms:
                return False
        return newsub.goal == oldsub.goal
    def contextSurjective(newcontext : FullContext, oldcontext : FullContext):
        for oldsub in oldcontext.subgoals:
            if not any([subgoalSurjective(newsub, oldsub)
                        for newsub in newcontext.subgoals]):
                return False
        return True
    def contextInPath(full_context : FullContext, path : List[LabeledNode]):
        return any([contextSurjective(full_context, n.context_before)
                    for n in path])
    hasUnexploredNode = False
    def search(current_path : List[LabeledNode]) -> SubSearchResult:
        nonlocal hasUnexploredNode
        predictions = make_predictions()
        context_before = get_fullcontext()
        predictionNodes = [mkNode(prediction, context_before)
                           for prediction in predictions]

        for predictionNode in predictionNodes:
            edgeToPrev(predictionNode, current_path)
        for prediction, predictionNode in zip(predictions, predictionNodes):
            subgoals_closed = 0
            try:
                coq.quiet = True
                coq.run_stmt(prediction)
                num_stmts = 1
                while coq.count_fg_goals() == 0 and not completed_proof(coq):
                    setNodeColor(predictionNode, "blue")
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
                context_after = get_fullcontext()
                if completed_proof(coq):
                    mkEdge(predictionNode, mkNode("QED", FullContext([]),
                                                  fillcolor="green", style="filled"))
                    for node in [start_node] + current_path + [predictionNode]:
                        setNodeColor(node, "green")
                    return SubSearchResult([TacticInteraction(n.prediction,
                                                              n.context_before)
                                            for n in current_path + [predictionNode]],
                                           subgoals_closed)
                elif contextInPath(context_after, current_path + [predictionNode]):
                    setNodeColor(predictionNode, "orange")
                    for _ in range(num_stmts):
                        coq.cancel_last()
                elif len(current_path) + 1 < args.search_depth:
                    sub_search_result = search(current_path + [predictionNode])
                    if sub_search_result.solution or \
                       sub_search_result.solved_subgoals > subgoals_opened:
                        return SubSearchResult(sub_search_result.solution,
                                               subgoals_closed +
                                               sub_search_result.solved_subgoals
                                               - subgoals_opened)
                    stmts_to_cancel = num_stmts - sub_search_result.solved_subgoals
                    for _ in range(stmts_to_cancel):
                        coq.cancel_last()
                    subgoals_closed = 0
                else:
                    hasUnexploredNode = True
                    for _ in range(num_stmts):
                        coq.cancel_last()
                    subgoals_closed = 0
            except (serapi_instance.CoqExn, serapi_instance.TimeoutError,
                    serapi_instance.OverflowError, serapi_instance.ParseError,
                    serapi_instance.UnrecognizedError):
                setNodeColor(predictionNode, "red")
                continue
            except serapi_instance.NoSuchGoalError:
                raise
            if subgoals_closed > 0:
                return SubSearchResult(None, subgoals_closed)
        return SubSearchResult(None, 0)
    command_list, _ = search([])
    search_graph.draw(args.output + "/" + escape_lemma_name(lemma_name) + ".png",
                      prog="dot")
    if command_list:
        return SearchResult(SearchStatus.SUCCESS, command_list)
    elif hasUnexploredNode:
        return SearchResult(SearchStatus.INCOMPLETE, None)
    else:
        return SearchResult(SearchStatus.FAILURE, None)


def completed_proof(coq : serapi_instance.SerapiInstance) -> bool:
    completed = coq.full_context == "none" and coq.tactic_history.curDepth() == 0
    return completed
