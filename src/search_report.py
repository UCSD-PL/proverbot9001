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
from format import format_goal
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
    context_before : TacticContext

class ProofBlock(NamedTuple):
    lemma_statement : str
    status : SearchStatus
    predicted_tactics : List[TacticInteraction]
    original_tactics : List[TacticInteraction]

DocumentBlock = Union[VernacBlock, ProofBlock]

def report_file(args : argparse.Namespace,
                context_filter_spec : str,
                filename : str) -> Optional[ReportStats]:
    num_proofs = 0
    num_proofs_failed = 0
    num_proofs_completed = 0
    commands_in = get_commands(filename, args.verbose or args.debug)
    print("Loaded {} commands for file {}".format(len(commands_in), filename))
    blocks_out : List[DocumentBlock] = []
    with serapi_instance.SerapiContext(coqargs, includes, prelude) as coq:
        coq.debug = args.debug
        while len(commands_in) > 0:
            # Get vernacular until the next proof (or end of file)
            vernacs : List[str] = []
            while not coq.full_context and len(commands_in) > 0:
                next_in_command = commands_in.pop(0)
                coq.run_stmt(next_in_command)
                if not coq.full_context:
                    vernacs.append(next_in_command)
            if len(vernacs) > 0:
                blocks_out.append(VernacBlock(vernacs))
            if len(commands_in) == 0:
                break

            # Get beginning of next proof
            num_proofs += 1
            lemma_statement = next_in_command
            initial_context = TacticContext(coq.prev_tactics, coq.get_hypothesis(),
                                            coq.get_goals())

            # Try to search
            search_status, tactic_solution = attempt_search(args, lemma_statement, coq)

            # Cancel until before the proof
            while coq.full_context != None:
                coq.cancel_last()
            # Run the original proof
            coq.run_stmt(lemma_statement)
            original_tactics : List[TacticInteraction] = []
            while coq.full_context != None:
                next_in_command = commands_in.pop(0)
                context_before = TacticContext(coq.prev_tactics, coq.get_hypothesis(),
                                               coq.get_goals())
                coq.run_stmt(next_in_command)
                original_tactics.append(TacticInteraction(next_in_command, context_before))

            empty_context = TacticContext([], [], "")
            # Append the proof data
            if not tactic_solution:
                if search_status == SearchStatus.FAILURE:
                    num_proofs_failed += 1
                blocks_out.append(ProofBlock(lemma_statement, search_status,
                                             [TacticInteraction("Proof.", initial_context),
                                              TacticInteraction("Admitted.",
                                                                initial_context)],
                                             original_tactics))
            else:
                num_proofs_completed += 1
                blocks_out.append(ProofBlock(lemma_statement, search_status,
                                             [TacticInteraction("Proof", initial_context)] +
                                             tactic_solution +
                                             [TacticInteraction("Qed.", empty_context)],
                                             original_tactics))
    write_html(args.output, filename, blocks_out)
    return ReportStats(filename, num_proofs, num_proofs_failed, num_proofs_completed)

def get_commands(filename : str, verbose : bool) -> List[str]:
    local_filename = prelude + "/" + filename
    loaded_commands = helper.try_load_lin(local_filename, verbose=verbose)
    if loaded_commands is None:
        fresh_commands = helper.preprocess_file_commands(
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
    for filename in extra_files:
        shutil.copy(os.path.dirname(os.path.abspath(__file__)) + "/../reports/" + filename,
                    args.output + "/" + filename)
    with open("{}/report.html".format(args.output), "w") as fout:
        fout.write(doc.getvalue())

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
def write_tactics(tactics : List[TacticInteraction],
                  region_idx : int,
                  tag : Tag, text : Text, doc : Doc):
    for t_idx, t in enumerate(tactics):
        idStr = '{}-{}'.format(region_idx, t_idx)
        with tag('span',
                 ('data-hyps', "\n".join(t.context_before.hypotheses)),
                 ('data-goal', format_goal(t.context_before.goal)),
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

class TimedCommand(NamedTuple):
    command : str
    seconds_taken : float

time_per_command = 2.
def tfs_proof_search(lemma_statement : str, coq : serapi_instance.SerapiInstance,
                     args : argparse.Namespace) -> Optional[List[str]]:
    max_time = args.search_depth * time_per_command
    def predictions() -> List[str]:
        return [pred.prediction for pred in
                predictor.predictKTactics(TacticContext(coq.prev_tactics,
                                                        coq.get_hypothesis(),
                                                        coq.get_goals()),
                                          args.search_width)]
    def total_time(commands : List[TimedCommand]) -> float:
        return sum([c.seconds_taken for c in commands])
    def get_commands(commands : List[TimedCommand]) -> List[str]:
        return [c.command for c in commands]
    def search(current_path : List[TimedCommand]) -> Optional[List[str]]:
        for prediction in predictions():
            try:
                start = time.time()
                coq.quiet = True
                old_timeout = coq.timeout
                coq.timeout = int(max_time - total_time(current_path))
                coq.run_stmt(prediction)
                coq.timeout = old_timeout
                time_taken = time.time() - start
                if completed_proof(coq):
                    return get_commands(current_path) + [prediction]
                elif total_time(current_path) + time_taken < max_time and \
                     len(current_path) < args.search_depth * 2:
                    sub_search_result = search(current_path +
                                               [TimedCommand(prediction, time_taken)])
                    if sub_search_result:
                        return sub_search_result
                coq.cancel_last()
            except (serapi_instance.CoqExn, serapi_instance.TimeoutError):
                continue
        return None
    return search([])

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

import pygraphviz as pgv
# from graphviz import Digraph

class LabeledNode(NamedTuple):
    prediction : str
    node_id : int
    full_context_before : Optional[str]
    context_before : TacticContext

def dfs_proof_search_with_graph(lemma_statement : str,
                                coq : serapi_instance.SerapiInstance,
                                args : argparse.Namespace) \
                                -> SearchResult:
    search_graph = pgv.AGraph(directed=True)
    next_node_id = 0
    def mkNode(prediction : str, full_context_before : Optional[str],
               context_before : TacticContext, **kwargs) -> LabeledNode:
        nonlocal next_node_id
        search_graph.add_node(next_node_id, label=prediction, **kwargs)
        node_obj = LabeledNode(prediction, next_node_id,
                               full_context_before, context_before)
        next_node_id += 1
        return node_obj
    def mkEdge(src : LabeledNode, dest : LabeledNode, **kwargs) -> None:
        search_graph.add_edge(src.node_id, dest.node_id, **kwargs)
    def setNodeColor(node : LabeledNode, color : str) -> None:
        node_handle = search_graph.get_node(node.node_id)
        node_handle.attr["fillcolor"] = color
        node_handle.attr["style"] = "filled"

    start_node = mkNode(serapi_instance.lemma_name_from_statement(lemma_statement), "",
                        TacticContext([], [], ""))
    def edgeToPrev(prediction : LabeledNode, current_path : List[LabeledNode]) -> None:
        if len(current_path) == 0:
            mkEdge(start_node, prediction)
        else:
            mkEdge(current_path[-1], prediction)
    def get_context() -> TacticContext:
        coq.run_stmt("Unshelve.")
        context = TacticContext(coq.prev_tactics, coq.get_hypothesis(),
                                coq.get_goals())
        coq.cancel_last()
        return context
    def get_fullcontext() -> Optional[str]:
        coq.run_stmt("Unshelve.")
        fullcontext = coq.full_context
        coq.cancel_last()
        return fullcontext
    def make_predictions() -> List[str]:
        return [pred.prediction for pred in
                predictor.predictKTactics(get_context(), args.search_width)]
    def contextInPath(full_context : str, path : List[LabeledNode]):
        return full_context in [n.full_context_before for n in path]
    hasUnexploredNode = False
    def search(current_path : List[LabeledNode]) -> Optional[List[TacticInteraction]]:
        nonlocal hasUnexploredNode
        predictions = make_predictions()
        context_before = get_context()
        predictionNodes = [mkNode(prediction, get_fullcontext(), context_before)
                           for prediction in predictions]

        for predictionNode in predictionNodes:
            edgeToPrev(predictionNode, current_path)
        for prediction, predictionNode in zip(predictions, predictionNodes):
            try:
                coq.quiet = True
                coq.run_stmt(prediction)
                context_after = get_fullcontext()
                if completed_proof(coq):
                    mkEdge(predictionNode, mkNode("QED", context_after,
                                                  TacticContext([], [], ""),
                                                  fillcolor="green", style="filled"))
                    for node in [start_node] + current_path + [predictionNode]:
                        setNodeColor(node, "green")
                    return [TacticInteraction(n.prediction, n.context_before)
                            for n in current_path + [predictionNode]]
                elif contextInPath(context_after, current_path + [predictionNode]):
                    setNodeColor(predictionNode, "orange")
                elif len(current_path) + 1 < args.search_depth:
                    sub_search_result = search(current_path + [predictionNode])
                    if sub_search_result:
                        return sub_search_result
                else:
                    hasUnexploredNode = True
                coq.cancel_last()
            except (serapi_instance.CoqExn, serapi_instance.TimeoutError):
                setNodeColor(predictionNode, "red")
                continue
        return None
    command_list = search([])
    lemma_name = serapi_instance.lemma_name_from_statement(lemma_statement)
    search_graph.draw(args.output + "/" + escape_lemma_name(lemma_name) + ".png",
                      prog="dot")
    if command_list:
        return SearchResult(SearchStatus.SUCCESS, command_list)
    elif hasUnexploredNode:
        return SearchResult(SearchStatus.INCOMPLETE, None)
    else:
        return SearchResult(SearchStatus.FAILURE, None)


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
    completed = coq.full_context == ""
    coq.cancel_last()
    return completed
