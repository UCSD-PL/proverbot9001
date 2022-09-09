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
import os
import sys
import re
import datetime
import itertools
import json
import subprocess
from pathlib import Path
from shutil import copyfile

from yattag import Doc
from tqdm import tqdm

from util import stringified_percent, escape_filename, safe_abbrev, escape_lemma_name
import util

from typing import (List, Tuple, Sequence, Dict, Callable,
                    Any, Iterable, Optional)

from models.tactic_predictor import TacticPredictor
import data

from search_results import (ReportStats, SearchStatus, SearchResult, DocumentBlock,
                            VernacBlock, ProofBlock, TacticInteraction)
from search_worker import get_file_jobs, get_predictor, project_dicts_from_args
import coq_serapy
from coq_serapy.contexts import ScrapedTactic, Obligation
import multi_project_report

index_css = ["report.css"]
index_js = ["report.js"]
extra_files = index_css + index_js + ["logo.png"]

details_css = "details.css"
details_javascript = "search-details.js"

Tag = Callable[..., Doc.Tag]
Text = Callable[..., None]
Line = Callable[..., None]

unnamed_goal_number: int = 0

def generate_report(args: argparse.Namespace, predictor: TacticPredictor,
                    project_dicts: List[Dict[str, Any]], time_taken: datetime.timedelta) -> None:
    base = Path(os.path.dirname(os.path.abspath(__file__)))

    if not args.output_dir.exists():
        os.makedirs(str(args.output_dir))
    for project_dict in tqdm([project_dict for project_dict in project_dicts
                              if len(project_dict["test_files"]) > 0],
                             desc="Report Projects"):
        generate_project_report(args, predictor, project_dict, time_taken)
    if len(project_dicts) > 1:
        multi_project_report.multi_project_index(args.output_dir)

def generate_project_report(args: argparse.Namespace, predictor: TacticPredictor,
                            project_dict: Dict[str, Any], time_taken: datetime.timedelta) \
                            -> None:
    model_name = dict(predictor.getOptions())["predictor"]
    stats: List[ReportStats] = []
    for filename in [details_css, details_javascript]:
        destpath = args.output_dir / project_dict["project_name"] / filename
        if not destpath.exists():
            srcpath = base.parent / 'reports' / filename
            copyfile(srcpath, destpath)
    for filename in tqdm(project_dict["test_files"], desc="Report Files", leave=False):
        file_solutions = []
        output_file_prefix = args.output_dir / project_dict["project_name"] / \
              (safe_abbrev(Path(filename),
                                [Path(path) for path in
                                 project_dict["test_files"]]))
        source_file = args.prelude / project_dict["project_name"] / filename
        try:
            with (Path(str(output_file_prefix) + "-proofs.txt")).open('r') as f:
                for line in f:
                    job, sol = json.loads(line)
                    file_solutions.append((job, SearchResult.from_dict(sol)))
        except FileNotFoundError:
            lemmas = get_file_jobs(args, project_dict["project_name"], filename)
            assert len(lemmas) == 0, lemmas
            stats.append(ReportStats(filename, 0, 0, 0))
            continue
        for (sol_project, sol_filename, _, _), _ in file_solutions:
            assert sol_project == project_dict["project_name"], \
              (project, project_dict["project_name"])
            assert sol_filename == filename, \
              (sol_filename, filename)
        blocks = blocks_from_scrape_and_sols(
            source_file,
            [(lemma_stmt, module_name, sol)
            for (project, filename, module_name, lemma_stmt), sol
            in file_solutions])

        write_solution_vfile(args, output_file_prefix.with_suffix(".v"),
                             model_name, blocks)
        write_html(args, output_file_prefix.with_suffix(".html"),
                   filename, blocks)
        write_csv(args, output_file_prefix.with_suffix(".csv"), blocks)
        stats.append(stats_from_blocks(blocks, str(filename)))
    produce_index(args, predictor,
                  args.output_dir / project_dict["project_name"],
                  stats, time_taken)

def blocks_from_scrape_and_sols(
        src_filename: Path,
        lemma_statements_done: List[Tuple[str, str, SearchResult]]
        ) -> List[DocumentBlock]:

    interactions = data.read_all_text_data(
        src_filename.with_suffix(".v.scrape"))

    def lookup(module: str, lemma_stmt: str) -> Optional[SearchResult]:
        for lstmt, lmod, lresult in lemma_statements_done:
            if (lmod == module and
                    coq_serapy.kill_comments(lstmt).strip()
                    == coq_serapy.kill_comments(lemma_stmt).strip()):
                return lresult
        return None

    def generate():
        cur_lemma_stmt = ""
        unique_lemma_stmt = ""

        sm_stack = coq_serapy.initial_sm_stack(src_filename)

        tactics_interactions_batch: List[TacticInteraction] = []
        vernac_cmds_batch: List[str] = []

        in_proof = False
        obl_num = 0
        last_program_statement = ""

        def yield_proof():
            nonlocal sm_stack
            nonlocal tactics_interactions_batch
            nonlocal cur_lemma_stmt
            nonlocal unique_lemma_stmt
            nonlocal in_proof
            nonlocal obl_num
            nonlocal last_program_statement

            sm_prefix = coq_serapy.sm_prefix_from_stack(sm_stack)
            batch_without_brackets = [t for t in tactics_interactions_batch
                                      if t.tactic.strip() != "{" and
                                      t.tactic.strip() != "}"]
            result = lookup(sm_prefix, unique_lemma_stmt)
            if result is None:
                return ProofBlock(cur_lemma_stmt, sm_prefix,
                                  SearchStatus.SKIPPED, [],
                                  batch_without_brackets)
            else:
                return ProofBlock(cur_lemma_stmt, sm_prefix,
                                  result.status, result.commands,
                                  batch_without_brackets)
            tactics_interactions_batch = []

        for interaction in interactions:
            if in_proof and isinstance(interaction, str):
                in_proof = False
                yield yield_proof()
            elif in_proof and isinstance(interaction, ScrapedTactic):
                tactics_interactions_batch.append(
                    interaction_from_scraped(interaction))
            elif isinstance(interaction, ScrapedTactic):
                assert not in_proof
                cur_lemma_stmt = vernac_cmds_batch[-1]
                if re.match(r"\s*Next\s+Obligation\s*\.\s*",
                            coq_serapy.kill_comments(
                                cur_lemma_stmt).strip()):
                    unique_lemma_stmt = \
                      f"{last_program_statement} Obligation {obl_num}."
                    obl_num += 1
                else:
                    unique_lemma_stmt = cur_lemma_stmt
                yield VernacBlock(vernac_cmds_batch[:-1])
                vernac_cmds_batch = []
                tactics_interactions_batch = []
                tactics_interactions_batch.append(
                    interaction_from_scraped(interaction))
                in_proof = True
            if isinstance(interaction, str):
                sm_stack = coq_serapy.update_sm_stack(sm_stack, interaction)
                vernac_cmds_batch.append(interaction)
                if re.match(r"\s*Program\s+.*",
                            coq_serapy.kill_comments(interaction).strip()):
                    last_program_statement = interaction
                    obl_num = 0
        if in_proof:
            yield yield_proof()
        pass
    blocks = list(generate())
    return blocks


def interaction_from_scraped(s: ScrapedTactic) -> TacticInteraction:
    return TacticInteraction(s.tactic, s.context)


def write_solution_vfile(args: argparse.Namespace, output_filename: Path,
                         model_name: str,
                         doc_blocks: List[DocumentBlock]):
    with output_filename.open('w') as sfile:
        for k, v in [("search-width", args.search_width),
                     ("search-depth", args.search_depth),
                     ("model", model_name)]:
            print(f"(* {k}: {v} *)", file=sfile)

        for block in doc_blocks:
            if isinstance(block, VernacBlock):
                for cmd in block.commands:
                    print(cmd, file=sfile, end='')
            else:
                assert isinstance(block, ProofBlock)
                print(block.lemma_statement, end='', file=sfile)
                if block.predicted_tactics:
                    for tac in block.predicted_tactics:
                        print(tac.tactic, file=sfile)
                else:
                    for tac in block.original_tactics:
                        print(tac.tactic, file=sfile)

        pass


def write_csv(args: argparse.Namespace,
              output_filename: Path,
              doc_blocks: List[DocumentBlock]):
    with output_filename.open('w', newline='') as csvfile:
        for k, v in vars(args).items():
            csvfile.write("# {}: {}\n".format(k, v))

        rowwriter = csv.writer(csvfile, lineterminator=os.linesep)
        for block in doc_blocks:
            if isinstance(block, ProofBlock):
                rowwriter.writerow([block.lemma_statement.strip(),
                                    block.status,
                                    len(block.original_tactics)])


def write_html(args: argparse.Namespace,
               output_file: Path, filename: Path,
               doc_blocks: List[DocumentBlock]) -> None:
    global unnamed_goal_number
    unnamed_goal_number = 0
    doc, tag, text, line = Doc().ttl()
    with tag('html'):
        html_header(tag, doc, text, [details_css], [details_javascript],
                    "Proverbot Detailed Report for {}".format(str(filename)))
        with tag('body', onload='init()'), tag('pre'):
            for block_idx, block in enumerate(doc_blocks):
                if isinstance(block, VernacBlock):
                    write_commands(block.commands, tag, text, doc)
                else:
                    assert isinstance(block, ProofBlock)
                    status_klass = classFromSearchStatus(block.status)
                    write_lemma_button(block.lemma_statement, block.module,
                                       status_klass, tag, text)
                    with tag('div', klass='region'):
                        with tag('div', klass='predicted'):
                            write_tactics(args, block.predicted_tactics,
                                          block_idx,
                                          tag, text, doc)
                        with tag('div', klass='original'):
                            write_tactics(args, block.original_tactics,
                                          block_idx,
                                          tag, text, doc)
    with output_file.open('w') as fout:
        fout.write(doc.getvalue())


def write_lemma_button(lemma_statement: str, module: Optional[str],
                       status_klass: str, tag: Tag, text: Text):
    global unnamed_goal_number
    lemma_name = \
        coq_serapy.lemma_name_from_statement(lemma_statement)
    if module:
        module_prefix = escape_lemma_name(module)
    else:
        module_prefix = ""
    if lemma_name == "":
        unnamed_goal_number += 1
        fullname = module_prefix + lemma_name + str(unnamed_goal_number)
    else:
        fullname = module_prefix + lemma_name
    if status_klass != "skipped":
        with tag('button', klass='collapsible {}'.format(status_klass),
                 onmouseover="hoverLemma(\"{}\")".format(fullname),
                 onmouseout="unhoverLemma(\"{}\")".format(fullname)):
            with tag('code', klass='buttontext'):
                text(lemma_statement.strip())
    else:
        with tag('button', klass='collapsible {}'.format(status_klass)):
            with tag('code', klass='buttontext'):
                text(lemma_statement.strip())


def write_commands(commands: List[str], tag: Tag, text: Text, doc: Doc):
    for cmd in commands:
        with tag('code', klass='plaincommand'):
            text(cmd.strip("\n"))
        doc.stag('br')


def escape_quotes(term: str):
    return re.sub("\"", "\\\"", term)


def subgoal_to_string(args: argparse.Namespace, sg: Obligation) -> str:
    return "(\"" + escape_quotes(sg.goal[:args.max_print_term]) + "\", (\"" + \
        "\",\"".join([escape_quotes(hyp[:args.max_print_term]) for hyp in
                      sg.hypotheses[:args.max_print_hyps]]) + "\"))"


def write_tactics(args: argparse.Namespace,
                  tactics: List[TacticInteraction],
                  region_idx: int,
                  tag: Tag, text: Text, doc: Doc):
    for t_idx, t in enumerate(tactics):
        idStr = '{}-{}'.format(region_idx, t_idx)
        subgoals_str = "(" + ",".join([subgoal_to_string(args, subgoal)
                                       for subgoal in
                                       t.context_before.all_goals[
                                           :args.max_print_subgoals]]) + ")"
        with tag('span',
                 ('data-subgoals', subgoals_str),
                 id='command-{}'.format(idStr),
                 onmouseover='hoverTactic("{}")'.format(idStr),
                 onmouseout='unhoverTactic()'):
            with tag('code', klass='plaincommand'):
                text(t.tactic.strip())
            doc.stag('br')


def classFromSearchStatus(status: SearchStatus) -> str:
    if status == SearchStatus.SUCCESS:
        return 'good'
    elif status == SearchStatus.INCOMPLETE:
        return 'okay'
    elif status == SearchStatus.SKIPPED:
        return 'skipped'
    elif status == SearchStatus.FAILURE:
        return 'bad'
    elif status == SearchStatus.CRASHED:
        return 'bad'
    else:
        assert False


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

def write_summary_html(filename : Path,
                       options : Sequence[Tuple[str, str]],
                       unparsed_args : List[str],
                       cur_commit : str, cur_date : datetime.datetime,
                       weights_hash: str,
                       individual_stats : List[ReportStats],
                       combined_stats : ReportStats,
                       time_taken: datetime.timedelta) -> None:
    def report_header(tag : Any, doc : Doc, text : Text) -> None:
        html_header(tag, doc, text,index_css, index_js,
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
            with tag('h2'):
                text(f"Time taken: {str(time_taken)}")
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
                            with tag('a', href=
                                     safe_abbrev(Path(fresult.filename),
                                                 [Path(result.filename) for result
                                                  in sorted_rows]) + ".html"):
                                text("Details")
                with tag('tr'):
                    line('td', "Total")
                    line('td', str(combined_stats.num_proofs))
                    line('td', stringified_percent(combined_stats.num_proofs_completed,
                                                   combined_stats.num_proofs))
                    line('td', stringified_percent(combined_stats.num_proofs -
                                                   (combined_stats.num_proofs_completed +
                                                    combined_stats.num_proofs_failed),
                                                   combined_stats.num_proofs))
                    line('td', stringified_percent(combined_stats.num_proofs_failed,
                                                   combined_stats.num_proofs))
            text(f'Trained as: {unparsed_args}')
            doc.stag('br')
            text(f"Reported as: {sys.argv}")
            doc.stag('br')
            text(f"Weights hash: {weights_hash}")

    with filename.open("w") as fout:
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

def write_summary(args : argparse.Namespace, report_dir: Path,
                  options : Sequence[Tuple[str, str]],
                  unparsed_args : List[str],
                  cur_commit : str, cur_date : datetime.datetime,
                  weights_hash: str,
                  individual_stats : List[ReportStats],
                  time_taken: datetime.timedelta) -> None:
    combined_stats = combine_file_results(individual_stats)
    write_summary_html(report_dir / "index.html",
                       options, unparsed_args,
                       cur_commit, cur_date, weights_hash,
                       individual_stats, combined_stats,
                       time_taken)
    write_summary_csv("{}/report.csv".format(report_dir), combined_stats, options)
    write_proof_summary_csv(str(report_dir), [str(s.filename) for s in individual_stats])
    base = Path(os.path.abspath(__file__)).parent.parent / "reports"
    for filename in extra_files:
        copyfile(base / filename, report_dir / filename)
def write_proof_summary_csv(output_dir : str, filenames : List[str]):
    with open('{}/proofs.csv'.format(output_dir), 'w') as fout:
        fout.write("lemma,status,prooflength\n")
        for filename in filenames:
            try:
                with open("{}/{}.csv".format(
                    output_dir, safe_abbrev(Path(filename),
                                            [Path(path) for path in filenames])), 'r') \
                    as fin:
                    fout.writelines(fin)
            except FileNotFoundError:
                pass

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

def read_stats_from_csv(output_dir : str, vfilename : str) -> \
    Tuple[argparse.Namespace, ReportStats]:
    num_proofs = 0
    num_proofs_failed = 0
    num_proofs_completed = 0
    with open("{}/{}.csv".format(output_dir, escape_filename(str(vfilename))),
              'r', newline='') as csvfile:
        saved_args, rest_iter = read_csv_options(csvfile)
        rowreader = csv.reader(rest_iter, lineterminator=os.linesep)
        for row in rowreader:
            num_proofs += 1
            if row[1] == "SUCCESS":
                num_proofs_completed += 1
            elif row[1] == "FAILURE":
                num_proofs_failed += 1
            else:
                assert row[1] == "INCOMPLETE", row
    return saved_args, ReportStats(str(vfilename),
                                   num_proofs, num_proofs_failed, num_proofs_completed)

def combine_file_results(stats : List[ReportStats]) -> ReportStats:
    return ReportStats("",
                       sum([s.num_proofs for s in stats]),
                       sum([s.num_proofs_failed for s in stats]),
                       sum([s.num_proofs_completed for s in stats]))

def produce_index(args: argparse.Namespace, predictor: TacticPredictor,
                  report_dir: Path,
                  report_stats: List[ReportStats], time_taken: datetime.timedelta) -> None:
    predictorOptions = predictor.getOptions()
    commit, date, weightshash = get_metadata(args)
    write_summary(args, report_dir,
                  predictorOptions +
                  [("report type", "search"),
                   ("search width", args.search_width),
                   ("search depth", args.search_depth)],
                  predictor.unparsed_args,
                  commit, date, weightshash, report_stats,
                  time_taken)


def stats_from_blocks(blocks: List[DocumentBlock], vfilename: str) \
      -> ReportStats:
    num_proofs = 0
    num_proofs_failed = 0
    num_proofs_completed = 0
    for block in blocks:
        if isinstance(block, ProofBlock):
            if block.status != SearchStatus.SKIPPED:
                num_proofs += 1
            if block.status == SearchStatus.SUCCESS:
                num_proofs_completed += 1
            elif block.status == SearchStatus.FAILURE:
                num_proofs_failed += 1
    return ReportStats(vfilename, num_proofs,
                       num_proofs_failed, num_proofs_completed)


def get_metadata(args: argparse.Namespace) -> Tuple[str, datetime.datetime, str]:
    cur_commit = subprocess.check_output(["git show --oneline | head -n 1"],
                                         shell=True).decode('utf-8').strip()
    cur_date = datetime.datetime.now()
    if args.weightsfile:
        weights_hash = str(subprocess.check_output(
            ["sha256sum", args.weightsfile]))
    else:
        weights_hash = ""
    return cur_commit, cur_date, weights_hash

def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("report_dir", type=Path)
    arg_parser.add_argument("-p", "--project", type=str, default=None)
    arg_parser.add_argument("-i", "--project-index-only", action="store_true")
    top_args = arg_parser.parse_args()
    assert not (top_args.project and top_args.project_index_only)

    with open(top_args.report_dir / "args.json", 'r') as f:
        args_dict = json.loads(f.read())
        args = argparse.Namespace()
        for k, v in args_dict.items():
            if k in ["output_dir", "prelude"]:
                setattr(args, k, Path(eval(v)))
            else:
                setattr(args, k, eval(v))

    predictor = get_predictor(arg_parser, args)
    project_dicts = project_dicts_from_args(args)
    with open(top_args.report_dir / "time_so_far.txt", 'r') as f:
        time_taken = util.read_time_taken(f.read())
    if top_args.project:
        matching_project_dicts = [project_dict for project_dict in project_dicts
                                  if project_dict["project_name"] == top_args.project]
        assert len(matching_project_dicts) != 0, \
          f"No project matches project name {top_args.project}"
        assert len(matching_project_dicts) == 1, \
          f"Multiple projects match project name {top_args.project}"
        generate_project_report(args, predictor, matching_project_dicts[0], time_taken)
    elif top_args.project_index_only:
        assert len(project_dicts) > 1
        multi_project_report.multi_project_index(args.output_dir)
    else:
        generate_report(args, predictor, project_dicts, time_taken)

if __name__ == "__main__":
    main()
