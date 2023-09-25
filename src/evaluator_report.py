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

from evaluate_state import static_evaluators, loadEvaluatorByFile, loadEvaluatorByName
from models.state_evaluator import StateEvaluator
import coq_serapy as coq_serapy
from context_filter import get_context_filter
from coq_serapy.contexts import (TacticContext, ScrapedCommand, ScrapedTactic,
                                 strip_scraped_output)
from data import read_all_text_data
from syntax import syntax_highlight, ColoredString

from pathlib_revised import Path2
from dataclasses import dataclass
import argparse
import os
import sys
import json
from yattag import Doc
from util import stringified_percent
import subprocess
import datetime

from typing import (List, Union, Tuple, Iterable, Callable, cast, Dict, Any)

Tag = Callable[..., Doc.Tag]
Text = Callable[..., None]
Line = Callable[..., None]

details_css = ["details.css"]
details_javascript = ["eval-details.js"]
index_css = ["report.css"]
index_js = ["report.js"]
extra_files = details_css + details_javascript + index_css + index_js + ["logo.png"]

@dataclass
class FileSummary:
    filename : Path2
    close : int
    correct : int
    total : int
    num_proofs : int

def main(arg_list : List[str]) -> None:

    args = parse_arguments(arg_list)
    evaluator = get_evaluator(args)

    file_summary_results = []

    if not args.output.exists():
        args.output.makedirs()

    for idx, filename in enumerate(args.filenames):
        file_summary_results.append(generate_evaluation_details(args, idx, filename, evaluator))

    if args.generate_index:
        generate_evaluation_index(file_summary_results,
                                  evaluator.unparsed_args,
                                  args.output)

def parse_arguments(arg_list : List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        "A report testing the ability of state evaluators")
    parser.add_argument("--prelude", default=".", type=Path2)
    parser.add_argument("--context-filter", default="default")
    parser.add_argument("--no-generate-index", dest="generate_index", action='store_false')
    parser.add_argument("--output", "-o", required=True, type=Path2)
    evaluator_group = parser.add_mutually_exclusive_group(required="true")
    evaluator_group.add_argument('--weightsfile', default=None, type=Path2)
    evaluator_group.add_argument('--evaluator', choices=list(static_evaluators.keys()),
                        default=None)
    parser.add_argument('filenames', nargs="+", help="proof file name (*.v)", type=Path2)
    return parser.parse_args(arg_list)

def get_evaluator(args) -> StateEvaluator:
    evaluator : StateEvaluator
    if args.weightsfile:
        evaluator = loadEvaluatorByFile(args.weightsfile)
    else:
        evaluator = loadEvaluatorByName(args.evaluator)

    return evaluator

@dataclass
class TacticInteraction:
    tactic : str
    context_before : TacticContext

@dataclass
class VernacBlock:
    commands : List[str]

@dataclass
class ProofBlock:
    lemma_statement : str
    proof_interactions : List[TacticInteraction]

def get_blocks(interactions : List[ScrapedCommand]) -> List[Union[VernacBlock, ProofBlock]]:
    def generate() -> Iterable[Union[VernacBlock, ProofBlock]]:
        in_proof = False
        cur_lemma = ""
        interaction_buffer : List[ScrapedCommand] = []
        for interaction in interactions:
            if isinstance(interaction, ScrapedTactic):
                if not in_proof:
                    yield VernacBlock(cast(List[str], interaction_buffer[:-1]))
                    cur_lemma = cast(str, interaction_buffer[-1])
                    interaction_buffer = []
                    in_proof = True
            else:
                assert isinstance(interaction, str)
                if in_proof:
                    yield ProofBlock(cur_lemma, cast(List[TacticInteraction], interaction_buffer))
                    interaction_buffer = []
                    in_proof = False
            interaction_buffer.append(interaction)
    return list(generate())

def generate_evaluation_details(args : argparse.Namespace, idx : int,
                                filename : Path2, evaluator : StateEvaluator) -> FileSummary:
    scrape_path = args.prelude / filename.with_suffix(".v.scrape")
    interactions = list(read_all_text_data(scrape_path))
    context_filter = get_context_filter(args.context_filter)
    json_rows : List[Dict[str, Any]] = []

    num_points = 0
    num_close = 0
    num_correct = 0
    num_proofs = 0

    doc, tag, text, line = Doc().ttl()

    def write_highlighted(vernac : str) -> None:
        nonlocal text
        nonlocal tag
        substrings = syntax_highlight(vernac)

        for substring in substrings:
            if isinstance(substring, ColoredString):
                with tag('span', style=f'color:{substring.color}'):
                    text(substring.contents)
            else:
                text(substring)
    def write_vernac(block : VernacBlock):
        nonlocal tag
        for command in block.commands:
            with tag('code', klass='plaincommand'):
                write_highlighted(command)

    def generate_proof_evaluation_details(block : ProofBlock, region_idx : int):
        nonlocal num_proofs
        nonlocal num_close
        nonlocal num_correct
        nonlocal json_rows
        num_proofs += 1

        nonlocal num_points

        distanced_tactics = label_distances(block.proof_interactions)

        proof_length = len(distanced_tactics)
        num_points += proof_length

        with tag('div', klass='region'):
            nonlocal evaluator
            for idx, (interaction, distance_from_end) in enumerate(distanced_tactics, 1):
                if interaction.tactic.strip() in ["Proof.", "Qed.", "Defined."]:
                    with tag('code', klass='plaincommand'):
                        write_highlighted(interaction.tactic.strip("\n"))
                    doc.stag('br')
                else:
                    predicted_distance_from_end = evaluator.scoreState(interaction.context_before)
                    grade = grade_prediction(distance_from_end, predicted_distance_from_end)
                    if grade == "goodcommand":
                        num_correct += 1
                    elif grade == "okaycommand":
                        num_close += 1

                    num_points += 1
                    json_rows.append({"lemma": block.lemma_statement,
                                      "hyps": interaction.context_before.hypotheses,
                                      "goal": interaction.context_before.goal,
                                      "actual-distance": distance_from_end,
                                      "predicted-distance": predicted_distance_from_end,
                                      "grade": grade})
                    with tag('span',
                             ('data-hyps', "\n".join(interaction.context_before.hypotheses)),
                             ('data-goal', interaction.context_before.goal),
                             ('data-actual-distance', str(distance_from_end)),
                             ('data-predicted-distance', str(predicted_distance_from_end)),
                             ('data-region', region_idx),
                             ('data-index', idx),
                             klass='tactic'), \
                             tag('code', klass=grade):
                        text(interaction.tactic)
                    doc.stag('br')


    def write_lemma_button(lemma_statement : str, region_idx : int):
        nonlocal tag
        nonlocal text
        lemma_name = \
            coq_serapy.lemma_name_from_statement(lemma_statement)
        with tag('button', klass='collapsible', id=f'collapsible-{region_idx}'):
            with tag('code', klass='buttontext'):
                write_highlighted(lemma_statement.strip())

    def grade_prediction(correct_number : int, predicted_number : float) -> str:
        distance = abs(correct_number - predicted_number)
        if distance < 1:
            return "goodcommand"
        elif distance < 5:
            return "okaycommand"
        else:
            return "badcommand"

    with tag('html'):
        header(tag, doc, text, details_css, details_javascript, "Proverbot9001 Report")
        with tag('body', onload='init()'), tag('pre'):
            for idx, block in enumerate(get_blocks(interactions)):
                if isinstance(block, VernacBlock):
                    write_vernac(block)
                else:
                    assert isinstance(block, ProofBlock)
                    write_lemma_button(block.lemma_statement, idx)
                    generate_proof_evaluation_details(block, idx)

    base = Path2(os.path.dirname(os.path.abspath(__file__)))
    for extra_filename in extra_files:
        (base.parent / "reports" / extra_filename).copyfile(args.output / extra_filename)

    with (args.output / filename.with_suffix(".html").name).open(mode='w') as fout:
        fout.write(doc.getvalue())

    with (args.output / filename.with_suffix(".json").name).open(mode='w') as fout:
        for row in json_rows:
            fout.write(json.dumps(row))
            fout.write("\n")

    return FileSummary(filename, num_close, num_correct, num_points, num_proofs)

def label_distances(tactics : List[TacticInteraction]) -> List[Tuple[TacticInteraction, int]]:
    path_segments : List[List[TacticInteraction]] = [[]]
    closed_distances : List[int] = [0, 0]
    result : List[List[Tuple[TacticInteraction, int]]] = [[], []]

    def open_goal():
        nonlocal path_segments
        nonlocal closed_distances
        nonlocal result
        path_segments.append([])
        closed_distances.append(0)
        result.append([])

    def close_goal():
        nonlocal path_segments
        nonlocal closed_distances
        nonlocal result
        last_segment = path_segments.pop()
        last_distance = closed_distances.pop()
        closed_tacs = list(reversed([(tac, distance) for (distance, tac) in
                                     enumerate(reversed(last_segment), last_distance + 1)]))
        already_closed_tacs = result.pop()
        result[-1] += closed_tacs + already_closed_tacs
        closed_distances[-1] += last_distance + len(last_segment)

    for interaction in tactics:
        if interaction.tactic.strip() == "{":
            open_goal()
        elif interaction.tactic.strip() == "}":
            close_goal()
        elif interaction.tactic.strip() == "Qed." or \
             interaction.tactic.strip() == "Defined.":
            close_goal()
            return result[-1] + [(interaction, 0)]
        else:
            path_segments[-1].append(interaction)
    assert len(path_segments) == 1
    close_goal()
    return result[-1]

def header(tag : Tag, doc : Doc, text : Text, css : List[str],
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
def generate_evaluation_index(file_summary_results : List[FileSummary],
                              unparsed_args : List[str],
                              output_dir : Path2):
    doc, tag, text, line = Doc().ttl()
    with tag('html'):
        header(tag, doc, text, index_css, index_js,
                    "Proverbot State Evaluation Report")
        with tag("body"):
            total_states  = sum([result.total for result in file_summary_results])
            total_correct = sum([result.correct for result in file_summary_results])
            total_close = sum([result.close for result in file_summary_results])
            with tag('h2'):
                text("States Correctly Scored: {}% ({}/{})"
                     .format(stringified_percent(total_correct, total_states),
                             total_correct, total_states))
            with tag('img',
                     ('src', 'logo.png'),
                     ('id', 'logo')):
                pass
            with tag('h5'):
                cur_commit = subprocess.check_output(["git show --oneline | head -n 1"],
                                                     shell=True).decode('utf-8').strip()
                diff_path = output_dir / "diff.txt"
                subprocess.run([f"git diff HEAD > {str(diff_path)}"], shell=True)
                subprocess.run([f"git status >> {str(diff_path)}"], shell=True)
                with tag('a', href=str("diff.txt")):
                    text('Commit: {}'.format(cur_commit))
            with tag('h5'):
                cur_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                text('Run on: {}'.format(cur_date))
            # with tag('ul'):
                # for k, v in options:
                #     if k == 'filenames':
                #         continue
                #     elif not v:
                #         continue
                #     with tag('li'):
                #         text("{}: {}".format(k, v))
            with tag('h4'):
                text("{} files processed".format(len(file_summary_results)))
            with tag('table'):
                with tag('tr', klass="header"):
                    line('th', 'Filename')
                    line('th', 'Number of States in File')
                    line('th', 'Correct (e < 1)')
                    line('th', 'Close (e < 5)')
                    line('th', 'Details')
                sorted_files = sorted(file_summary_results,
                                      key=lambda fresult:fresult.total,
                                      reverse=True)
                for fresult in sorted_files:
                    if fresult.total == 0:
                        continue
                    with tag('tr'):
                        line('td', fresult.filename.name)
                        line('td', str(fresult.total))
                        line('td', stringified_percent(fresult.correct, fresult.total) + "%")
                        line('td', stringified_percent(fresult.close, fresult.total) + "%")
                        with tag('td'):
                            with tag('a',
                                     href=str(fresult.filename
                                                  .with_suffix(".html").name)):
                                text("Details")
                with tag('tr'):
                    line('td', "Total")
                    line('td', str(total_states))
                    line('td', stringified_percent(total_correct, total_states))
                    line('td', stringified_percent(total_close, total_states))
            text(f'Trained as: {" ".join(unparsed_args)}')
            doc.stag('br')
            text(f'Reported as: {" ".join(sys.argv)}')

    with (output_dir / "report.html").open("w") as fout:
        fout.write(doc.getvalue())

    pass
