#!/usr/bin/env python3.7
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

import subprocess
import argparse
import multiprocessing
import tempfile
import functools
import sys
import contextlib
import os
import shutil
import json

import linearize_semicolons
import serapi_instance

from pathlib_revised import Path2
from sexpdata import *
from traceback import *
from util import *

from typing import Dict, Any, TextIO, List

def main():
    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description="scrape a proof")
    parser.add_argument('-o', '--output', help="output data file name", default=None)
    parser.add_argument('-j', '--threads', default=1, type=int)
    parser.add_argument('-c', '--continue', dest='cont', default=False, const=True, action='store_const')
    parser.add_argument('--hardfail', default=False, const=True, action='store_const')
    parser.add_argument('--prelude', default=".")
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument("--progress", "-P", help="show progress of files",
                        action='store_const', const=True, default=False)
    parser.add_argument('--skip-nochange-tac', default=False, const=True, action='store_const',
                    dest='skip_nochange_tac')
    parser.add_argument("--relevant-lemmas", dest="relevant_lemmas", default='local',
                        choices=['local', 'hammer', 'searchabout'])
    parser.add_argument("--no-linearize", dest="linearize", action='store_false')
    parser.add_argument('inputs', nargs="+", help="proof file name(s) (*.v)")
    args = parser.parse_args()

    try:
        with open(args.prelude + "/_CoqProject", 'r') as includesfile:
            includes = includesfile.read()
    except FileNotFoundError:
        eprint("Didn't find a _CoqProject file in prelude dir")
        includes = ""
    thispath = os.path.dirname(os.path.abspath(__file__))
    # Set up the command which runs sertop.
    coqargs = ["sertop", "--implicit"]
    with multiprocessing.Pool(args.threads) as pool:
        scrape_result_files = pool.imap_unordered(
            functools.partial(scrape_file, coqargs, args, includes),
            enumerate(args.inputs))
        with (open(args.output, 'w') if args.output
              else contextlib.nullcontext(sys.stdout)) as out:
            for idx, scrape_result_file in enumerate(scrape_result_files, start=1):
                if scrape_result_file is None:
                    eprint("Failed file {} of {}".format(idx, len(args.inputs)))
                else:
                    if args.verbose:
                        eprint("Finished file {} of {}".format(idx, len(args.inputs)))
                    with open(scrape_result_file, 'r') as f:
                        for line in f:
                            out.write(line)

from tqdm import tqdm
def scrape_file(coqargs : List[str], args : argparse.Namespace, includes : str,
                file_tuple : Tuple[int, str]) -> Optional[str]:
    sys.setrecursionlimit(4500)
    file_idx, filename = file_tuple
    full_filename = args.prelude + "/" + filename
    result_file = full_filename + ".scrape"
    temp_file = full_filename + ".scrape.partial"
    if args.cont:
        with contextlib.suppress(FileNotFoundError):
            with open(result_file, 'r') as f:
                if args.verbose:
                    eprint(f"Found existing scrape at {result_file}! Using it")
                return result_file
    try:
        if args.linearize:
            commands = serapi_instance.try_load_lin(args, file_idx, full_filename)
            if not commands:
                commands = linearize_semicolons.preprocess_file_commands(
                    args, file_idx,
                    serapi_instance.load_commands_preserve(args, 0, full_filename),
                    coqargs, includes, args.prelude, full_filename, filename, args.skip_nochange_tac)
                serapi_instance.save_lin(commands, full_filename)
        else:
            with Path2(full_filename).open(mode='r') as f:
                commands = serapi_instance.read_commands_preserve(args, file_idx, f.read())
        with serapi_instance.SerapiContext(coqargs, includes, args.prelude, args.relevant_lemmas=="hammer") as coq:
            coq.verbose = args.verbose
            try:
                with open(temp_file, 'w') as f:
                    for command in tqdm(commands, file=sys.stdout,
                                        disable=(not args.progress),
                                        position=file_idx * 2,
                                        desc="Scraping file", leave=False,
                                        dynamic_ncols=True, bar_format=mybarfmt):
                        process_statement(args, coq, command, f)
                shutil.move(temp_file, result_file)
                return result_file
            except serapi_instance.TimeoutError:
                eprint("Command in {} timed out.".format(filename))
                return temp_file
    except Exception as e:
        eprint("FAILED: In file {}:".format(filename))
        eprint(e)
        if args.hardfail or len(args.inputs) == 1:
            raise e
    return None

def process_statement(args : argparse.Namespace,
                      coq : serapi_instance.SerapiInstance, command : str,
                      result_file : TextIO) -> None:
    if not re.match("\s*[{}]\s*", command):
        if coq.proof_context:
            prev_tactics = coq.prev_tactics
            prev_hyps = coq.hypotheses
            prev_goal = coq.goals
            if args.relevant_lemmas == "local":
                relevant_lemmas = [re.sub("\n", " ", lemma) for lemma in coq.local_lemmas[:-1]]
            elif args.relevant_lemmas == "hammer":
                relevant_lemmas = coq.get_hammer_premises()
            elif args.relevant_lemmas == "searchabout":
                relevant_lemmas = coq.get_lemmas_about_head()
            else:
                assert False, args.relevant_lemmas

            result_file.write(json.dumps({"prev_tactics": prev_tactics,
                                          "prev_hyps": prev_hyps,
                                          "prev_goal": prev_goal,
                                          "relevant_lemmas": relevant_lemmas,
                                          "tactic": command}))
        else:
            result_file.write(json.dumps(command))
        result_file.write("\n")
    coq.run_stmt(command, timeout=120)

if __name__ == "__main__":
    main()
