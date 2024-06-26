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
import os.path

import linearize_semicolons
import coq_serapy as serapi_instance

from util import eprint, mybarfmt

from typing import TextIO, List, Tuple, Optional
from tqdm import tqdm


def main():
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
    parser.add_argument("--sertop-flags", default=None, type=str)
    parser.add_argument("--text-encoding", default='utf-8', type=str)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument('inputs', nargs="+", help="proof file name(s) (*.v)")
    args = parser.parse_args()

    # Set up the command which runs sertop.
    coqargs = ["sertop", "--implicit"]
    if args.sertop_flags:
        coqargs += args.sertop_flags.split()
    if os.path.splitext(args.inputs[0])[1] == ".json":
        with open(args.inputs[0]) as f:
            file_jobs = [filename for project_dict in json.load(f)
                         for filename in
                         ((project_dict["test_files"] + project_dict["train_files"])
                          if args.split == "all" else
                          (project_dict["test_files"] if args.split == "test" else
                           project_dict["train_files"]))]
    else:
        file_jobs = args.inputs
    tasks = [(idx % args.threads, filename) for (idx, filename)
             in enumerate(file_jobs)]
    with multiprocessing.Pool(args.threads) as pool:
        scrape_result_files = pool.imap_unordered(
            functools.partial(scrape_file, coqargs, args, file_jobs),
            tasks)
        with (open(args.output, 'w') if args.output
              else contextlib.nullcontext(sys.stdout)) as out:
            for idx, scrape_result_file in enumerate(scrape_result_files,
                                                     start=1):
                if scrape_result_file is None:
                    eprint("Failed file {} of {}"
                           .format(idx, len(file_jobs)))
                else:
                    if args.verbose:
                        eprint("Finished file {} of {}"
                               .format(idx, len(file_jobs)))
                    with open(scrape_result_file, 'r') as f:
                        for line in f:
                            out.write(line)


def scrape_file(coqargs: List[str], args: argparse.Namespace, all_file_jobs: List[str],
                file_tuple: Tuple[int, str]) -> Optional[str]:
    sys.setrecursionlimit(4500)
    file_idx, filename = file_tuple
    full_filename = args.prelude + "/" + filename

    components = filename.split("/")
    prelude = args.prelude
    for component_idx in reversed(range(len(components) - 1)):
        possible_project_path = os.path.join(*([args.prelude] +
                                               components[:component_idx+1]))
        if os.path.exists(os.path.join(possible_project_path, "_CoqProject")):
             prelude = possible_project_path
             break

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
            commands = linearize_semicolons.get_linearized(args, coqargs, file_idx, filename)
        else:
            commands = serapi_instance.load_commands_preserve(
                args, file_idx, str(full_filename))
        if args.switch:
            serapi_instance.set_switch(args.switch)
        with serapi_instance.SerapiContext(
                coqargs,
                serapi_instance.get_module_from_filename(filename),
                prelude, args.relevant_lemmas == "hammer") as coq:
            coq.verbose = args.verbose
            try:
                with open(temp_file, 'w') as f:
                    for command in tqdm(commands, file=sys.stdout,
                                        disable=(not args.progress),
                                        position=file_idx * 2,
                                        desc="Scraping file", leave=False,
                                        dynamic_ncols=True,
                                        bar_format=mybarfmt):
                        process_statement(args, coq, command, f)
                    print(json.dumps("\"(* End of File *)\""), file=f)
                shutil.move(temp_file, result_file)
                return result_file
            except serapi_instance.CoqTimeoutError:
                eprint("Command in {} timed out.".format(filename))
                return temp_file
    except Exception as e:
        eprint("FAILED: In file {}:".format(filename))
        eprint(e)
        if args.hardfail or len(all_file_jobs) == 1 or args.hardfail_scrape:
            raise e
    return None


def process_statement(args: argparse.Namespace,
                      coq: serapi_instance.SerapiInstance, command: str,
                      result_file: TextIO) -> None:
    if coq.proof_context:
        prev_tactics = coq.prev_tactics
        context = coq.proof_context
        if args.relevant_lemmas == "local":
            relevant_lemmas = [re.sub("\n", " ", lemma)
                               for lemma in coq.local_lemmas[:-1]]
        elif args.relevant_lemmas == "hammer":
            relevant_lemmas = coq.get_hammer_premises()
        elif args.relevant_lemmas == "searchabout":
            relevant_lemmas = coq.get_lemmas_about_head()
        else:
            assert False, args.relevant_lemmas

        result_file.write(json.dumps({"relevant_lemmas": relevant_lemmas,
                                      "prev_tactics": prev_tactics,
                                      "context": context.to_dict(),
                                      "tactic": command}))
    else:
        result_file.write(json.dumps(command))
    result_file.write("\n")

    coq.run_stmt(command, timeout=600)


if __name__ == "__main__":
    main()
