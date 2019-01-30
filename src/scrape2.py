#!/usr/bin/env python3

import subprocess
import argparse
import multiprocessing
import tempfile
import functools

from helper import *
import linearize_semicolons
import serapi_instance

from sexpdata import *
from traceback import *
from format import format_context, format_tactic

from typing import Dict, Any, TextIO

def main():
    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description="scrape a proof")
    parser.add_argument('-o', '--output', help="output data file name", default=None)
    parser.add_argument('-j', '--threads', default=1, type=int)
    parser.add_argument('--prelude', default=".")
    parser.add_argument('--debug', default=False, const=True, action='store_const')
    parser.add_argument('inputs', nargs="+", help="proof file name(s) (*.v)")
    args = parser.parse_args()


    try:
        with open("{}/_CoqProject".format(args.prelude), 'r') as coqproject:
            includes = coqproject.read().strip()
    except FileNotFoundError:
        includes=subprocess.Popen(['make', '-C', args.prelude, 'print-includes'],
                                  stdout=subprocess.PIPE).communicate()[0]\
                           .strip().decode('utf-8')

    thispath = os.path.dirname(os.path.abspath(__file__))
    # Set up the command which runs sertop.
    coqargs = ["{}/../coq-serapi/sertop.native".format(thispath),
               "--prelude={}/../coq".format(thispath)]

    with multiprocessing.Pool(args.threads) as pool:
        scrape_result_files = pool.imap_unordered(
            functools.partial(scrape_file, coqargs, args.debug, includes, args.prelude),
            args.inputs)
        with open(args.output or "scrape.txt", 'w') as out:
            for idx, scrape_result_file in enumerate(scrape_result_files, start=1):
                print("Finished file {} of {}".format(idx, len(args.inputs)))
                with open(scrape_result_file, 'r') as f:
                    for line in f:
                        out.write(line)

def scrape_file(coqargs : List[str], debug : bool, includes : str,
                prelude : str, filename : str) -> str:
    full_filename = prelude + "/" + filename
    commands = try_load_lin(full_filename)
    if not commands:
        commands = lift_and_linearize(load_commands(full_filename),
                                      coqargs, includes, prelude, full_filename, debug=debug)
        save_lin(commands, full_filename)

    with serapi_instance.SerapiContext(coqargs, includes, prelude) as coq:
        result_file = full_filename + ".scrape"
        coq.debug = debug
        try:
            with open(result_file, 'w') as f:
                for command in commands:
                    process_statement(coq, command, f)
        except serapi_instance.TimeoutError:
            print("Command in {} timed out.".format(filename))
        return result_file

def process_statement(coq : serapi_instance.SerapiInstance, command : str,
                      result_file : TextIO) -> None:
    if coq.proof_context:
        prev_tactics = coq.prev_tactics
        prev_hyps = coq.get_hypothesis()
        prev_goal = coq.get_goals()
        result_file.write(format_context(prev_tactics, prev_hyps, prev_goal, ""))
        result_file.write(format_tactic(command))
    else:
        subbed_command = re.sub(r"\n", r"\\n", command)
        result_file.write(subbed_command+"\n-----\n")

    coq.run_stmt(command)

if __name__ == "__main__":
    main()
