#!/usr/bin/env python3.7

import argparse
import datetime
import os
import os.path
import queue
import re
import subprocess
import sys
import threading

# This dependency is in pip, the python package manager
from sexpdata import *
from timer import TimerBucket
from traceback import *
from compcert_linearizer_failures import compcert_failures

import serapi_instance
from serapi_instance import (AckError, CompletedError, CoqExn,
                             BadResponse, TimeoutError, ParseError)

from typing import Optional, List, Iterator, Iterable, Any, Match

# exception for when things go bad, but not necessarily because of the linearizer
class LinearizerCouldNotLinearize(Exception):
    pass

# exception for when the linearizer trips onto itself
class LinearizerThisShouldNotHappen(Exception):
    pass

def linearize_commands(commands_sequence, coq, filename, relative_filename, skip_nochange_tac, debug):
    command = next(commands_sequence, None)
    while command:
        # Run up to the next proof
        while coq.count_fg_goals() == 0:
            coq.run_stmt(command)
            if coq.count_fg_goals() == 0:
                yield command
                command = next(commands_sequence, None)
                if not command:
                    return

        # Cancel the proof starting command so that we're right before the proof
        coq.cancel_last()

        # Pull the entire proof from the lifter into command_batch
        command_batch = []
        while command and not serapi_instance.ending_proof(command):
            command_batch.append(command)
            command = next(commands_sequence, None)
        # Get the QED on there too.
        command_batch.append(command)

        # Now command_batch contains everything through the next
        # Qed/Defined.
        theorem_statement = serapi_instance.kill_comments(command_batch.pop(0))
        theorem_name = theorem_statement.split(":")[0].strip()
        coq.run_stmt(theorem_statement)
        yield theorem_statement
        if [relative_filename, theorem_name] in compcert_failures:
            print("Skipping {}".format(theorem_name))
            for command in command_batch:
                coq.run_stmt(command)
                yield command
            command = next(commands_sequence, None)
            continue

        # This might not be super robust?
        match = re.fullmatch("\s*Proof with (.*)\.", command_batch[0])
        if match and match.group(1):
            with_tactic = match.group(1)
        else:
            with_tactic = ""

        orig = command_batch[:]
        command_batch = list(command_batch)
        try:
            batch_handled = list(handle_with(command_batch, with_tactic))
            linearized_commands = list(linearize_proof(coq, theorem_name, batch_handled,
                                                       skip_nochange_tac))
            leftover_commands = []

            for command in command_batch:
                # goals = coq.query_goals()
                if command and (coq.count_fg_goals() != 0 or
                                serapi_instance.ending_proof(command) or
                                "Transparent" in command):
                    coq.run_stmt(command)
                    leftover_commands.append(command)

            yield from linearized_commands
            for command in leftover_commands:
                yield command
        except (BadResponse, CoqExn, LinearizerCouldNotLinearize, ParseError, TimeoutError) as e:
            print("Aborting current proof linearization!")
            print("Proof of:\n{}\nin file {}".format(theorem_name, filename))
            print()
            if debug:
                raise e
            coq.run_stmt("Abort.")
            coq.run_stmt(theorem_statement)
            for command in orig:
                if command:
                    coq.run_stmt(command)
                    yield command

        command = next(commands_sequence, None)

def linearize_proof(coq : serapi_instance.SerapiInstance,
                    theorem_name : str,
                    command_batch : List[str],
                    skip_nochange_tac:bool=False) -> Iterable[str]:
    assert coq.count_fg_goals() == 1
    pending_commands_stack = []
    num_goals_stack = []
    while command_batch:
        while coq.count_fg_goals() == 0:
            print("hit the end of this subgoal")
            indentation = "  " * (len(num_goals_stack) - 1)
            if len(num_goals_stack) == 0:
                return
            print("closing")
            coq.run_stmt("}")
            yield indentation + "}"
            if num_goals_stack[-1] > 1:
                num_goals_stack[-1] -= 1
                coq.run_stmt("{")
                yield indentation + "{"
                pending_commands = pending_commands_stack[-1]
                print(f"pending commands are {pending_commands}")
                if pending_commands:
                    command_batch.insert(0, pending_commands)
            else:
                pending_commands_stack.pop()
                num_goals_stack.pop()
        command = command_batch.pop(0)
        assert isinstance(command, str), command
        if re.match("\s*[*+-]+\s*|\s*[{}]\s*", command):
            continue

        semi_match = re.match("\s*(.*?)\s*;\s*(.*\.)", command)
        if not semi_match:
            print("Running 1")
            coq.run_stmt(command)
            indentation = "  " * len(num_goals_stack)
            yield indentation + command.strip()
            if coq.count_fg_goals() > 1:
                num_goals_stack.append(coq.count_fg_goals())
                pending_commands_stack.append(None)
                coq.run_stmt("{")
                yield indentation + "{"
        else:
            print(f"Matching {command}")
            base_command, rest = semi_match.group(1, 2)
            print("Running 2")
            coq.run_stmt(base_command + ".")
            indentation = "  " * len(num_goals_stack)
            yield indentation + base_command.strip() + "."
            command_batch.insert(0, rest)
            if coq.count_fg_goals() > 1:
                num_goals_stack.append(coq.count_fg_goals())
                pending_commands_stack.append(rest)
                coq.run_stmt("{")
                yield indentation + "{"
    pass

def handle_with(command_batch : Iterable[str],
                with_tactic : str) -> Iterable[str]:
    if not with_tactic:
        for command in command_batch:
            assert "..." not in command
            yield command
    else:
        for command in command_batch:
            newcommand = re.sub("(\S+)\s*\.\.\.", f"\1 ; {with_tactic}", command)
            yield newcommand

def split_commas(commands : Iterator[str]) -> Iterator[str]:
    def split_commas_command(command : str) -> Iterator[str]:
        if not "," in command:
            yield command
        else:
            stem, args_str = serapi_instance.split_tactic(command)
            if stem == "rewrite" or stem == "rewrite <-" or stem == "unfold":
                multi_in_match = re.match(r"\s*({}\s+.*?\s+in\b\s+)(\S+)\s*,\s*(.*\.)"
                                          .format(stem.split()[0]),
                                          command)
                if multi_in_match:
                    command, first_context, rest_context = multi_in_match.group(1, 2, 3)
                    yield from split_commas_command(command + first_context + ".")
                    yield from split_commas_command(command + rest_context)
                    return
                pattern = r"\s*({}\s+\S*\s*),\s*(.*)\s+in\b(.*\.)"\
                    .format(stem.split()[0])
                in_match = re.match(pattern, command)
                if in_match:
                    first_command, rest, context = in_match.group(1, 2, 3)
                    yield first_command + " in " + context
                    yield from split_commas_command("{} {} in {}"
                                                    .format(stem, rest, context))
                    return
                parts_match = re.match(r"\s*({}\s+(!?\s*\S+|\(.*?\))\s*),\s*(.*)"
                                       .format(stem.split()[0]),
                                       command)
                if parts_match:
                    first_command, rest = parts_match.group(1, 3)
                    yield first_command + ". "
                    yield from split_commas_command("{} ".format(stem.split()[0]) + rest)
                    return

                yield command
            else:
                yield command
    for command in commands:
        split_commands = split_commas_command(command)
        # print("Split {} into {}".format(command, list(split_commands)))
        yield from split_commands
def postlinear_desugar_tacs(commands :  Iterable[str]) -> Iterable[str]:
    # yield from commands
    for split_command in split_commas(commands):
        yield split_command
def prelinear_desugar_tacs(commands : Iterable[str]) -> Iterable[str]:
    for command in commands:
        now_match = re.search(r"\bnow\s+", command)
        if now_match:
            paren_depth = 0
            arg_end = -1
            for idx, (c, next_c) in enumerate(
                    zip(command[now_match.end():],
                        command[now_match.end()+1:])):
                if c == "(" or c == "[":
                    paren_depth += 1
                elif (c == ")" or c == "]" or
                      (c == "." and re.match("\W", next_c))
                      or c == "|") \
                      and paren_depth == 0:
                    arg_end = idx + now_match.end()
                    break
                elif c == ")" or c == "]":
                    paren_depth -= 1
            result = command[:now_match.start()] + \
                "(" + command[now_match.end():arg_end] + "; easy)" \
                + command[arg_end:]
            yield result
        else:
            yield command

def lifted_vernac(command : str) -> Optional[Match[Any]]:
    return re.match("Ltac\s", serapi_instance.kill_comments(command).strip())

def generate_lifted(commands : List[str], coq : serapi_instance.SerapiInstance) \
    -> Iterator[str]:
    lemma_stack = [] # type: List[List[str]]
    for command in commands:
        if serapi_instance.possibly_starting_proof(command):
            coq.run_stmt(command)
            if coq.proof_context != None:
                lemma_stack.append([])
            coq.cancel_last()
        if len(lemma_stack) > 0 and not lifted_vernac(command):
            lemma_stack[-1].append(command)
        else:
            yield command
        if serapi_instance.ending_proof(command):
            yield from lemma_stack.pop()
    assert(len(lemma_stack) == 0)

def preprocess_file_commands(commands : List[str], coqargs : List[str], includes : str,
                             prelude : str, filename : str, relative_filename : str,
                             skip_nochange_tac : bool,
                             debug=False) -> List[str]:
    try:
        with serapi_instance.SerapiContext(coqargs, includes, prelude) as coq:
            coq.debug = debug
            result = list(
                postlinear_desugar_tacs(linearize_commands(
                    prelinear_desugar_tacs(generate_lifted(commands, coq)),
                    coq, filename, relative_filename, skip_nochange_tac, debug)))
        return result
    except (CoqExn, BadResponse, AckError, CompletedError):
        print("In file {}".format(filename))
        raise
    except serapi_instance.TimeoutError:
        print("Timed out while lifting commands! Skipping linearization...")
        return commands

import helper
def main():
    parser = argparse.ArgumentParser(description=
                                     "linearize a set of files")
    parser.add_argument('--prelude', default=".")
    parser.add_argument('--debug', default=False, const=True, action='store_const')
    parser.add_argument('--skip-nochange-tac', default=False, const=True, action='store_const',
                        dest='skip_nochange_tac')
    parser.add_argument('filenames', nargs="+", help="proof file name (*.v)")
    arg_values = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__)) + "/.."
    includes = subprocess.Popen(['make', '-C', arg_values.prelude, 'print-includes'],
                                stdout=subprocess.PIPE).communicate()[0].decode('utf-8')
    coqargs = ["{}/coq-serapi/sertop.native".format(base),
               "--prelude={}/coq".format(base)]

    for filename in arg_values.filenames:
        local_filename = arg_values.prelude + "/" + filename
        fresh_commands = preprocess_file_commands(
            helper.load_commands_preserve(arg_values.prelude + "/" + filename),
            coqargs, includes, arg_values.prelude,
            local_filename, filename,
            arg_values.skip_nochange_tac, debug=arg_values.debug)
        helper.save_lin(fresh_commands, local_filename)

if __name__ == "__main__":
    main()
