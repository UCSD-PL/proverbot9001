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

from tqdm import tqdm
import argparse
import re
import sys
from pathlib import Path

# This dependency is in pip, the python package manager
from sexpdata import *
from traceback import *
from util import *
from util import eprint, split_by_char_outside_matching, hash_file
from compcert_linearizer_failures import compcert_failures
from add_proof_using import add_proof_using_with_running_instance

import coq_serapy as serapi_instance
from coq_serapy import (AckError, CompletedError, CoqExn,
                        BadResponse, CoqTimeoutError, ParseError,
                        NoSuchGoalError, CoqAnomaly)

from typing import (Optional, List, Iterator, Iterable, Any, Match,
                    Union, cast)

from itertools import islice
import signal

# exception for when things go bad, but not necessarily because of the linearizer


class LinearizerCouldNotLinearize(Exception):
    pass

# exception for when the linearizer trips onto itself


class LinearizerThisShouldNotHappen(Exception):
    pass


def linearize_commands(args: argparse.Namespace,
                       commands_sequence: Iterable[str],
                       coq: serapi_instance.CoqAgent,
                       filename: str, relative_filename: str,
                       skip_nochange_tac: bool,
                       known_failures: List[List[str]]):
    commands_iter = iter(commands_sequence)
    command = next(commands_iter, None)
    if not command:
        return commands_sequence
    assert command, "Got an empty sequence!"
    while command:
        # Run up to the next proof
        while coq.count_fg_goals() == 0:
            coq.run_stmt(command)
            if coq.count_fg_goals() == 0:
                yield command
                command = next(commands_iter, None)
                if not command:
                    return

        # Cancel the proof starting command so that we're right before the proof
        coq.cancel_last()

        # Pull the entire proof from the lifter into command_batch
        command_batch = []
        while command and not serapi_instance.ending_proof(command):
            command_batch.append(command)
            command = next(commands_iter, None)
        # Get the QED on there too.
        if command:
            command_batch.append(command)

        # Now command_batch contains everything through the next
        # Qed/Defined.
        theorem_statement = serapi_instance.kill_comments(command_batch.pop(0))
        theorem_name = theorem_statement.split(":")[0].strip()
        coq.run_stmt(theorem_statement)
        yield theorem_statement
        if [relative_filename, theorem_name] in known_failures:
            eprint("Skipping {}".format(theorem_name), guard=args.verbose >= 1)
            for command in command_batch:
                coq.run_stmt(command)
                yield command
            command = next(commands_iter, None)
            continue

        # This might not be super robust?
        match = re.fullmatch(r"\s*Proof with (.*)\.\s*", command_batch[0])
        if match and match.group(1):
            with_tactic = match.group(1)
        else:
            with_tactic = ""

        orig = command_batch[:]
        command_batch = list(prelinear_desugar_tacs(command_batch))
        try:
            try:
                batch_handled = list(handle_with(command_batch, with_tactic))
                linearized_commands = list(linearize_proof(coq, theorem_name, batch_handled,
                                                           args.verbose, skip_nochange_tac))
                yield from linearized_commands
            except (BadResponse, CoqExn, LinearizerCouldNotLinearize, ParseError, CoqTimeoutError, NoSuchGoalError) as e:
                if args.verbose:
                    eprint("Aborting current proof linearization!")
                    eprint("Proof of:\n{}\nin file {}".format(
                        theorem_name, filename))
                    eprint()
                if vars(args).get("hardfail", False):
                    raise e
                coq.run_stmt("Abort.")
                for command in orig:
                    inner_ltac_match = re.match(r"\s*(?:(?:Local|Global)\s+)?Ltac\s+(\S+)\s+", command)
                    if inner_ltac_match:
                        coq.run_stmt("Reset {}.".format(inner_ltac_match.group(1)))
                coq.run_stmt(theorem_statement)
                for command in orig:
                    if command:
                        coq.run_stmt(command, timeout=360)
                        yield command
        except CoqAnomaly:
            eprint(
                f"Anomaly! Raising with {[relative_filename, theorem_name]}", guard=args.verbose >= 1)
            raise CoqAnomaly([relative_filename, theorem_name])

        command = next(commands_iter, None)


def linearize_proof(coq: serapi_instance.SerapiInstance,
                    theorem_name: str,
                    command_batch: List[str],
                    verbose: int = 0,
                    skip_nochange_tac: bool = False) -> Iterable[str]:
    pending_commands_stack: List[Union[str, List[str], None]] = []
    while command_batch:
        while coq.count_fg_goals() == 0:
            indentation = "  " * (len(pending_commands_stack))
            if len(pending_commands_stack) == 0:
                while command_batch:
                    command = command_batch.pop(0)
                    if "Transparent" in command or \
                       serapi_instance.ending_proof(command):
                        coq.run_stmt(command)
                        yield command
                return
            coq.run_stmt("}")
            yield indentation + "}"
            if coq.count_fg_goals() > 0:
                coq.run_stmt("{")
                yield indentation + "{"
                pending_commands = pending_commands_stack[-1]
                if isinstance(pending_commands, list):
                    next_cmd, *rest_cmd = pending_commands
                    dotdotmatch = re.match(
                        r"(.*)<\.\.>", next_cmd, flags=re.DOTALL)
                    for cmd in rest_cmd:
                        dotdotmatch = re.match(
                            r"(.*)<\.\.>", cmd, flags=re.DOTALL)
                        if dotdotmatch:
                            continue
                        assert serapi_instance.isValidCommand(cmd), \
                            f"\"{cmd}\" is not a valid command"
                    if (not rest_cmd) and dotdotmatch:
                        pending_commands_stack[-1] = [next_cmd]
                        assert serapi_instance.isValidCommand(dotdotmatch.group(1)), \
                            f"\"{dotdotmatch.group(1)}\" is not a valid command"
                        command_batch.insert(0, dotdotmatch.group(1))
                    else:
                        assert serapi_instance.isValidCommand(next_cmd), \
                            f"\"{next_cmd}\" is not a valid command"
                        command_batch.insert(0, next_cmd)
                    pending_commands_stack[-1] = rest_cmd if rest_cmd else None
                    pass
                elif pending_commands:
                    assert serapi_instance.isValidCommand(pending_commands), \
                        f"\"{command}\" is not a valid command"
                    command_batch.insert(0, pending_commands)
            else:
                popped = pending_commands_stack.pop()
                if isinstance(popped, list) and len(popped) > 0 and len(pending_commands_stack) > 1:
                    if pending_commands_stack[-1] is None:
                        pending_commands_stack[-1] = popped
                    elif isinstance(pending_commands_stack[-1], list):
                        if isinstance(popped, list) and "<..>" in popped[-1]:
                            raise LinearizerCouldNotLinearize
                        pending_commands_stack[-1] = popped + \
                            pending_commands_stack[-1]
        command = command_batch.pop(0)
        assert serapi_instance.isValidCommand(command), \
            f"command is \"{command}\", command_batch is {command_batch}"
        comment_before_command = ""
        command_proper = command
        while re.fullmatch(r"\s*\(\*.*", command_proper, flags=re.DOTALL):
            next_comment, command_proper = \
                split_to_next_matching(r"\(\*", r"\*\)", command_proper)
            command_proper = command_proper[1:]
            comment_before_command += next_comment
        if comment_before_command:
            yield comment_before_command
        if re.match(r"\s*[*+-]+\s*|\s*[{}]\s*", command):
            continue

        command = serapi_instance.kill_comments(command_proper)
        if verbose >= 2:
            eprint(f"Linearizing command \"{command}\"")

        if "Ltac" in command:
            coq.run_stmt(command)
            yield command
            continue

        goal_selector_match = re.fullmatch(
            r"\s*(\d+)\s*:\s*(.*)\.\s*", command)
        if goal_selector_match:
            goal_num = int(goal_selector_match.group(1))
            rest = goal_selector_match.group(2)
            if goal_num < 2:
                raise LinearizerCouldNotLinearize
            if pending_commands_stack[-1] is None:
                completed_rest = rest + "."
                assert serapi_instance.isValidCommand(rest + "."),\
                    f"\"{completed_rest}\" is not a valid command in {command}"
                pending_commands_stack[-1] = ["idtac."] * \
                    (goal_num - 2) + [completed_rest]
            elif isinstance(pending_commands_stack[-1], str):
                pending_cmd = pending_commands_stack[-1]
                pending_commands_stack[-1] = [pending_cmd] * (goal_num - 2) + \
                    [rest + " ; " + pending_cmd] + [pending_cmd + "<..>"]
            else:
                assert isinstance(pending_commands_stack[-1], list)
                pending_cmd_lst = pending_commands_stack[-1]
                try:
                    old_selected_cmd = pending_cmd_lst[goal_num - 2]
                except IndexError:
                    raise LinearizerCouldNotLinearize
                match = re.match(r"(.*)\.$", old_selected_cmd, re.DOTALL)
                assert match, f"\"{old_selected_cmd}\" doesn't match!"
                cmd_before_period = unwrap(match).group(1)
                new_selected_cmd = f"{cmd_before_period} ; {rest}."
                pending_cmd_lst[goal_num - 2] = new_selected_cmd
            continue

        if split_by_char_outside_matching(r"\(", r"\)", r"\|\||&&", command):
            coq.run_stmt(command)
            yield command
            continue

        if re.match(r"\(", command.strip()):
            inside_parens, after_parens = split_to_next_matching(
                r'\(', r'\)', command)
            command = inside_parens.strip()[1:-1] + after_parens

        # Extend this to include "by \(" as an opener if you don't
        # desugar all the "by"s
        semi_match = split_by_char_outside_matching(r"try \(|\(|\{\|", r"\)|\|\}",
                                                    r"\s*;\s*", command)
        if semi_match:
            base_command, rest = semi_match
            rest = rest.lstrip()[1:]
            coq.run_stmt(base_command + ".")
            indentation = "  " * (len(pending_commands_stack) + 1)
            yield indentation + base_command.strip() + "."

            if re.match(r"\(", rest) and not \
               split_by_char_outside_matching(r"\(", r"\)", r"\|\|", rest):
                inside_parens, after_parens = split_to_next_matching(
                    r'\(', r'\)', rest)
                rest = inside_parens[1:-1] + after_parens
            bracket_match = re.match(r"\[", rest.strip())
            if bracket_match:
                bracket_command, rest_after_bracket = \
                    split_to_next_matching(r'\[', r'\]', rest)
                rest_after_bracket = rest_after_bracket.lstrip()[1:]
                clauses = multisplit_matching(r"\[", r"\]", r"(?<!\|)\|(?!\|)",
                                              bracket_command.strip()[1:-1])
                commands_list = [cmd.strip() if cmd.strip().strip(".") != ""
                                 else "idtac" + cmd.strip() for cmd in
                                 clauses]
                assert commands_list, command
                dotdotpat = re.compile(r"(.*)\.\.($|\W)")
                ending_dotdot_match = dotdotpat.match(commands_list[-1])
                if ending_dotdot_match:
                    commands_list = commands_list[:-1] + \
                        ([ending_dotdot_match.group(1)] *
                         (coq.count_fg_goals() -
                          len(commands_list) + 1))
                else:
                    starting_dotdot_match = dotdotpat.match(commands_list[0])
                    if starting_dotdot_match:
                        starting_tac = starting_dotdot_match.group(1)
                        commands_list = [starting_tac] * (coq.count_fg_goals() -
                                                          len(commands_list) + 1)\
                            + commands_list[1:]
                    else:
                        for idx, command_case in enumerate(commands_list[1:-1]):
                            middle_dotdot_match = dotdotpat.match(command_case)
                            if middle_dotdot_match:
                                commands_list = \
                                    commands_list[:idx] + \
                                    [command_case] * (coq.count_fg_goals() -
                                                      len(commands_list) + 1) + \
                                    commands_list[idx+1:]
                                break
                if rest_after_bracket.strip():
                    command_remainders = [cmd + ";" + rest_after_bracket
                                          for cmd in commands_list]
                else:
                    command_remainders = [cmd + "." for cmd in commands_list]
                assert serapi_instance.isValidCommand(command_remainders[0]), \
                    f"\"{command_remainders[0]}\" is not a valid command"
                command_batch.insert(0, command_remainders[0])
                if coq.count_fg_goals() > 1:
                    for command in command_remainders[1:]:
                        assert serapi_instance.isValidCommand(command), \
                            f"\"{command}\" is not a valid command"
                    pending_commands_stack.append(command_remainders[1:])
                    coq.run_stmt("{")
                    yield indentation + "{"
            else:
                if coq.count_fg_goals() > 0:
                    assert serapi_instance.isValidCommand(rest), \
                        f"\"{rest}\" is not a valid command, from {command}"
                    command_batch.insert(0, rest)
                if coq.count_fg_goals() > 1:
                    assert serapi_instance.isValidCommand(rest), \
                        f"\"{rest}\" is not a valid command, from {command}"
                    pending_commands_stack.append(rest)
                    coq.run_stmt("{")
                    yield indentation + "{"
        elif coq.count_fg_goals() > 0:
            coq.run_stmt(command)
            indentation = "  " * \
                (len(pending_commands_stack) +
                 1) if command.strip() != "Proof." else ""
            yield indentation + command.strip()
            if coq.count_fg_goals() > 1:
                pending_commands_stack.append(None)
                coq.run_stmt("{")
                yield indentation + "{"
    pass


def handle_with(command_batch: Iterable[str],
                with_tactic: str) -> Iterable[str]:
    if not with_tactic:
        for command in command_batch:
            yield re.sub(r"(.*)\s*\.\.\.", r"\1.", command)
    else:
        yield "Proof."
        for command in islice(command_batch, 1, None):
            newcommand = re.sub(
                r"(.*)\s*\.\.\.", rf"\1 ; {with_tactic}.", command)
            yield newcommand


def split_commas(command: str) -> str:
    rewrite_match = re.match(r"(.*)(?:\s|^)(rewrite\s+)([^;,]*?,\s*.*)", command,
                             flags=re.DOTALL)
    unfold_match = re.match(r"(.*)(unfold\s+)([^;,]*?),\s*(.*)", command,
                            flags=re.DOTALL)
    if rewrite_match:

        prefix, rewrite_command, id_and_rest = rewrite_match.group(1, 2, 3)
        split = split_by_char_outside_matching(r"\(", r"\)", ",", id_and_rest)
        if not split:
            return command
        first_id, rest = split
        rest = rest[1:]
        split = split_by_char_outside_matching(r"\(", r"\)", r";|\.", rest)
        assert split
        rewrite_rest, command_rest = split
        by_match = re.match(r"(.*)(\sby\s.*)", rewrite_rest)
        in_match = re.match(r"(.*)(\sin\s.*)", rewrite_rest)
        postfix = ""
        if by_match:
            if " by " not in first_id:
                postfix += by_match.group(2)
        if in_match:
            if " in " not in first_id:
                postfix += in_match.group(2)
        first_command = "(" + rewrite_command + first_id + " " + postfix + ")"
        result = prefix + first_command + ";" + \
            split_commas(rewrite_command + rest)
        return result
    elif unfold_match:
        prefix, unfold_command, first_id, rest = unfold_match.group(1, 2, 3, 4)
        if re.search(r"\sin\s", unfold_command + first_id):
            return command
        split = split_by_char_outside_matching(r"\(", r"\)", r";|\.", rest)
        assert split
        unfold_rest, command_rest = split
        in_match = re.match(r"(.*)(\sin\s.*)", unfold_rest)
        postfix = ""
        if in_match:
            if "in" not in first_id:
                postfix += in_match.group(2)
        first_command = unfold_command + first_id + " " + postfix
        return prefix + first_command + ";" + split_commas(unfold_command + rest)
    else:
        return command


def postlinear_desugar_tacs(commands:  Iterable[str]) -> Iterable[str]:
    yield from commands


def desugar_rewrite_by(cmd: str) -> str:
    rewrite_by_match = re.search(r"\b(rewrite\s*(?:!|<-)?.*?\s+)by\s+", cmd)
    if rewrite_by_match:
        prefix = cmd[:rewrite_by_match.start()]
        after_match = cmd[rewrite_by_match.end():]
        split = split_by_char_outside_matching(r"[\[(]", r"[\])]",
                                               r"\.\W|\.$|[|]|\)|;", after_match)
        assert split
        body, postfix = split
        postfix = desugar_rewrite_by(postfix)
        return f"{prefix}{rewrite_by_match.group(1)} ; [|{body} ..] {postfix}"
    else:
        return cmd


def desugar_assert_by(cmd: str) -> str:
    assert_by_match = re.search(r"\b(assert.*)\s+by\s+", cmd)
    if assert_by_match:
        prefix = cmd[:assert_by_match.start()]
        after_match = cmd[assert_by_match.end():]
        split = split_by_char_outside_matching(r"[\[(]", r"[\])]",
                                               r"\.\W|\.$|[|]",
                                               after_match)
        assert split
        body, postfix = split
        return f"{prefix}{assert_by_match.group(1)} ; [{body}..|] {postfix}"
    else:
        return cmd


def desugar_now(command: str) -> str:
    now_match = re.search(r"\bnow\s+", command)
    while(now_match):
        prefix = command[:now_match.start()]
        after_match = command[now_match.end():]
        split = split_by_char_outside_matching(r"[\[(]", r"[\])]",
                                               r"\.\W|\.$|]|\||\)",
                                               after_match)
        assert split
        body, postfix = split
        command = f"{prefix}({body} ; easy){postfix}"
        now_match = re.search(r"\bnow\s+", command)
    return command


desugar_passes = [split_commas, desugar_now,
                  desugar_rewrite_by, desugar_assert_by]


def prelinear_desugar_tacs(commands: Iterable[str]) -> Iterable[str]:
    for command in commands:
        comment_before_command = ""
        command_proper = command
        while re.fullmatch(r"\s*\(\*.*", command_proper, flags=re.DOTALL):
            next_comment, command_proper = \
                split_to_next_matching(r"\(\*", r"\*\)", command_proper)
            comment_before_command += next_comment
        for f in desugar_passes:
            newcommand = f(command_proper)
            assert serapi_instance.isValidCommand(
                newcommand), (command_proper, newcommand)
            command_proper = newcommand
        full_command = comment_before_command + command_proper
        yield full_command


def lifted_vernac(command: str) -> Optional[Match[Any]]:
    return re.match(r"Ltac\s", serapi_instance.kill_comments(command).strip())


def generate_lifted(commands: List[str], coq: serapi_instance.CoqAgent,
                    pbar: tqdm) \
        -> Iterator[str]:
    lemma_stack = []  # type: List[List[str]]
    for command in commands:
        if serapi_instance.possibly_starting_proof(command):
            coq.run_stmt(command)
            if coq.proof_context:
                lemma_stack.append([])
            coq.cancel_last()
        if len(lemma_stack) > 0 and not lifted_vernac(command):
            lemma_stack[-1].append(command)
        else:
            pbar.update(1)
            yield command
        if serapi_instance.ending_proof(command):
            pending_commands = lemma_stack.pop()
            pbar.update(len(pending_commands))
            yield from pending_commands
    assert len(lemma_stack) == 0, f"Stack still contains {lemma_stack}"


def preprocess_file_commands(args: argparse.Namespace, file_idx: int,
                             commands: List[str], coqargs: List[str],
                             prelude: str, filename: str,
                             relative_filename: str,
                             skip_nochange_tac: bool) -> List[str]:
    try:
        failed = True
        failures = list(compcert_failures)
        while failed:
            with serapi_instance.SerapiContext(
                    coqargs,
                    serapi_instance.get_module_from_filename(filename),
                    prelude) as coq:
                coq.verbose = args.verbose
                coq.quiet = True
                with tqdm(file=sys.stderr,
                          disable=not args.progress,
                          position=(file_idx * 2),
                          desc="Linearizing", leave=False,
                          total=len(commands),
                          dynamic_ncols=True,
                          bar_format= mybarfmt) as pbar:
                    try:
                        failed = False
                        result = list(
                            add_proof_using_with_running_instance(
                                coq,
                                postlinear_desugar_tacs(
                                    linearize_commands(
                                        args,
                                        generate_lifted(commands, coq, pbar),
                                        coq, filename, relative_filename,
                                        skip_nochange_tac, failures))))
                    except CoqAnomaly as e:
                        if isinstance(e.msg, str):
                            raise
                        failed = True
                        failures.append(cast(List[str], e.msg))
        return result
    except (CoqExn, BadResponse, AckError, CompletedError):
        eprint("In file {}".format(filename))
        raise
    except serapi_instance.CoqTimeoutError:
        eprint("Timed out while lifting commands! Skipping linearization...")
        return commands


class LinearizerTimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise LinearizerTimeoutException("Linearizer timed out")


def get_linearized(args: argparse.Namespace, coqargs: List[str],
                   bar_idx: int, filename: str) -> List[str]:
    local_filename = str(args.prelude) + "/" + filename
    loaded_commands = try_load_lin(
        args, bar_idx, local_filename)
    if loaded_commands is None:
        original_commands = \
            serapi_instance.load_commands_preserve(
                args, bar_idx, str(args.prelude) + "/" + filename)
        try:
            if "linearizer_timeout" in vars(args):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(args.linearizer_timeout)
            fresh_commands = preprocess_file_commands(
                args, bar_idx,
                original_commands,
                coqargs, str(args.prelude),
                local_filename, filename, False)
            signal.alarm(0)
        except LinearizerTimeoutException:
            fresh_commands = original_commands
        except (CoqAnomaly, CoqExn):
            fresh_commands = original_commands
        try:
            text_encoding = args.text_encoding
        except AttributeError:
            text_encoding = 'utf-8'
        save_lin(fresh_commands, local_filename, text_encoding)

        return fresh_commands
    else:
        return loaded_commands


def try_load_lin(args: argparse.Namespace, file_idx: int, filename: str) \
        -> Optional[List[str]]:
    lin_path = Path(filename + ".lin")
    if args.verbose > 0:
        eprint("Attempting to load cached linearized version from {}"
               .format(lin_path))
    if not lin_path.exists():
        return None
    try:
        ignore_lin_hash = args.ignore_lin_hash
    except AttributeError:
        ignore_lin_hash = False
    text_encoding = vars(args).get('text_encoding', 'utf-8')
    with lin_path.open(mode='r', encoding=text_encoding) as f:
        first_line = f.readline().strip()
        hash_str = first_line[3:-3]
        if ignore_lin_hash or hash_file(filename) == hash_str:
            contents = f.read()
            num_lines = len(contents.split("\n"))
            eprint(f"Parsing file with {num_lines} lines")
            commands = serapi_instance.read_commands(contents)
            return commands
        return None


def save_lin(commands: List[str], filename: str, text_encoding: str) -> None:
    output_file = filename + '.lin'
    with open(output_file, 'w', encoding=text_encoding) as f:
        print(f"(* {hash_file(filename)} *)", file=f)
        for command in commands:
            print(command, file=f)


def main():
    parser = argparse.ArgumentParser(description="linearize a set of files")
    parser.add_argument('--prelude', default=".")
    parser.add_argument('--hardfail', default=False,
                        const=True, action='store_const')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('--skip-nochange-tac', default=False, const=True,
                        action='store_const',
                        dest='skip_nochange_tac')
    parser.add_argument("--progress",
                        action='store_const', const=True, default=False)
    parser.add_argument("--linearizer-timeout",
                        type=int, default=(60 * 60 * 2))
    parser.add_argument("--text-encoding", default='utf-8', type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('filenames', nargs="+", help="proof file name (*.v)")
    arg_values = parser.parse_args()

    coqargs = ["sertop", "--implicit"]

    for filename in arg_values.filenames:
        if arg_values.verbose:
            eprint(f"Linearizing {filename}")
        local_filename = arg_values.prelude + "/" + filename
        if arg_values.resume:
            cmds = try_load_lin(arg_values, 0, local_filename)
            if cmds is not None:
                return
        original_commands = serapi_instance.load_commands_preserve(
            arg_values, 0, local_filename)
        try:
            fresh_commands = preprocess_file_commands(
                arg_values, 0, original_commands,
                coqargs, arg_values.prelude,
                local_filename, filename, False)
            save_lin(fresh_commands, local_filename, arg_values.text_encoding)
        except CoqAnomaly:
            save_lin(original_commands, local_filename, arg_values.text_encoding)


if __name__ == "__main__":
    main()
