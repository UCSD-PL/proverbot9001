#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
from typing import List, Iterable, Iterator

from tqdm import tqdm
import coq_serapy

from util import eprint, unwrap

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prelude", default=".", type=Path)
    parser.add_argument("-v", "--verbose", action='count', default=0, dest="verbosity")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    in_commands = coq_serapy.load_commands(args.input_file)
    out_commands = list(add_proof_using(tqdm(in_commands), args.prelude, verbosity=args.verbosity))

    with args.output_file.open('w') as f:
        for command in out_commands:
            print(command, file=f, end='')

def add_proof_using(commands: Iterable[str], prelude: Path, verbosity: int = 0) \
        -> Iterator[str]:
    with coq_serapy.CoqContext(str(prelude), verbosity) as coq:
        def run_commands(commands: Iterable[str]) -> Iterator[str]:
            for command in commands:
                coq.run_stmt(command)
                yield command
        yield from add_proof_using_with_running_instance(coq, run_commands(commands))

def add_proof_using_with_running_instance(coq: coq_serapy.CoqAgent, commands: Iterable[str]) \
        -> Iterator[str]:
    cur_proof_commands: List[str] = []
    in_proof = False
    # Start by running the command to allow us to see proof using
    # suggestions
    coq.run_stmt("Set Suggest Proof Using.")
    for cmd in commands:
        if in_proof:
            cur_proof_commands.append(cmd)
        else:
            yield cmd
        if coq_serapy.ending_proof(cmd):
            assert isinstance(coq.backend, coq_serapy.CoqSeraPyInstance)
            if cmd.strip() == "Qed." and coq.backend.feedback_string.strip() != "":
                suggestion_match = re.match(
                    r".*\n?The proof of ([^ \n]+)(?: |\n)"
                    r"should start with one of the following commands:"
                    r"(?: |\n)Proof using\s*([^.]+)\.",
                    coq.backend.feedback_string, re.DOTALL)
                suggested_deps = unwrap(suggestion_match).group(2)
                if suggested_deps.strip() == "":
                    suggested_deps = "Type"
                proof_cmd_match = re.match(r"Proof(\s*[^.]*)\.",
                                           cur_proof_commands[0].strip())
                if proof_cmd_match:
                    proof_cmd_suffix = proof_cmd_match.group(1)
                    cur_proof_commands[0] = "Proof using " + suggested_deps + proof_cmd_suffix + ".\n"
                else:
                    cur_proof_commands.insert(0, "Proof using " + suggested_deps + ".\n")

            yield from cur_proof_commands
            cur_proof_commands = []
            in_proof = False
        if coq.proof_context:
            in_proof = True


if __name__ == "__main__":
    main()
