#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
from typing import List

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

    in_commands = coq_serapy.load_commands(args.prelude / args.input_file)
    out_commands = add_proof_using(in_commands, args.prelude, verbosity=args.verbosity)

    with args.output_file.open('w') as f:
        for command in out_commands:
            print(command, file=f, end='')

def add_proof_using(commands: List[str], prelude: Path, verbosity: int = 0) -> List[str]:
    resulting_commands: List[str] = []
    cur_proof_commands: List[str] = []
    with coq_serapy.CoqContext(str(prelude), verbosity) as coq:
        # Start by running the command to allow us to see proof using suggestions
        coq.run_stmt("Set Suggest Proof Using.")
        for cmd in tqdm(commands):
            if coq.proof_context:
                cur_proof_commands.append(cmd)
            else:
                resulting_commands.append(cmd)
            coq.run_stmt(cmd)
            if coq_serapy.ending_proof(cmd):
                assert isinstance(coq.backend, coq_serapy.CoqSeraPyInstance)
                suggestion_match = re.match(
                    r"\n?The proof of ([^ \n]+)(?: |\n)"
                    r"should start with one of the following commands:"
                    r"(?: |\n)(Proof using[^.]+\.)",
                    coq.backend.feedback_string)
                suggested_command = unwrap(suggestion_match).group(2) + "\n"
                if suggested_command != "Proof using Type.\n":
                    if cur_proof_commands[0].strip() == "Proof.":
                        cur_proof_commands[0] = suggested_command
                    else:
                        if not re.match(r"Proof\s+[^.]+\.", cur_proof_commands[0].strip()):
                            cur_proof_commands.insert(0, suggested_command)

                resulting_commands += cur_proof_commands
                cur_proof_commands = []
    return resulting_commands

if __name__ == "__main__":
    main()
