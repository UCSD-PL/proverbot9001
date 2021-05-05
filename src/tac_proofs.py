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
import re

from typing import Tuple
from sys import stderr
from tqdm import tqdm

from util import stringified_percent

import coq_serapy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count the number of proofs with only certain tactics")
    parser.add_argument('--verbose', "-v", action='count', default=0)
    parser.add_argument('--progress', "-P", action='store_true')
    parser.add_argument('filenames', nargs="+", help="proof file name (*.v)")
    args = parser.parse_args()

    total_proofs = 0
    total_match = 0
    for filename in args.filenames:
        num_proofs, num_match = count_proofs(args, filename)
        print(f"{filename}: "
              f"{num_match}/{num_proofs} "
              f"{stringified_percent(num_match,num_proofs)}%")
        total_proofs += num_proofs
        total_match += num_match
    print(f"{total_match}/{total_proofs} "
          f"{stringified_percent(total_match,total_proofs)}%")


def count_proofs(args: argparse.Namespace, filename: str) -> Tuple[int, int]:
    commands = coq_serapy.load_commands(filename, progress_bar=args.progress)
    num_proofs = 0
    num_match = 0
    in_proof = False
    proof_matches = True
    for command in tqdm(reversed(commands), disable=not args.progress,
                        total=len(commands)):
        if coq_serapy.ending_proof(command):
            in_proof = True
            proof_matches = True
        elif in_proof and coq_serapy.possibly_starting_proof(command):
            lemma_name = coq_serapy.lemma_name_from_statement(command)
            in_proof = False
            if proof_matches:
                if args.verbose >= 2:
                    print(f"Proof {lemma_name} matches", file=stderr)
                num_match += 1
            elif args.verbose >= 2:
                print(f"Proof {lemma_name} doesnt match", file=stderr)
            num_proofs += 1
        elif in_proof and proof_matches:
            stem = coq_serapy.get_stem(command)
            if stem == "Proof":
                continue

            # General automation
            if re.match("e?auto", stem):
                continue

            # Goal touchers
            if re.match("intros?", stem):
                continue
            if re.match("induction", stem):
                continue
            if re.match("destruct", stem):
                continue
            if re.match("unfold", stem):
                continue

            # Hyp touchers
            if re.match("e?rewrite", stem):
                continue
            if re.match("e?apply", stem):
                continue

            # These are optional
            # if re.match("simpl", stem):
            #     continue
            # if re.match("congruence", stem):
            #     continue
            # if re.match("reflexivity", stem):
            #     continue
            # if re.match("inversion", stem):
            #     continue

            if args.verbose >= 2:
                print(
                    f"Command {command} doesn't match, eliminating proof",
                    file=stderr)
            proof_matches = False

        pass
    return num_proofs, num_match


if __name__ == "__main__":
    main()
