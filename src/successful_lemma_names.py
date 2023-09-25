#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib_revised import Path2

import coq_serapy as coq_serapy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=
        "Script which prints out the names of "
        "successful lemmas in a search report")
    parser.add_argument("proofs_files",
                        help="-proofs.txt files",
                        nargs="+",
                        type=Path2)
    args = parser.parse_args()

    for filename in args.proofs_files:
        with filename.open('r') as proof_file:
            for line in proof_file:
                (filename, module, lemma_stmt), sol = json.loads(line)
                if sol["status"] == "SUCCESS":
                    print(coq_serapy.lemma_name_from_statement(
                        lemma_stmt))

if __name__ == "__main__":
    main()
