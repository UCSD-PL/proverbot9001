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
import csv
import itertools

import coq_serapy as serapi_instance
from coq_serapy.contexts import *
from util import *
from data import read_all_text_data
from pathlib_revised import Path2

from typing import Dict, Tuple, Any, cast, Pattern, Match

def main() -> None:
    args, parser = parse_arguments()

    csv_filenames = [count_lengths(args, filename)
                     for filename in args.filenames]
    with open(args.outfile, 'w') as outfile:
        for csv_filename in csv_filenames:
            with open(csv_filename, 'r') as infile:
                for line in infile:
                    outfile.write(line)

def parse_arguments() -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description=
        "Count the length of proofs in a file criteria")
    parser.add_argument('--prelude', default=".")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--outfile', '-o', default="proofs.csv", type=str)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--post-linearized", dest="post_linear", action='store_true')
    g.add_argument("--add-semis", dest="add_semis", action='store_true')

    parser.add_argument('filenames', nargs="+", help="proof file name (*.v)")
    return parser.parse_args(), parser

def toStr(obj) -> str:
    if isinstance(obj, ScrapedTactic):
        return obj.tactic
    else:
        return obj
def norm(statement : str):
    return serapi_instance.kill_comments(toStr(statement)).strip()

def count_lengths(args : argparse.Namespace, filename : str):
    print(f"Counting {filename}")
    full_filename = args.prelude + "/" + filename
    scraped_commands = list(read_all_text_data(Path2(full_filename + ".scrape")))
    scraped_iter = iter(scraped_commands)
    if args.post_linear:
        original_commands = serapi_instance.load_commands_preserve(args, 0,
                                                                   full_filename + ".lin")
    else:
        original_commands = serapi_instance.load_commands_preserve(args, 0, full_filename)

    with open(full_filename + ".csv", 'w') as fout:
        rowwriter = csv.writer(fout)
        lemma_statement = ""
        in_proof = False
        cur_len = 0
        for cmd in original_commands:
            if not serapi_instance.possibly_starting_proof(cmd) and not in_proof:
                continue
            if serapi_instance.possibly_starting_proof(cmd) and not in_proof:
                normalized_command = norm(cmd)
                cur_scraped = norm(next(scraped_iter))
                while cur_scraped != normalized_command:
                    cur_scraped = norm(next(scraped_iter))
                try:
                    next_after_start = next(scraped_iter)
                except StopIteration:
                    next_after_start = ""
                if isinstance(next_after_start, ScrapedTactic):
                    lemma_statement = norm(cmd)
                    in_proof = True
                    cur_len = 0
                else:
                    scraped_iter = itertools.chain([next_after_start], scraped_iter)
            elif serapi_instance.ending_proof(cmd):
                assert in_proof
                rowwriter.writerow([lemma_statement.strip(),
                                    cur_len])
                cur_len = -1
                in_proof=False
            elif in_proof:
                assert cur_len >= 0
                if re.match("[{}]|[*-+]*$", norm(cmd)):
                    continue
                if re.match("Proof\.", norm(cmd)):
                    continue
                cur_len += 1
                if args.add_semis or args.post_linear:
                    cur_len += count_outside_matching("\{\|", "\|\}", ";", norm(cmd))
    return full_filename + ".csv"

def count_outside_matching(openpat : str, closepat : str, substrpat : str,
                           target : str) -> int:
    count = 0
    depth = 0
    curpos = 0
    openp = re.compile(openpat)
    closep = re.compile(closepat)
    substrp = re.compile(substrpat)

    def search_pat(pat : Pattern) -> Tuple[Optional[Match], int]:
        match = pat.search(target, curpos)
        return match, match.end() if match else len(target) + 1
    while curpos <= len(target):
        _, nextopenpos = search_pat(openp)
        _, nextclosepos = search_pat(closep)
        _, nextsubpos = search_pat(substrp)

        nextpos = min(nextopenpos, nextclosepos, nextsubpos)
        if nextpos == nextopenpos:
            depth += 1
            curpos = nextpos
        elif nextpos == nextclosepos:
            depth -= 1
            assert depth >= 0
            curpos = nextpos
        else:
            assert nextpos == nextsubpos
            if depth == 0:
                count += 1
            curpos = nextpos
    return count
if __name__ == "__main__":
    main()
