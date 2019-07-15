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
import re
import serapi_instance

from typing import Tuple


def main() -> None:
    args, parser = parse_arguments()

    length_map = {}
    with open(args.lengthinput, 'r') as lengthfile:
        reader = csv.reader(lengthfile)
        for lemma_statement, length in reader:
            length_map[lemma_statement] = length

    with open(args.outfile, 'w') as outfile:
        outfile.write("lemma,status,prooflength\n")
        writer = csv.writer(outfile)
        with open(args.statusinput, 'r') as statusfile:
            reader = csv.reader(statusfile)
            for line in reader:
                if len(line) == 1:
                    assert re.match("#.*", line[0])
                    writer.writerow(line)
                    continue
                lemma_statement, status, old_length = line
                if lemma_statement == "lemma":
                    continue
                assert lemma_statement in length_map, (lemma_statement, length_map)
                writer.writerow([lemma_statement, status, length_map[lemma_statement]])
def norm(statement : str):
    return serapi_instance.kill_comments(stmt).strip()

def parse_arguments() -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description=
        "Merge two CSV's containing proof lengths")
    parser.add_argument('statusinput')
    parser.add_argument('lengthinput')
    parser.add_argument('outfile')
    return parser.parse_args(), parser
if __name__ == "__main__":
    main()
