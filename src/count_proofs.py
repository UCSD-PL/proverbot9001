#!/usr/bin/env python3.7
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
import sys

import coq_serapy as serapi_instance
from coq_serapy.contexts import (ScrapedCommand, ScrapedTactic,
                                 strip_scraped_output, TacticContext)
from context_filter import get_context_filter
from util import eprint, stringified_percent
from data import read_all_text_data
from pathlib_revised import Path2

from typing import Dict, Tuple, Any, cast

def main() -> None:
    args, parser = parse_arguments()

    # Set up --all and --some flags
    if (not args.all) and (not args.some):
        args.all = True

    # Do the counting
    sub_counts, sub_totals = zip(*[count_proofs(args, filename)
                                   for filename in args.filenames])

    sub_count = sum(sub_counts)
    sub_total = sum(sub_totals)
    if not args.print_name and not args.print_stmt:
        print(f"Total: {sub_count}/{sub_total} "
              f"({stringified_percent(sub_count, sub_total)}%)")


def parse_arguments() -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Count the number of proofs matching criteria")
    parser.add_argument('--prelude', default=".")
    parser.add_argument('--debug', default=False, const=True, action='store_const')
    parser.add_argument("--verbose", "-v", help="verbose output", action='store_true')
    parser.add_argument('--context-filter', dest="context_filter", type=str,
                        default="count-default")
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        '--print-name', dest="print_name",
        help="Don't print counts just print the names of matching lemmas",
        action='store_true')
    g.add_argument(
        '--print-stmt', dest="print_stmt",
        help="Don't print counts just print the matching lemma statements",
        action='store_true')
    parser.add_argument("--max-length", dest="max_length", type=int,
                        default=120)

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--all", "-a", action='store_true')
    g.add_argument("--some", "-s", action='store_true')

    parser.add_argument('filenames', nargs="+", help="proof file name (*.v)")
    return parser.parse_args(), parser


def count_proofs(args: argparse.Namespace, filename: str) \
  -> Tuple[int, int]:
    eprint(f"Counting {filename}", guard=args.debug)
    scrapefile = args.prelude + "/" + filename + ".scrape"
    interactions = list(read_all_text_data(Path2(scrapefile)))
    filter_func = get_context_filter(args.context_filter)

    count = 0
    total_count = 0
    cur_proof_counts = False
    cur_lemma_stmt = ""
    extended_interactions: List[Optional[ScrapedCommand]] = \
        cast(List[Optional[ScrapedCommand]], interactions[1:]) + [None]
    for inter, next_inter in zip(interactions, extended_interactions):
        if isinstance(inter, ScrapedTactic):
            context_before = strip_scraped_output(inter)
            command = inter.tactic
        else:
            context_before = TacticContext([], [], [], "")
            command = inter

        if next_inter and isinstance(next_inter, ScrapedTactic):
            context_after = strip_scraped_output(next_inter)
        else:
            context_after = TacticContext([], [], [], "")

        entering_proof = bool((not context_before.goal) and context_after.goal)
        exiting_proof = bool(context_before.goal and not context_after.goal)

        if entering_proof:
            cur_lemma_stmt = next_inter.prev_tactics[0]
            cur_proof_counts = False if args.some else True
            continue

        if cur_lemma_stmt:
            if filter_func(context_before, command,
                           context_after, args):
                if args.some and not cur_proof_counts:
                    cur_proof_counts = True
            else:
                if args.all and cur_proof_counts:
                    cur_lemma_name = serapi_instance.lemma_name_from_statement(cur_lemma_stmt)
                    eprint(f"Eliminating proof {cur_lemma_name} "
                           f"because tactic {command.strip()} doesn't match",
                           guard=args.debug)
                    cur_proof_counts = False

        if exiting_proof:
            if cur_proof_counts:
                cur_lemma_name = serapi_instance.lemma_name_from_statement(
                    cur_lemma_stmt)
                if args.print_name:
                    print(cur_lemma_name)
                if args.print_stmt:
                    print(re.sub("\n", "\\n", cur_lemma_stmt))
                eprint(f"Proof of {cur_lemma_name} counts",
                       guard=args.debug)
                count += 1
            total_count += 1
            cur_lemma_stmt = ""
    if not args.print_name and not args.print_stmt:
        print(f"{filename}: {count}/{total_count} "
              f"({stringified_percent(count, total_count)}%)")
    return count, total_count


if __name__ == "__main__":
    main()
