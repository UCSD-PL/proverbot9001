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
import multiprocessing
import functools
import itertools

import coq_serapy as serapi_instance
from coq_serapy.contexts import (ScrapedCommand, ScrapedTactic,
                                 strip_scraped_output, TacticContext)
from context_filter import get_context_filter
from util import eprint, stringified_percent
from data import (read_all_text_data, read_all_text_data_worker__,
                  MixedDataset, file_chunks)
from pathlib_revised import Path2

from typing import List, Optional, Tuple, cast


def main() -> None:
    args, parser = parse_arguments()

    # Set up --all and --some flags
    if (not args.all) and (not args.some):
        args.all = True

    # Do the counting
    total_proofs = 0
    total_match = 0
    with multiprocessing.Pool(args.num_threads) as pool:
        results = pool.imap(functools.partial(count_proofs, args),
                            args.filenames)
        for (matches, num_proofs), filename in zip(results, args.filenames):
            if not args.print_name and not args.print_stmt:
                print(f"{filename}: "
                      f"{len(matches)}/{num_proofs} "
                      f"{stringified_percent(len(matches),num_proofs)}%")
            if args.print_stmt:
                for match in matches:
                    print(match)
            elif args.print_name:
                for match in matches:
                    print(serapi_instance.lemma_name_from_statement(match))
            else:
                print(f"{filename}: "
                      f"{len(matches)}/{num_proofs} "
                      f"{stringified_percent(len(matches),num_proofs)}%")
            total_proofs += num_proofs
            total_match += len(matches)

    if not args.print_name and not args.print_stmt:
        print(f"Total: {total_match}/{total_proofs} "
              f"({stringified_percent(total_match, total_proofs)}%)")


def parse_arguments() -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Count the number of proofs matching criteria")
    parser.add_argument('--prelude', default=".")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--verbose", "-v", help="verbose output",
                        action='count')
    parser.add_argument('--context-filter', dest="context_filter", type=str,
                        default="count-default")
    parser.add_argument('--num-threads', "-j", default=None, type=int)
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
  -> Tuple[List[str], int]:
    eprint(f"Counting {filename}", guard=args.debug)
    scrapefile = args.prelude + "/" + filename + ".scrape"
    interactions = list(read_all_text_data_singlethreaded(Path2(scrapefile)))
    filter_func = get_context_filter(args.context_filter)

    matches = []
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

        entering_proof = bool(isinstance(inter, str) and
                              isinstance(next_inter, ScrapedTactic))
        exiting_proof = bool(isinstance(inter, ScrapedTactic) and
                             isinstance(next_inter, str))

        if entering_proof:
            assert isinstance(next_inter, ScrapedTactic)
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
                    cur_lemma_name = serapi_instance.lemma_name_from_statement(
                        cur_lemma_stmt)
                    if cur_proof_counts:
                        eprint(f"Eliminating proof {cur_lemma_name} "
                               f"because tactic {command.strip()} "
                               "doesn't match",
                               guard=args.debug)
                        cur_proof_counts = False

        if exiting_proof:
            cur_lemma_name = serapi_instance.lemma_name_from_statement(
                cur_lemma_stmt)
            if not cur_lemma_name:
                cur_lemma_stmt = ""
                continue
            if cur_proof_counts:
                eprint(f"Proof of {cur_lemma_name} counts",
                       guard=args.debug)
                matches.append(cur_lemma_stmt)
            total_count += 1
            cur_lemma_stmt = ""

    return matches, total_count


def read_all_text_data_singlethreaded(data_path: Path2,
                                      num_threads: Optional[int] = None) \
                                    -> MixedDataset:
    line_chunks = file_chunks(data_path, 32768)
    try:
        yield from itertools.chain.from_iterable((
            read_all_text_data_worker__(chunk) for chunk in line_chunks))
    except AssertionError:
        print(f"Couldn't parse data in {str(data_path)}")
        raise


if __name__ == "__main__":
    main()
