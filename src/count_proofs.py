import argparse
import sys
from tqdm import tqdm

import serapi_instance
from context_filter import get_context_filter
from format import *
from util import *
from data import read_all_text_data

from typing import Dict, Tuple, Any, cast

def main() -> None:
    args, parser = parse_arguments()

    # Set up --all and --some flags
    assert not (args.all and args.some)
    if (not args.all) and (not args.some):
        args.all = True

    # Load the includes from a _CoqProject file in prelude
    try:
        with open(args.prelude + "/_CoqProject", 'r') as includesfile:
            includes = includesfile.read()
    except FileNotFoundError:
        eprint("Didn't find a _CoqProject file in prelude dir")
        includes = ""

    # Do the counting
    sub_counts, sub_totals = zip(*[count_proofs(args, includes, filename) for filename in args.filenames])

    sub_count = sum(sub_counts)
    sub_total = sum(sub_totals)
    print(f"Total: {sub_count}/{sub_total} "
          f"({stringified_percent(sub_count, sub_total)}%)")

def parse_arguments() -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description=
        "Count the number of proofs matching criteria")
    parser.add_argument('--prelude', default=".")
    parser.add_argument('--debug', default=False, const=True, action='store_const')
    parser.add_argument("--verbose", "-v", help="verbose output", action='store_true')
    parser.add_argument('--context-filter', dest="context_filter", type=str,
                        default=None)
    parser.add_argument("--max-length", dest="max_length", type=int,
                        default=120)

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--all", "-a", action='store_true')
    g.add_argument("--some", "-s", action='store_true')

    parser.add_argument('filenames', nargs="+", help="proof file name (*.v)")
    return parser.parse_args(), parser

def count_proofs(args : argparse.Namespace, includes : str, filename : str) \
    -> Tuple[int, int]:
    eprint(f"Counting {filename}", guard=args.debug)
    scrapefile= args.prelude + "/" + filename + ".scrape"
    interactions = list(read_all_text_data(args.prelude + "/" + filename + ".scrape"))
    filter_func = get_context_filter(args.context_filter)

    count = 0
    total_count = 0
    cur_proof_counts = False
    cur_lemma_name = ""
    extended_interactions : List[Optional[ScrapedCommand]] = \
        cast(List[Optional[ScrapedCommand]], interactions[1:])  + [None]
    for inter, next_inter in zip(interactions, extended_interactions):
        if isinstance(inter, ScrapedTactic):
            goal_before = inter.goal
            hyps_before = inter.hypotheses
            command = inter.tactic
        else:
            goal_before = ""
            hyps_before = []
            command = inter

        if next_inter and isinstance(next_inter, ScrapedTactic):
            goal_after = next_inter.goal
            hyps_after = next_inter.hypotheses
        else:
            goal_after = ""
            hyps_after = []

        entering_proof = bool((not goal_before) and goal_after)
        exiting_proof = bool(goal_before and not goal_after)

        if entering_proof:
            cur_lemma_name = serapi_instance.lemma_name_from_statement(next_inter.prev_tactics[0])
            cur_proof_counts = False if args.some else True
            continue

        if cur_lemma_name:
            if filter_func({"goal":format_goal(goal_before),
                            "hyps":hyps_before},
                           command,
                           {"goal":format_goal(goal_after),
                            "hyps":goal_after},
                           args):
                if args.some and not cur_proof_counts:
                    cur_proof_counts = True
            else:
                if args.all and cur_proof_counts:
                    eprint(f"Eliminating proof {cur_lemma_name} "
                           f"because tactic {command.strip()} doesn't match",
                           guard=args.debug)
                    cur_proof_counts = False

        if exiting_proof:
            if cur_proof_counts:
                eprint(f"Proof of {cur_lemma_name} counts",
                       guard=args.debug)
                count += 1
            total_count += 1
            cur_lemma_name = ""
    print(f"{filename}: {count}/{total_count} "
          f"({stringified_percent(count, total_count)}%)")
    return count, total_count

def stringified_percent(total : float, outof : float) -> str:
    if outof == 0:
        return "NaN"
    else:
        return "{:5.2f}".format(total * 100 / outof)

if __name__ == "__main__":
    main()
