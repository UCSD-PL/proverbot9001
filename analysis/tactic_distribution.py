
import csv
import argparse
import re
import format
import data
import collections

from data import get_text_data
from serapi_instance import get_stem

from typing import Counter

def main() -> None:
    parser = argparse.ArgumentParser(description
                                     ="Get the counts of each tactic from a scrapefile")
    parser.add_argument("--max-tuples", dest="max_tuples", default=None, type=str)
    parser.add_argument("--context-filter", dest="context_filter", default="default",
                        type=str)
    parser.add_argument("--stems-only", dest="stems_only", default=False, const=True,
                        action='store_const')
    parser.add_argument("filename", type=str)
    args = parser.parse_args()

    data = get_text_data(args.filename, args.context_filter, args.max_tuples, verbose=True)

    tactic_counts : Counter[str] = collections.Counter()
    for hyps, goal, tactic in data:
        if args.stems_only:
            tactic_counts.update([get_stem(tactic)])
        else:
            tactic_counts.update([tactic])

    print("tactic_counts: {}".format(tactic_counts))

main()
