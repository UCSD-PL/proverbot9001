import csv
import argparse
import re

from context_filter import ContextFilter, get_context_filter
from format import format_goal, format_hypothesis
from serapi_instance import possibly_starting_proof, ending_proof

from report_csv import read_csvfile, pairwise, check_cfilter_row, TacticRow, CommandRow


def main() -> None:
    parser = argparse.ArgumentParser(description="Tells you what percentage of tactics, "
                                     "and entire proofs, pass a predicate")

    parser.add_argument("filenames", nargs="+", help="csv file names")
    parser.add_argument("--context-filter", dest="context_filter",
                        type=str, default="default")
    args = parser.parse_args()

    cfilter = get_context_filter(args.context_filter)

    num_tactics_pass = 0
    num_proofs_pass = 0
    num_tactics_total = 0
    num_proofs_total = 0

    for filename in args.filenames:
        options, rows = read_csvfile(filename)
        in_proof = False
        current_proof_perfect = False
        cur_lemma_name = ""
        for row, nextrow in pairwise(rows):
            if isinstance(row, TacticRow):
                if not in_proof:
                    current_proof_perfect = True
                in_proof = True
                passes = check_cfilter_row(cfilter, row, nextrow)
                num_tactics_total += 1
                if not passes:
                    current_proof_perfect = False
                    # print("{} doesn't pass.".format(row))
                else:
                    # print("{} passes!".format(row))
                    num_tactics_pass += 1
            elif ending_proof(row.command) and in_proof:
                in_proof = False
                num_proofs_total += 1
                if current_proof_perfect:
                    num_proofs_pass += 1
                    # print("Proof : {},\n in {}, passed!".format(cur_lemma_name, filename))
            else:
                if possibly_starting_proof(row.command):
                    cur_lemma_name = row.command

    print("Filter {}: {}/{} tactics pass, {}/{} complete proofs pass"
          .format(args.context_filter,
                  num_tactics_pass, num_tactics_total,
                  num_proofs_pass, num_proofs_total))

if __name__ == "__main__":
    main()
