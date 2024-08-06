#!/usr/bin/env python3

import argparse
import os.path
import json
from glob import glob
from dataclasses import dataclass

@dataclass
class ReportStats:
    succ_search_step_sum: int
    succ_proof_length_sum: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+")
    parser.add_argument("--full-csv", default=None)
    parser.add_argument("--names", nargs="*", type=str, default=None)
    args = parser.parse_args()

    if args.names is None:
        names = ["Report " + str(i) for i in range(len(args.reports))]
    else:
        assert len(args.names) == len(args.reports), \
            f"Number of names don't match number of reports! " \
            f"{len(args.names)} names but {len(args.reports)} reports"
        names = args.names


    proof_stats = [ReportStats(0,0) for _ in args.reports]
    both_first_two_but_not_third = 0
    third_but_not_both_first_two = 0
    either_first_two_but_not_third = 0
    third_but_not_either_first_two = 0

    for filename_0 in glob(os.path.join(args.reports[0], "*-proofs.txt")):
        filenames = [os.path.join(report, os.path.basename(filename_0))
                     for report in args.reports]
        proof_data_lists = []
        for filename in filenames:
            with open(filename, 'r') as f:
                proof_data_lists.append([json.loads(line) for line in f])

        lookup_dicts = [{(line[0][2], line[0][3]): line
                          for line in proof_data}
                        for proof_data in proof_data_lists]

        for idx, (job, _) in enumerate(proof_data_lists[0]):
            solutions = []
            for d in lookup_dicts:
                try:
                    other_job, solution = d[(job[2], job[3])]
                except KeyError:
                    print(f"Warning: couldn't find job {(job[2], job[3])} from file {filenames[0]} "
                          f"in filename {filenames[idx]}")
                    continue
                assert other_job == job, (job, other_job)
                solutions.append(solution)
            if solutions[0]["status"] == "SUCCESS" and solutions[1]["status"] == "SUCCESS" and not solutions[2]["status"] == "SUCCESS":
                both_first_two_but_not_third += 1
            if (solutions[0]["status"] == "SUCCESS" or solutions[1]["status"] == "SUCCESS") and not solutions[2]["status"] == "SUCCESS":
                either_first_two_but_not_third += 1
            if ((not solutions[0]["status"] == "SUCCESS") and (not solutions[1]["status"] == "SUCCESS")) and solutions[2]["status"] == "SUCCESS":
                third_but_not_either_first_two += 1
            if ((not solutions[0]["status"] == "SUCCESS") or (not solutions[1]["status"] == "SUCCESS")) and solutions[2]["status"] == "SUCCESS":
                third_but_not_both_first_two += 1

            if all(sol["status"] == "SUCCESS" for sol in solutions):
                for solution, proof_stat_obj in zip(solutions, proof_stats):
                    proof_stat_obj.succ_proof_length_sum += len(solution['commands']) - 2
                    proof_stat_obj.succ_search_step_sum += solution['steps_taken']
    print(f"{both_first_two_but_not_third} proofs where BOTH {names[0]} AND {names[1]} succeed but {names[2]} didn't")
    print(f"{either_first_two_but_not_third} proofs where EITHER {names[0]} OR {names[1]} succeed but {names[2]} didn't")
    print(f"{third_but_not_either_first_two} proofs where NEITHER {names[0]} NOR {names[1]} succeed but {names[2]} did")
    print(f"{third_but_not_both_first_two} proofs where ONE OF {names[0]} OR {names[1]} failed but {names[2]} succeeded")
    print(f"For proofs where all {len(args.reports)} reports succeeded:")
    for name, stats in zip(names, proof_stats):
        print(f"{name}: {stats.succ_proof_length_sum} sum of proof lengths, "
              f"{stats.succ_search_step_sum} sum of steps taken")

if __name__ == "__main__":
    main()
