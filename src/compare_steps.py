#!/usr/bin/env python3

import argparse
import os.path
import json
import csv
from glob import glob
from pathlib import Path

from typing import List

import coq_serapy

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reporta")
    parser.add_argument("reportb")
    parser.add_argument("--print-a-shorter", action='store_true')
    parser.add_argument("--print-a-only", action="store_true")
    parser.add_argument("--print-b-shorter", action='store_true')
    parser.add_argument("--print-b-only", action="store_true")
    parser.add_argument("--full-csv", default=None)
    parser.add_argument("--a-name", default="Report A")
    parser.add_argument("--b-name", default="Report B")
    args = parser.parse_args()

    compare_steps(args)

def compare_steps(args: argparse.Namespace):

    a_succ_steps = 0
    a_shorter = 0
    b_succ_steps = 0
    b_shorter = 0
    a_succ_proof_steps = 0
    a_proof_shorter = 0
    b_succ_proof_steps = 0
    b_proof_shorter = 0
    same_length = 0
    a_succ_not_b = 0
    b_succ_not_a = 0
    both_succ = 0

    if args.full_csv:
        with open(args.full_csv, 'w', newline='') as csvfile:
            headerwriter = csv.writer(csvfile)
            headerwriter.writerow(["file", "module_prefix", "lemma_name",
                                   f"{args.a_name.lower()}_steps_searched",
                                   f"{args.a_name.lower()}_solution_length",
                                   f"{args.b_name.lower()}_steps_searched",
                                   f"{args.b_name.lower()}_solution_length"])

    for filename_a in glob(os.path.join(args.reporta, "*-proofs.txt")):
        with open(filename_a, 'r') as f:
            proof_data_a = [json.loads(line) for line in f]
        filename_b = os.path.join(args.reportb, os.path.basename(filename_a))
        try:
            with open(filename_b) as f:
                proof_data_b = [json.loads(line) for line in f]
        except FileNotFoundError:
            print(f"Couldn't find file in directory {args.reportb} "
                  f"cooresponding to {filename_a}")
            raise

        b_dict = {(line[0][2], line[0][3]): line
                  for line in proof_data_b}

        for job, sol_a in proof_data_a:
            try:
                job_b, sol_b = b_dict[(job[2], job[3])]
                assert job_eq(job_b, job), (job, job_b)
            except KeyError:
                print(f"Warning: couldn't find job {(job[2], job[3])} from file {filename_a} "
                      f"in filename {filename_b}")
                continue

            lemma_name = coq_serapy.lemma_name_from_statement(job[3])
            if args.full_csv:
                if sol_a['status'] == "SUCCESS" or sol_b['status'] == "SUCCESS":
                    with open(args.full_csv, 'a', newline='') as csvfile:
                        rowwriter = csv.writer(csvfile)
                        rowwriter.writerow([
                            job[1], job[2],
                            lemma_name,
                            sol_a['steps_taken'] if sol_a["status"] == "SUCCESS" else None,
                            len(sol_a['commands']) - 2 if sol_a["status"] == "SUCCESS" else None,
                            sol_b['steps_taken'] if sol_b["status"] == "SUCCESS" else None,
                            len(sol_b['commands']) - 2 if sol_b["status"] == "SUCCESS" else None,
                            ])
            if sol_a["status"] == "SUCCESS" and sol_b["status"] == "SUCCESS":
                if (args.print_a_shorter and sol_a['steps_taken'] < sol_b['steps_taken']) or \
                   (args.print_b_shorter and sol_b['steps_taken'] < sol_a['steps_taken']):
                    print(f"For job {job[1]}:{job[2]}:{lemma_name}, "
                          f"{args.a_name} took {sol_a['steps_taken']} steps, "
                          f"{args.b_name} took {sol_b['steps_taken']+1} steps.")
                a_succ_proof_steps += len(sol_a['commands']) - 2
                b_succ_proof_steps += len(sol_b['commands']) - 2
                a_succ_steps += sol_a['steps_taken']
                b_succ_steps += sol_b['steps_taken']
                both_succ += 1
                if sol_a['steps_taken'] < sol_b['steps_taken']:
                    a_shorter += 1
                if sol_b['steps_taken'] < sol_a['steps_taken']:
                    b_shorter += 1
                else:
                    same_length += 1
                if len(sol_a['commands']) < len(sol_b['commands']):
                    a_proof_shorter += 1
                elif len(sol_b['commands']) < len(sol_a['commands']):
                    b_proof_shorter += 1
            elif sol_b["status"] == "SUCCESS":
                b_succ_not_a += 1
                if args.print_b_only:
                    print(f"For job {job[1]}:{job[2]}:{lemma_name}, "
                          f"Only {args.b_name} succeeded.")
            #     print("Only second report succeeded")
            elif sol_a["status"] == "SUCCESS":
                a_succ_not_b += 1
                if args.print_a_only:
                    print(f"For job {job[1]}:{job[2]}:{lemma_name}, "
                          f"Only {args.a_name} succeeded.")
            #     print("Only first report succeeded")

    print(f"Total steps: {a_succ_steps} ({args.a_name}) vs {b_succ_steps} ({args.b_name})")
    print(f"Total solution lengths: {a_succ_proof_steps} ({args.a_name}) vs "
          f"{b_succ_proof_steps} ({args.b_name})")
    print(f"{a_shorter} proofs where {args.a_name} was took fewer steps, "
          f"{b_shorter} proofs where {args.b_name} was took fewer steps, "
          f"{same_length} proofs where they were the same")
    print(f"{a_proof_shorter} proofs where {args.a_name}'s solution was shorter, "
          f"{b_proof_shorter} proofs where {args.b_name}'s solution was shorter, ")
    print(f"{a_succ_not_b} proofs where {args.a_name} succeeded but {args.b_name} did not.")
    print(f"{b_succ_not_a} proofs where {args.b_name} succeeded but {args.a_name} did not.")
    print(f"{both_succ} proofs where both succeeded")

def job_eq(job1: List[str], job2: List[str]) -> bool:
    assert len(job1) == 4
    assert len(job2) == 4
    return job1[0] == job2[0] and \
            Path(job1[1]) == Path(job2[1]) and \
            job1[2] == job2[2] and \
            job1[3] == job2[3]

if __name__ == "__main__":
    main()
