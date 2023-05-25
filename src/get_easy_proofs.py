#!/usr/bin/env python

import argparse
import json
import os.path
from glob import glob
from pathlib import Path

import coq_serapy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir")
    parser.add_argument("-l", "--max-length", type=int, default=None)
    parser.add_argument("-f", "--prooffile", default=None)
    parser.add_argument("-d", "--disambiguate", action="store_true")
    args = parser.parse_args()

    if args.prooffile:
        files = [f"{args.report_dir}/{os.path.splitext(args.prooffile)[0]}-proofs.txt"]
    else:
        files = glob(f"{args.report_dir}/*-proofs.txt")

    with open("/home/zhannakaufma_umass_edu/work/proverbot9001/" + args.prooffile) as f:
        project_dict = json.load(f)
    files = [f"{args.report_dir}/{os.path.splitext(prooffile.split('/')[-1])[0]}-proofs.txt" for prooffile in project_dict[0]["test_files"]]
    #files = [os.path.join(Path("/home/zhannakaufma_umass_edu/work/proverbot9001/CompCert"), Path(afile)) for afile in project_dict[0]["train_files"]]

    for path in files:
        with open(path) as f:
            file_proofs = [json.loads(l) for l in f]
            for job, solution in file_proofs:
                if solution["status"] != "SUCCESS":
                    continue
                stripped_solution = [cmd for cmd in solution["commands"]
                                     if cmd["tactic"] not in ["Qed.", "Proof.", "{", "}"]]
                if args.max_length and len(stripped_solution) > args.max_length:
                    continue
                lemma_name = coq_serapy.lemma_name_from_statement(job[3])
                if args.disambiguate:
                    print(job[2] + lemma_name)
                else:
                    print(lemma_name)


if __name__ == "__main__":
    main()
