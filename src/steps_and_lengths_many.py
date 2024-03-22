#!/usr/bin/env python

import argparse
import os
import json
from glob import glob

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("report_dirs", nargs="+")
  args = parser.parse_args()

  reports_contents = []
  for report_dir in args.report_dirs:
    report_contents = []
    for proof_file in glob(os.path.join(report_dir, "*-proofs.txt")):
      with open(proof_file, 'r') as f:
        report_contents += [json.loads(entry) for entry in f]
    reports_contents.append(report_contents)

  common_proofs = set((tuple(entry[0]) for entry in reports_contents[0] if entry[1]["status"] == "SUCCESS"))
  for report_contents in reports_contents[1:]:
    common_proofs = common_proofs.intersection((tuple(entry[0]) for entry in report_contents if entry[1]["status"] == "SUCCESS"))

  print(f"Across {len(common_proofs)} common proofs")
  for report_dir, contents in zip(args.report_dirs, reports_contents):
    print(report_dir)
    lengths_sum = sum((len(entry[1]["commands"]) for entry in contents if tuple(entry[0]) in common_proofs))
    steps_sum = sum((entry[1]["steps_taken"] for entry in contents if tuple(entry[0]) in common_proofs))
    print(f"Average length: {lengths_sum / len(common_proofs)}")
    print(f"Average # steps: {steps_sum / len(common_proofs)}")

if __name__ == "__main__":
  main()
