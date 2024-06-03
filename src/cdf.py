#!/usr/bin/env python

import argparse
import os
import json
from pathlib import Path
from typing import List, Tuple
from glob import glob
from collections import Counter

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("outfile", type=Path)
    parser.add_argument("--mode", choices=["steps", "time"], default="steps")
    args = parser.parse_args()

    data = load_proof_data(args)
    cum_data = cumulate_data(data)
    with args.outfile.open('w') as f:
        for x, y in cum_data:
            print(x, y, file=f)

def cumulate_data(data: List[float]) -> List[Tuple[float, float]]:
    cumulative_points: List[Tuple[float,float]] = []
    point_counts = Counter(data)
    total_so_far = 0
    for x, y in sorted(point_counts.items(), key=lambda item: item[0]):
        total_so_far += y
        cumulative_points.append((x, total_so_far))
    return cumulative_points

def load_proof_data(args: argparse.Namespace) -> List[float]:
    run_dir = os.getcwd()
    os.chdir(args.report)
    files = glob("**/*-proofs.txt", recursive=True)
    os.chdir(run_dir)

    steps_or_times: List[float] = []

    for filename in files:
        with open(os.path.join(args.report, filename), 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry[1]["status"] == "SUCCESS":
                    if args.mode == "steps":
                        steps_or_times.append(entry[1]["steps_taken"])
                    else:
                        assert args.mode == "time"
                        assert "time_taken" in entry[1],\
                            "This report doesn't have the time taken in all its entries!"
                        steps_or_times.append(entry[1]["time_taken"])
    return steps_or_times

if __name__ == "__main__":
    main()
