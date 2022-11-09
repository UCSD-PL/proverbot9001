
import csv
import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import util

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("project_dicts", type=Path)
    parser.add_argument("--linearized", action="store_true")
    parser.add_argument("-v", "--verbose", action='count', default=0)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--bucket-max", default=10, type=int)
    parser.add_argument("--bucket-step", default=1, type=int)
    args = parser.parse_args()

    with args.project_dicts.open('r') as f:
        project_dicts = json.loads(f.read())
    all_rows = []
    for project_dict in project_dicts:
        if args.verbose > 0:
            print(f"Project: {project_dict['project_name']}")
        project_files = [Path(f) for f in project_dict["test_files"]]
        for filename in project_files:
            afile = util.safe_abbrev(filename, project_files)
            lengths_file = (args.report_dir / project_dict["project_name"] /
                            (afile + "-lengths.csv"))
            if not lengths_file.exists():
                print(f"Warning: couldn't find results for file {filename}, skipping...")
                continue
            with lengths_file.open('r') as f:
                _header = next(f)
                file_rows = list(csv.reader(f))
                all_rows += file_rows

    print(f"Graphing {len(all_rows)} points")
    font = {'family' : 'serif',
            'weight' : 'bold',
            'size'   : 22}

    plt.rc('font', **font)
    buckets = list(range(0, args.bucket_max, args.bucket_step))
    bars_vals = [list([0, 0, 0]) for _ in buckets]
    for row in all_rows:
        length = int(row[2 if args.linearized else 1])
        if length > args.bucket_max:
            continue
        for bucket_idx in range(len(buckets)):
            if length >= buckets[bucket_idx] and (bucket_idx == len(buckets) - 1 or
                                                  length < buckets[bucket_idx+1]):
                if row[3] == "SUCCESS":
                    bars_vals[bucket_idx][0] += 1
                elif row[3] == "INCOMPLETE":
                    bars_vals[bucket_idx][1] += 1
                elif row[3] == "FAILURE":
                    bars_vals[bucket_idx][2] += 1
                break
    width = 0.5 * args.bucket_step
    succ_vals, inc_vals, fail_vals = zip(*bars_vals)
    plt.bar(buckets, succ_vals, width, label="Success")
    plt.bar(buckets, inc_vals, width, bottom=succ_vals, label="Incomplete")
    plt.bar(buckets, fail_vals, width, bottom=[s + i for s, i in zip(succ_vals, inc_vals)] ,
            label="Failure")
    plt.xlabel("Length (Number of Tactics)")
    plt.ylabel("Number of Proofs")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
