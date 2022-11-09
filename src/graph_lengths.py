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
                rows = list(csv.reader(f))
                all_rows += [row for row in rows if row[3] == "SUCCESS"]

    print(f"Graphing {len(rows)} points")
    font = {'family' : 'serif',
            'weight' : 'bold',
            'size'   : 22}

    plt.rc('font', **font)
    xs = [int(row[2 if args.linearized else 1]) for row in all_rows]
    ys = [int(row[4]) for row in all_rows]
    plt.scatter(xs, ys, alpha=0.2)
    plt.plot([0, max(xs + ys)], [0, max(xs + ys)], color="green")
    plt.xticks(list(range(0, max(xs), 5)))
    plt.yticks(list(range(0, max(ys), 5)))
    plt.xlabel("Generated proof length")
    plt.tight_layout()
    if args.linearized:
        plt.ylabel("Original proof length (linearized)")
    else:
        plt.ylabel("Original proof length")
    if args.output:
        plt.savefig(args.output, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    main()
