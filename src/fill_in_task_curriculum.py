#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    with args.input_file.open('r') as f:
        entries = [json.loads(l) for l in f]
    with args.output_file.open('w') as f:
        pass

    for entry in entries:
        for i in reversed(range(entry["target_length"] - 1)):
            new_entry = dict(entry)
            new_entry["tactic_prefix"] = \
                new_entry["tactic_prefix"] + new_entry["orig_solution"][:i+1]
            new_entry["orig_solution"] = new_entry["orig_solution"][i+1:]
            new_entry["target_length"] = len(new_entry["orig_solution"])
            with args.output_file.open('a') as f:
                print(json.dumps(new_entry), file=f)
        with args.output_file.open('a') as f:
            print(json.dumps(entry), file=f)


if __name__ == "__main__":
    main()
