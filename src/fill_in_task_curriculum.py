#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from typing import Dict, List, Any, Tuple

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--subobligations", "-s", action="store_true")
    args = parser.parse_args()

    with args.input_file.open('r') as f:
        entries = [json.loads(l) for l in f]
    with args.output_file.open('w') as f:
        pass

    for entry in entries:
        for sub_obl in entry_sub_obligations(entry) if args.subobligations else [entry]:
            for tail_entry in entry_valid_tails(sub_obl):
                with args.output_file.open('a') as f:
                    print(json.dumps(tail_entry), file=f)
            with args.output_file.open('a') as f:
                print(json.dumps(sub_obl), file=f)

def entry_valid_tails(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    tail_entries: List[Dict[str, Any]] = []
    for i in reversed(range(entry["target_length"] - 1)):
        new_entry = dict(entry)
        new_entry["tactic_prefix"] = \
            new_entry["tactic_prefix"] + new_entry["orig_solution"][:i+1]
        new_entry["orig_solution"] = new_entry["orig_solution"][i+1:]
        if len([tac for tac in new_entry["orig_solution"] if tac == "{"]) != \
           len([tac for tac in new_entry["orig_solution"] if tac == "}"]):
            continue
        if new_entry["orig_solution"][0] == "{":
            continue
        new_entry["target_length"] = len([tac for tac in new_entry["orig_solution"]
                                          if tac not in ["{", "}"]])
        tail_entries.append(new_entry)
    return tail_entries

def entry_sub_obligations(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    def get_cur_obl_solution(remaining_commands: List[str]) -> List[str]:
        sol: List[str] = []
        bracket_depth = 0
        for cmd in remaining_commands:
            if cmd == "}":
                if bracket_depth == 0:
                    return sol
                bracket_depth -= 1
            if cmd == "{":
                bracket_depth += 1
            sol.append(cmd)
        return sol
    obligations: List[Tuple[List[str], List[str]]] = [([], entry["orig_solution"])]
    for cmd_idx, cmd in enumerate(entry["orig_solution"]):
        if cmd == "{":
            obligations.append(
                (entry["orig_solution"][:cmd_idx+1],
                 get_cur_obl_solution(entry["orig_solution"][cmd_idx+1:])))
    obl_entries: List[Dict[str, Any]] = []
    for prefix, sol in obligations:
        new_entry = dict(entry)
        new_entry["tactic_prefix"] = entry["tactic_prefix"] + prefix
        new_entry["orig_solution"] = sol
        new_entry["target_length"] = len([tac for tac in new_entry["orig_solution"]
                                          if tac not in ["{", "}"]])
        obl_entries.append(new_entry)

    return obl_entries

if __name__ == "__main__":
    main()
