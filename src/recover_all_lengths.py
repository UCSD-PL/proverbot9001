import csv
import json
import argparse
import itertools
from pathlib import Path
from typing import List, Dict

import coq_serapy
import util

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prelude", type=Path)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("project_dicts", type=Path)
    parser.add_argument("-v", "--verbose", action='count', default=0)
    args = parser.parse_args()
    with args.project_dicts.open('r') as f:
        project_dicts = json.loads(f.read())
    for project_dict in project_dicts:
        if args.verbose > 0:
            print(f"Project: {project_dict['project_name']}")
        project_files = [Path(f) for f in project_dict["test_files"]]
        for filename in project_files:
            if args.verbose > 0:
                print(f"Filename: {str(filename)}")
            afile = util.safe_abbrev(filename, project_files)
            proofs_file = (args.report_dir / project_dict["project_name"] /
                           (afile + "-proofs.txt"))
            csv_file = (args.report_dir / project_dict["project_name"] /
                        (afile + ".csv"))
            if not proofs_file.exists():
                print(f"Warning: couldn't find results for file {filename}, skipping...")
                continue
            with proofs_file.open('r') as f:
                proof_results = [json.loads(line) for line in f]
            with csv_file.open('r') as f:
                line = next(f)
                while line[0] == "#":
                    line = next(f)
                rest_lines = itertools.chain([line], f)
                lin_lens = {row[0].strip(): int(row[2])
                            for row in csv.reader(rest_lines)}
            orig_lens = proof_lengths(coq_serapy.load_commands(
                args.prelude / project_dict["project_name"] / filename),
                                      list(lin_lens.keys()),
                                      args.verbose)
            output_file = (args.report_dir / project_dict["project_name"] /
                           (afile + "-lengths.csv"))
            with output_file.open('w') as out:
                writer = csv.writer(out)
                writer.writerow(["proof", "original length",
                                 "linearized length", "search status", "result length"])
                for proof_result in proof_results:
                    assert proof_result[0][0] == project_dict["project_name"]
                    assert Path(proof_result[0][1]) == filename, \
                      (Path(proof_result[0][1]), filename)
                    lemma_statement = coq_serapy.kill_comments(
                        proof_result[0][3].replace("\\n", "\n")).strip()
                    if "Obligation" in lemma_statement:
                        continue
                    try:
                        orig_len = orig_lens[lemma_statement]
                    except KeyError:
                        print(f"Keys are {list(orig_lens.keys())}")
                        raise
                    # Subtracting two from the numbers in the linearized
                    # results because they count the "Proof" and "Qed"
                    writer.writerow([coq_serapy.lemma_name_from_statement(lemma_statement),
                                     orig_len,
                                     max(lin_lens[lemma_statement] - 2, 1),
                                     proof_result[1]["status"],
                                     len(proof_result[1]["commands"]) - 2])


def proof_lengths(file_commands: List[str], lemma_statements: List[str],
                  verbosity: int = 0) -> Dict[str, int]:
    result: Dict[str, int] = {}
    in_proof = False
    proof_length = None
    cur_proof = None
    todo_lemma_statements = list(lemma_statements)
    for command in file_commands:
        clean_comm = coq_serapy.kill_comments(command).strip()
        if verbosity >= 2:
            print(f"Looking at command {command}")
        if clean_comm == "Proof.":
            continue
        if in_proof:
            assert proof_length is not None
            assert cur_proof is not None
            if coq_serapy.ending_proof(command):
                in_proof = False
                # If they closed the proof without any tactics, they must have
                # used a proof term, so give it a length of 1.
                result[cur_proof] = max(proof_length, 1)
                cur_proof = None
                proof_length = None
            else:
                proof_length += 1
        if clean_comm in todo_lemma_statements:
            in_proof = True
            proof_length = 0
            cur_proof = clean_comm
            todo_lemma_statements.remove(cur_proof)
    assert len(todo_lemma_statements) == 0, f"Didn't find lemmas {todo_lemma_statements}"
    return result

if __name__ == "__main__":
    main()
