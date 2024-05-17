#!/usr/bin/env python

import argparse
import json
import re
import os
from collections import Counter
from tqdm import tqdm
from glob import glob, iglob

from coq_serapy import lemma_name_from_statement

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("emily_style_json")
  parser.add_argument("result_folder")
  parser.add_argument("--all-theorems", default=None)
  args = parser.parse_args()

  compare(args)

def es_normalize_proofname(proofname: str) -> str:
  # Emily First says this is correct
  if proofname == "degree_aux_tcc":
    return "degree_aux"
  # Work around an obligation-mis-numbering bug in astactic-based tools. I got
  # these correspondences by going through the files manually.
  if proofname == "receive_prep_req_loop_obligation_5":
    return "receive_prep_req_loop_obligation_4"
  if proofname == "receive_commabrt_loop_obligation_5":
    return "receive_commabrt_loop_obligation_4"
  if proofname == "compute_list_f_obligation_3":
    return "compute_list_f_obligation_1"
  if proofname == "compute_list_f_obligation_4":
    return "compute_list_f_obligation_2"
  if proofname == "receive_prep_loop_obligation_5":
    return "receive_prep_loop_obligation_4"
  if proofname == "receive_req_loop_obligation_4":
    return "receive_req_loop_obligation_2"
  if proofname == "receive_resp_loop_obligation_5":
    return "receive_resp_loop_obligation_4"
  return proofname

def es_to_tuple(es_datum: dict[str, str]) -> tuple[str, str, str]:
  return (es_datum["filename"].split("/")[2],
          os.path.splitext(os.path.join(*es_datum["filename"].split("/")[3:]))[0] + ".v",
          es_normalize_proofname(es_datum["proofname"]))

def get_es_successes(args: argparse.Namespace) -> list[tuple[str, str, str]]:
  with open(args.emily_style_json, 'r') as f:
    es_data = json.loads(f.read())["results"]
  es_all = [es_to_tuple(es_datum) for es_datum in es_data]
  es_successes = [es_to_tuple(es_datum) for es_datum in es_data if "result" in es_datum and es_datum["result"] == "success"]
  return es_all, es_successes

def nname_from_statement(statement: str) -> str:
  obl_match = re.match(r"(.*)Obligation\s+(\d+)\.", statement, flags=re.DOTALL)
  if obl_match:
    bare_name = lemma_name_from_statement(obl_match.group(1) + ".")
    new_name = bare_name + "_obligation_" + str(int(obl_match.group(2)) + 1)
    return new_name
  else:
    bare_name = lemma_name_from_statement(statement)
    return bare_name

def get_ps_successes(args: argparse.Namespace) -> list[tuple[str, str, str]]:
  run_dir = os.getcwd()
  os.chdir(args.result_folder)
  files = list(tqdm(iglob("**/*-proofs.txt", recursive=True)))
  os.chdir(run_dir)

  ps_data = []
  for filename in tqdm(files):
    with open(os.path.join(args.result_folder, filename), 'r') as f:
      ps_file_data = [json.loads(line) for line in f]
    ps_data += ps_file_data
  ps_all = [(datum[0][0], datum[0][1], nname_from_statement(datum[0][3]))
            for datum in ps_data]
  ps_successes = [(datum[0][0], datum[0][1], nname_from_statement(datum[0][3]))
                  for datum in ps_data if datum[1]["status"] == "SUCCESS"]
  return ps_all, ps_successes

def fix_filename(filename: str) -> str:
  return filename.replace("Minimun", "Minimum")

def compare(args: argparse.Namespace) -> None:
  es_all, es_successes = get_es_successes(args)
  print(f"Emily-style data: {len(es_all)} entries, {len(es_successes)} successes")
  ps_all, ps_successes = get_ps_successes(args)
  print(f"Proverbot-style data: {len(ps_all)} entries, {len(ps_successes)} successes")

  both_successes = [entry for entry in ps_successes if entry in es_successes]
  if args.all_theorems:
    with open(args.all_theorems, 'r') as f:
      all_theorems = [(obj[0].split("/")[2],
                      os.path.splitext(os.path.join(*obj[0].split("/")[3:]))[0] + ".v",
                      es_normalize_proofname(obj[1])) for obj in json.loads(f.read())]
    ps_overlap_all = [entry for entry in ps_all if entry in all_theorems]
    ps_overlap_successes = [entry for entry in ps_successes if entry in all_theorems]
    print(f"Proverbot-style overlap data: {len(ps_overlap_all)} entries, {len(ps_overlap_successes)} successes")
    ps_non_overlap_all = [entry for entry in ps_all if entry not in all_theorems]
    all_theorems_non_overlap = [entry for entry in all_theorems if entry not in ps_all]
    es_only = [entry for entry in es_successes if entry not in ps_successes and entry in ps_all]
    ps_only = [entry for entry in ps_overlap_successes if entry not in es_successes]
    print(f"Using all_theorems.json: Of overlap, {len(both_successes)} where both succeed, "
          f"{len(es_only)} where only emily-style data succeeds, "
          f"{len(ps_only)} where only proverbot-style data succeeds")

    print(f"There are {len(all_theorems_non_overlap)} theorems in the all theorems file "
          f"that aren't in the Proverbot9001-style data.")
    # proj_counts = Counter([entry[0] for entry in all_theorems_non_overlap])
    # # print(proj_counts)
    # file_counts = Counter([entry[1] for entry in all_theorems_non_overlap if entry[0] == "buchberger"])
    # # print(file_counts)
    # file_proofs = [entry[2] for entry in all_theorems_non_overlap if entry[0] == "buchberger" and entry[1] == "BuchAux.v"]
    proj_counts = Counter([entry[0] for entry in ps_non_overlap_all])
    # print(proj_counts)
    file_counts = Counter([entry[1] for entry in ps_non_overlap_all if entry[0] == "coquelicot"])
    # print(file_counts)
    file_proofs = [entry[2] for entry in ps_non_overlap_all if entry[0] == "coquelicot" and entry[1] == "theories/Derive.v"]
    print(file_proofs[:10])
  else:
    ps_overlap_all = [entry for entry in ps_all if entry in es_all]
    ps_overlap_successes = [entry for entry in ps_successes if entry in es_all]
    print(f"Proverbot-style overlap data: {len(ps_overlap_all)} entries, {len(ps_overlap_successes)} successes")
    ps_non_overlap_all = [entry for entry in ps_all if entry not in es_all]
    es_only = [entry for entry in es_successes if entry not in ps_successes and entry in ps_all]
    ps_only = [entry for entry in ps_overlap_successes if entry not in es_successes]
    print(f"Of overlap, {len(both_successes)} where both succeed, "
          f"{len(es_only)} where only emily-style data succeeds, "
          f"{len(ps_only)} where only proverbot-style data succeeds")
  

if __name__ == "__main__":
  main()
