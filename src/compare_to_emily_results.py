import argparse
import json
import os
from tqdm import tqdm
from glob import glob

from coq_serapy import lemma_name_from_statement

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("emily_style_json")
  parser.add_argument("result_folder")
  args = parser.parse_args()

  compare(args)

def es_to_tuple(es_datum: dict[str, str]) -> tuple[str, str, str]:
  return (es_datum["filename"].split("/")[2],
          os.path.splitext(os.path.join(*es_datum["filename"].split("/")[3:]))[0] + ".v",
          es_datum["proofname"])

def get_es_successes(args: argparse.Namespace) -> list[tuple[str, str, str]]:
  with open(args.emily_style_json, 'r') as f:
    es_data = json.loads(f.read())["results"]
  es_all = [es_to_tuple(es_datum) for es_datum in es_data]
  es_successes = [es_to_tuple(es_datum) for es_datum in es_data if "result" in es_datum]
  return es_all, es_successes

def get_ps_successes(args: argparse.Namespace) -> list[tuple[str, str, str]]:
  run_dir = os.getcwd()
  os.chdir(args.result_folder)
  files = glob("**/*-proofs.txt", recursive=True)
  os.chdir(run_dir)

  ps_data = []
  for filename in tqdm(files):
    with open(os.path.join(args.result_folder, filename), 'r') as f:
      ps_file_data = [json.loads(line) for line in f]
    ps_data += ps_file_data
  ps_all = [(datum[0][0], datum[0][1], lemma_name_from_statement(datum[0][3]))
            for datum in ps_data]
  ps_successes = [(datum[0][0], datum[0][1], lemma_name_from_statement(datum[0][3]))
                  for datum in ps_data if datum[1]["status"] == "SUCCESS"]
  return ps_all, ps_successes

def compare(args: argparse.Namespace) -> None:
  es_all, es_successes = get_es_successes(args)
  print(f"Emily-style data: {len(es_all)} entries, {len(es_successes)} successes")
  ps_all, ps_successes = get_ps_successes(args)
  print(f"Proverbot-style data: {len(ps_all)} entries, {len(ps_successes)} successes")
  ps_overlap_all = [entry for entry in ps_all if entry in es_all]
  ps_overlap_successes = [entry for entry in ps_successes if entry in es_all]
  print(f"Proverbot-style overlap data: {len(ps_overlap_all)} entries, {len(ps_overlap_successes)} successes")
  ps_non_overlap_all = [entry for entry in ps_all if entry not in es_all]
  both_successes = [entry for entry in ps_successes if entry in es_successes]
  es_only = [entry for entry in es_successes if entry not in ps_successes and entry in ps_all]
  ps_only = [entry for entry in ps_overlap_successes if entry not in es_successes]
  print(f"Of overlap, {len(both_successes)} where both succeed, "
        f"{len(es_only)} where only emily-style data succeeds, "
        f"{len(ps_only)} where only proverbot-style data succeeds")
  

if __name__ == "__main__":
  main()
