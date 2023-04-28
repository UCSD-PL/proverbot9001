#!/usr/bin/env python3

import json
import argparse
import sys

from typing import Dict
from pathlib import Path

from tqdm import tqdm

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("-o", "--outfile", default="-")
  parser.add_argument("input_file", type=Path)
  parser.add_argument("-v", "--verbose", dest="verbosity", action='count', default=0)
  args = parser.parse_args()
  
  entries: Dict[str, Dict[str, List[List[str]]]] = {}
  with args.input_file.open("r") as f:
    for idx, line in enumerate(tqdm(f)):
      try:
        entry = json.loads(line)
      except json.decoder.JSONDecodeError:
        if args.verbosity > 0:
          print(f"Ignoring incomplete entry {repr(line)} at line {idx+1}")
        continue
      if not entry["Module/Section"] in entries:
        entries[entry["Module/Section"]] = {}
      if not entry["Proof"] in entries[entry["Module/Section"]]:
        entries[entry["Module/Section"]][entry["Proof"]] = []
      entries[entry["Module/Section"]][entry["Proof"]].append(entry["Tactics"])

  if args.outfile == "-":
    f = sys.stdout
  else:
    f = open(args.outfile, 'w')

  for sm_prefix, proofs in entries.items():
    for proof, tactics in proofs.items():
      print(json.dumps({"Module/Section": sm_prefix,
                        "Proof": proof,
                        "Tactic Traces": tactics}), file=f)

  if args.outfile != "-":
      f.close()

if __name__ == "__main__":
    main()
