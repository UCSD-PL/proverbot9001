#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import TypedDict, Dict, Optional, Tuple, Union

Entry = TypedDict('Entry', {"filename": str, "proofname": str,
                            "result": str, "time": Union[float, str]}, 
                  total=False)
ProofSpec = Tuple[str, str]
ProofResult = TypedDict('ProofResult', {"result": str, "time": Union[float, str]})

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("output_json", type=Path)
  parser.add_argument("input_jsons", nargs="+", type=Path)
  args = parser.parse_args()

  combined_results: Dict[ProofSpec, Optional[ProofResult]] = {}
  for input_file in args.input_jsons:
    with input_file.open('r') as f:
      for entry_dict in json.load(f)["results"]:
        # The Entry constructor here basically functions as a runtime type
        # assertion. Other than that, it's a no op, taking a dict and producing
        # a dict with the same keys and values.
        entry: Entry = Entry(**entry_dict) #type: ignore [typeddict-item]
        spec = entry_to_proofspec(entry)
        result = entry_to_proofresult(entry)
        if (spec not in combined_results or
           combined_results[spec] is None or
           isinstance(combined_results[spec]["time"], str) or #type: ignore [index]
           (result is not None and
            isinstance(result["time"], float) and
            combined_results[spec]["time"] > #type: ignore [index]
            result["time"])): #type: ignore[operator]
          combined_results[spec] = result

  with args.output_json.open("w") as f:
    print(json.dumps({"results": [entry_from_parts(spec, result)
                                  for spec, result in combined_results.items()]}),
          file=f)

def entry_to_proofspec(entry: Entry) -> ProofSpec:
  assert "filename" in entry
  assert "proofname" in entry
  return (entry["filename"], entry["proofname"])

def entry_to_proofresult(entry: Entry) -> Optional[ProofResult]:
  if "result" not in entry:
    return None
  assert "time" in entry
  return ProofResult(result=entry["result"],
                     time=entry["time"])

def entry_from_parts(spec: ProofSpec, result: Optional[ProofResult]) -> Entry:
  entry: Entry = Entry(filename=spec[0], proofname=spec[1])
  if result is not None:
    entry["result"] = result["result"]
    entry["time"] = result["time"]
  return entry

if __name__ == "__main__":
  main()
