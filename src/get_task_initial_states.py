#!/usr/bin/env python
import argparse
import json
from typing import List
from pathlib import Path
from tqdm import tqdm

from coq_serapy import Obligation

from gen_rl_tasks import RLTask
from rl import FileReinforcementWorker

def main() -> None:
  argparser = argparse.ArgumentParser()
  argparser.add_argument("-v", "--verbose", action='count', default=0)
  argparser.add_argument("--print-timings", action='store_true')
  argparser.add_argument("--prelude", default=".", type=Path)
  argparser.add_argument("tasks_file", type=Path)
  argparser.add_argument("output_file", type=Path)
  args = argparser.parse_args()
  args.backend = "serapi"
  args.set_switch = True

  with args.tasks_file.open("r") as f:
    tasks = [RLTask(**json.loads(line)) for line in f]
  obls = get_starting_obls(args, tasks)
  with args.output_file.open("w") as f:
    for obl, task in zip(obls, tasks):
      print(json.dumps((task.as_dict(), obl.to_dict())), file=f)
  
def get_starting_obls(args: argparse.Namespace, tasks: List[RLTask])\
   -> List[Obligation]:
  results: List[Obligation] = []
  with FileReinforcementWorker(args, None) as worker:
    for task in tqdm(tasks, desc="Retrieving proof states"):
      worker.run_into_task(task.to_job(), task.tactic_prefix)
      results.append(worker.coq.proof_context.fg_goals[0])
  return results

  pass

if __name__ == "__main__":
  main()
