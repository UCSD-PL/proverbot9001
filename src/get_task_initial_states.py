#!/usr/bin/env python
import argparse
import json
from typing import List
from pathlib import Path
from tqdm import tqdm

from coq_serapy import Obligation

from gen_rl_tasks import RLTask
from rl import FileReinforcementWorker
from util import eprint

def main() -> None:
  argparser = argparse.ArgumentParser()
  argparser.add_argument("-v", "--verbose", action='count', default=0)
  argparser.add_argument("--print-timings", action='store_true')
  argparser.add_argument("--prelude", default=".", type=Path)
  argparser.add_argument("--no-resume", action='store_false', dest="resume")
  argparser.add_argument("tasks_file", type=Path)
  argparser.add_argument("output_file", type=Path)
  args = argparser.parse_args()
  args.backend = "serapi"
  args.set_switch = True

  with args.tasks_file.open("r") as f:
    tasks = [RLTask(**json.loads(line)) for line in f]
  if args.output_file.exists():
    if args.resume:
      with args.output_file.open('r') as f:
        done_tasks = [RLTask(**task_dict) for l in f
                      for task_dict, obl_dict in (json.loads(l),)]
      tasks = list(set(tasks).difference(set(done_tasks)))
      eprint(f"Resumed from file {str(args.output_file)}")
    else:
      with args.output_file.open("w") as f:
        pass
  with args.output_file.open("a") as f:
    with FileReinforcementWorker(args, None) as worker:
      for task in tqdm(tasks, desc="Retrieving proof states"):
        worker.run_into_task(task.to_job(), task.tactic_prefix)
        print(json.dumps((
          task.as_dict(),
          worker.coq.proof_context.fg_goals[0].to_dict())), file=f)

if __name__ == "__main__":
  main()
