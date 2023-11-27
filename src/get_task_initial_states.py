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
      start_bar_from = len(done_tasks)
      tasks = list(set(tasks).difference(set(done_tasks)))
      eprint(f"Resumed from file {str(args.output_file)}")
    else:
      with args.output_file.open("w") as f:
        pass
      start_bar_from = 0
  with args.output_file.open("a") as f:
    cur_project = ""
    cur_file = ""
    tasks_left = tasks[1:]
    next_task = tasks[0]
    with tqdm(desc="Retrieving proof states",
              initial=start_bar_from,
              total=len(tasks) + start_bar_from,
              dynamic_ncols=True) as bar:
      while len(tasks_left) > 0:
        with FileReinforcementWorker(args, None) as worker:
          while next_task.project == cur_project and \
                next_task.src_file == cur_file:
            worker.run_into_task(next_task.to_job(), next_task.tactic_prefix)
            print(json.dumps((
              next_task.as_dict(),
              worker.coq.proof_context.fg_goals[0].to_dict())), file=f)
            next_task = tasks_left.pop(0)
            bar.update()

if __name__ == "__main__":
  main()
