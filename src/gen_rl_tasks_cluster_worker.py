#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from os import environ

from gen_rl_tasks import add_args_to_parser, get_job_tasks
from search_worker import ReportJob
from search_file import get_predictor

from util import FileLock, eprint

def main() -> None:
    assert 'SLURM_ARRAY_TASK_ID' in environ
    workerid = int(environ['SLURM_ARRAY_TASK_ID'])

    arg_parser = argparse.ArgumentParser()
    add_args_to_parser(arg_parser)
    arg_parser.add_argument("--num-workers", default=32, type=int)
    arg_parser.add_argument("--workers-output-dir", default=Path("output"),
                            type=Path)
    arg_parser.add_argument("--worker-timeout", default="6:00:00")
    arg_parser.add_argument("--partition", default="defq")
    arg_parser.add_argument("--mem", default="2G")
    args = arg_parser.parse_args()
    args.output_dir = Path(str(args.output_file) + ".d")

    with (args.output_dir / "workers_scheduled.txt").open('a') as f, FileLock(f):
        print(workerid, file=f)
    gen_rl_tasks_worker(args, workerid)
    eprint(f"Finished worker {workerid}")

def gen_rl_tasks_worker(args: argparse.Namespace, workerid: int) -> None:
    predictor = get_predictor(args)
    with (args.output_dir / "jobs.txt").open('r') as f:
        all_jobs = [ReportJob(*json.loads(line)) for line in f]
        pass
    while True:
        with (args.output_dir / "taken.txt").open('r+') as f, FileLock(f):
            taken_jobs = [ReportJob(*json.loads(line)) for line in f]
            current_job = None
            for job in all_jobs:
                if job not in taken_jobs:
                    current_job = job
                    break
            if current_job:
                eprint(f"Starting job {current_job}")
                print(json.dumps(current_job), file=f, flush=True)
            else:
                eprint(f"Finished worker {workerid}")
                break

        tasks = get_job_tasks(args, predictor, current_job)

        with (args.output_dir / f"output-{workerid}.json").open('a') as f, FileLock(f):
            eprint(f"Finished job {current_job}")
            for task in tasks:
                print(json.dumps(task.as_dict()), file=f)
        with (args.output_dir / f"worker-{workerid}-done.json").open('a') as f:
            print(json.dumps(current_job), file=f)

if __name__ == "__main__":
    main()
