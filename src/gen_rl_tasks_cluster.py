#!/usr/bin/env python3

import argparse
import os
import json
from json.decoder import JSONDecodeError
import subprocess
import sys
import signal
import time
from pathlib import Path
from glob import glob

from typing import List

from tqdm import tqdm

import util

with util.print_time("Importing"):
    from gen_rl_tasks import add_args_to_parser
    from search_file_cluster import get_all_jobs_cluster
    from search_worker import ReportJob

def main():
    parser = argparse.ArgumentParser(
        description="Generate demonstrating-tasks up to a given length "
        "for training an rl agent refining a given predictor, "
        "but using a cluster managed via slurm")
    add_args_to_parser(parser)
    parser.add_argument("--num-workers", default=32, type=int)
    parser.add_argument("--workers-output-dir", default=Path("output"),
                            type=Path)
    parser.add_argument("--worker-timeout", default="6:00:00")
    parser.add_argument("--partition", default="cpu")
    parser.add_argument("--mem", default="2G")
    args = parser.parse_args()
    args.output_dir = Path(str(args.output_file) + ".d")
    args.splits_file = args.json_project_file

    os.makedirs(str(args.output_dir), exist_ok=True)
    if args.resume:
        solved_jobs = get_already_done_jobs(args)
    else:
        remove_already_done_jobs(args)
        solved_jobs = []
    os.makedirs(str(args.output_dir / args.workers_output_dir), exist_ok=True)
    get_all_jobs_cluster(args, partition=args.data_partition + "_files")
    with open(args.output_dir / "all_jobs.txt") as f:
        jobs = [ReportJob(*json.loads(line)) for line in f]
        jobs = [job for job in jobs if "Obligation" not in job.lemma_statement]
        assert len(jobs) > 0
    if len(solved_jobs) < len(jobs):
        if args.just_print_jobs:
            for job in jobs:
                if job not in solved_jobs:
                    print(job)
            sys.exit(0)
        setup_jobsstate(args.output_dir, jobs, solved_jobs)
        dispatch_workers(args, sys.argv[1:])
        with util.sighandler_context(signal.SIGINT, interrupt_early):
            show_progress(args)
        cancel_workers()
    else:
        assert len(solved_jobs) == len(jobs), \
          f"There are {len(solved_jobs)} solved jobs but only {len(jobs)} jobs total detected"
        util.eprint(f"Already done! {len(solved_jobs)} solved jobs")
    with args.output_file.open("w") as out_f:
        for filename in glob(str(args.output_dir / "output-*.json")):
            with open(filename, 'r') as in_f:
                for line in in_f:
                    print(line, end="", file=out_f)

def setup_jobsstate(output_dir: Path, all_jobs: List[ReportJob],
                    solved_jobs: List[ReportJob]) -> None:
    with (output_dir / "taken.txt").open("w") as f:
        pass
    with (output_dir / "jobs.txt").open("w") as f:
        for job in all_jobs:
            if job not in solved_jobs:
                print(json.dumps(job), file=f)
        print("", end="", flush=True, file=f)
def dispatch_workers(args: argparse.Namespace, rest_args: List[str]) -> None:
    with (args.output_dir / "num_workers_dispatched.txt").open("w") as f:
        print(args.num_workers, file=f)
    with (args.output_dir / "workers_scheduled.txt").open("w") as f:
        pass
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    num_workers_left = args.num_workers
    # For some reason it looks like the maximum job size array is 1001 for
    # slurm, so we're going to use batches of 1000
    while num_workers_left > 0:
        num_dispatching_workers = min(num_workers_left, 1000)
        # If you have a different cluster management software, that still allows
        # dispatching jobs through the command line and uses a shared filesystem,
        # modify this line.
        subprocess.run([f"{cur_dir}/sbatch-retry.sh",
                        "-J", "proverbot9001-task-worker",
                        "-p", args.partition,
                        "-t", str(args.worker_timeout),
                        "-o", str(args.output_dir / args.workers_output_dir
                                  / "worker-%a.out"),
                        "--mem", args.mem,
                        f"--array=0-{num_dispatching_workers-1}",
                        f"{cur_dir}/gen_rl_tasks_cluster_worker.sh"] + rest_args,
                       check=False)
        num_workers_left -= num_dispatching_workers

def show_progress(args: argparse.Namespace) -> None:
    with (args.output_dir / "all_jobs.txt").open('r') as f:
        all_jobs = [ReportJob(*json.loads(l)) for l in f]
        all_jobs = [job for job in all_jobs if "Obligation" not in job.lemma_statement]
    with (args.output_dir / "num_workers_dispatched.txt").open('r') as f:
        num_workers_total = int(f.read())
    with (args.output_dir / "workers_scheduled.txt").open('r') as f:
        workers_scheduled = list(f)
    jobs_done = get_already_done_jobs(args)
    crashed_workers: List[int] = []
    with tqdm(desc="Jobs finished", total=len(all_jobs), initial=len(jobs_done), dynamic_ncols=True) as bar, \
         tqdm(desc="Workers scheduled", total=num_workers_total, initial=len(workers_scheduled), dynamic_ncols=True) as wbar:
        while len(jobs_done) < len(all_jobs):
            new_workers_alive = [int(wid) for wid in
                                 subprocess.check_output(
                                   "squeue -r -u$USER -h -n proverbot9001-task-worker -o%K",
                                   shell=True, text=True).strip().split("\n")
                                 if wid != ""]
            time.sleep(0.2)
            new_jobs_done = get_already_done_jobs(args)
            bar.update(len(new_jobs_done) - len(jobs_done))
            jobs_done = new_jobs_done
            if len(new_workers_alive) == 0:
                time.sleep(1)
                if len(jobs_done) < len(all_jobs):
                    util.eprint("All workers exited, but jobs aren't done!")
                    sys.exit(1)
            with (args.output_dir / "workers_scheduled.txt").open('r') as f:
                new_workers_scheduled = list(f)
            wbar.update(len(new_workers_scheduled) - len(workers_scheduled))
            workers_scheduled = new_workers_scheduled
def interrupt_early(*args) -> None:
    cancel_workers()
    sys.exit(0)

def cancel_workers() -> None:
    subprocess.run(["scancel -u $USER -n proverbot9001-task-worker"], shell=True, check=True)

def get_already_done_jobs(args: argparse.Namespace) -> List[ReportJob]:
    try:
        jobs = []
        for filename in glob(str(args.output_dir / "*-done.json")):
            with open(filename, 'r') as f:
                for line_num, line in enumerate(f):
                    try:
                        jobs.append(ReportJob(*json.loads(line)))
                    except JSONDecodeError:
                        util.eprint(f"In {filename}:{line_num}")
                        raise
        return jobs
    except FileNotFoundError:
        with (args.output_dir / "done.json").open('w') as f:
            pass
        return []
def remove_already_done_jobs(args: argparse.Namespace) -> None:
    for workerid in range(args.num_workers):
        with (args.output_dir / f"worker-{workerid}-done.json").open('w') as f:
            pass

if __name__ == "__main__":
    main()
