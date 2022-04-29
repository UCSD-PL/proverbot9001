#!/usr/bin/env python3
##########################################################################
#
#    This file is part of Proverbot9001.
#
#    Proverbot9001 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Proverbot9001 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Proverbot9001.  If not, see <https://www.gnu.org/licenses/>.
#
#    Copyright 2019 Alex Sanchez-Stern and Yousef Alhessi
#
##########################################################################

import argparse
import os
import time
import sys
import json
import subprocess
from tqdm import tqdm
from pathlib_revised import Path2

from typing import List

from search_file import (get_jobs_todo, add_args_to_parser, SearchResult,
                         generate_report, get_predictor)
import util

def main(arg_list: List[str]) -> None:
    arg_parser = argparse.ArgumentParser()

    add_args_to_parser(arg_parser)
    arg_parser.add_argument("--num-workers", default=32, type=int)
    arg_parser.add_argument("--workers-output-dir", default=Path2("output"),
                            type=Path2)
    arg_parser.add_argument("--worker-timeout", default="6:00:00")
    arg_parser.add_argument("-p", "--partition", default="defq")
    args = arg_parser.parse_args(arg_list)
    predictor = get_predictor(arg_parser, args)

    if not args.output_dir.exists():
        args.output_dir.makedirs()

    setup_jobsstate(args)
    dispatch_workers(args, arg_list)
    show_progress(args)
    generate_report(args, predictor)

def setup_jobsstate(args: argparse.Namespace) -> None:
    todo_jobs, file_solutions = get_jobs_todo(args, args.filenames)
    solved_jobs = [job for solutions in file_solutions
                   for job, sol in solutions]

    with (args.output_dir / "jobs.txt").open("w") as f:
        for job in todo_jobs:
            print(json.dumps(job), file=f)
        for job in solved_jobs:
            print(json.dumps(job), file=f)
    with (args.output_dir / "taken.txt").open("w") as f:
        for job in solved_jobs:
            print(json.dumps(job), file=f)
        pass
    for filename in args.filenames:
        solutions_filename = (args.output_dir /
              (util.safe_abbrev(filename, args.filenames) + "-proofs.txt"))
        if not solutions_filename.exists():
            with solutions_filename.open('w') as f:
                pass

def dispatch_workers(args: argparse.Namespace, rest_args: List[str]) -> None:
    os.makedirs(str(args.output_dir / args.workers_output_dir), exist_ok=True)
    with (args.output_dir / "num_workers_dispatched.txt").open("w") as f:
        print(args.num_workers, file=f)
    with (args.output_dir / "workers_scheduled.txt").open("w") as f:
        pass
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    # If you have a different cluster management software, that still allows
    # dispatching jobs through the command line and uses a shared filesystem,
    # modify this line.
    subprocess.run([f"{cur_dir}/sbatch-retry.sh",
                    "-J", "proverbot9001-worker",
                    "-p", args.partition,
                    "-t", str(args.worker_timeout),
                    "--cpus-per-task", str(args.num_threads),
                    "-o",str(args.output_dir / args.workers_output_dir
                             / "worker-%a.out"),
                    f"--array=0-{args.num_workers-1}",
                    f"{cur_dir}/search_file_cluster_worker.sh"] + rest_args)

def get_jobs_done(args: argparse.Namespace) -> int:
    jobs_done = 0
    for filename in args.filenames:
        with (args.output_dir /
              (util.safe_abbrev(filename, args.filenames) + "-proofs.txt")).open('r') as f:
            jobs_done += len([line for line in f])
    return jobs_done

def show_progress(args: argparse.Namespace) -> None:
    with (args.output_dir / "jobs.txt").open('r') as f:
        num_jobs_total = len([line for line in f])
    num_jobs_done = get_jobs_done(args)
    with (args.output_dir / "num_workers_dispatched.txt").open('r') as f:
        num_workers_total = int(f.read())
    with (args.output_dir / "workers_scheduled.txt").open('r') as f:
        num_workers_scheduled = len([line for line in f])

    with tqdm(desc="Jobs finished", total=num_jobs_total,
              initial=num_jobs_done, dynamic_ncols=True) as bar, \
         tqdm(desc="Workers scheduled", total=num_workers_total,
              initial=num_workers_scheduled, dynamic_ncols=True) as wbar:
        while num_jobs_done < num_jobs_total:
            new_jobs_done = get_jobs_done(args)
            bar.update(new_jobs_done - num_jobs_done)
            num_jobs_done = new_jobs_done

            with (args.output_dir / "workers_scheduled.txt").open('r') as f:
                new_workers_scheduled = len([line for line in f])
            wbar.update(new_workers_scheduled - num_workers_scheduled)
            num_workers_scheduled = new_workers_scheduled

            time.sleep(0.2)

if __name__ == "__main__":
    main(sys.argv[1:])
