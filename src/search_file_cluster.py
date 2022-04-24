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

from search_file import get_jobs_todo, SearchResult, generate_report
import util

def main(arg_list: List[str]) -> None:
    cluster_arg_parser = argparse.ArgumentParser()
    cluster_arg_parser.add_argument("--num-workers", default=32, type=int)
    cluster_arg_parser.add_argument("--workers-output-dir", default=Path2("output"),
                                    type=Path2)
    cluster_arg_parser.add_argument("--worker-timeout", default="6:00:00")
    cluster_arg_parser.add_argument("-p", "--partition", default="defq")
    cluster_arg_parser.add_argument("--output", "-o", dest="output_dir",
                                    help="output data folder name",
                                    default="search-report",
                                    type=Path2)
    cluster_arg_parser.add_argument('filenames', help="proof file name (*.v)",
                                    nargs='+', type=Path2)
    cluster_args, rest_args = cluster_arg_parser.parse_known_args(rest_args)
    rest_args.append("--output=" + args.output)
    rest_args += args.filenames
    file_args, _, _ = parse_arguments(rest_args)
    all_args = argparse.Namespace(**vars(cluster_args), **vars(file_args))

    setup_jobsstate(cluster_args)
    dispatch_workers(cluster_args, rest_args)
    show_progress(cluster_args)
    generate_report(all_args, "features polyarg")

def setup_jobsstate(args: argparse.Namespace) -> None:
    todo_jobs, file_solutions = get_jobs_todo(args, args.filenames)
    solved_jobs = [job for filename, solutions in file_solutions
                   for job, sol in solutions]

    with (args.output / "jobs.txt").open("w") as f:
        for job in todo_jobs:
            print(json.dumps(job), file=f)
        for job, solution in solved_jobs:
            print(json.dumps(job), file=f)
    with (args.output / "taken.txt").open("w") as f:
        for job, solution in solved_jobs:
            print(json.dumps(job), file=f)
        pass
    for filename in args.filenames:
        solutions_filename = args.output /
              (util.safe_abbrev(filename, args.filenames) + "-proofs.txt")
        if not solutions_filename.exists():
            with solutions_filename.open('w') as f:
                pass

def dispatch_workers(args: argparse.Namespace, rest_args: List[str]) -> None:
    os.makedirs(str(args.output / args.workers_output_dir), exist_ok=True)
    with (args.output / "num_workers_dispatched.txt").open("w") as f:
        print(args.num_workers, file=f)
    with (args.output / "workers_scheduled.txt").open("w") as f:
        pass
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    # If you have a different cluster management software, that still allows
    # dispatching jobs through the command line and uses a shared filesystem,
    # modify this line.
    subprocess.run([f"{cur_dir}/sbatch_retry.sh",
                    "-J", "proverbot9001-worker",
                    "-p", args.partition,
                    "-t", args.worker_timeout,
                    f"--output=" + str(args.output / args.workers_output_dir
                                       / "worker-%a.out")
                    f"--array=0-{args.num_workers-1}",
                    f"{cur_dir}/search_file_cluster_worker.py",
                    args.output] + rest_args)

def get_jobs_done(args: argparse.Namespace) -> int:
    jobs_done = 0
    for filename in args.filenames:
        with (args.output /
              (util.safe_abbrev(filename, args.filenames) + "-proofs.txt")).open('r') as f:
            jobs_done += len([line for line in f])
    return jobs_done

def show_progress(args: argparse.Namespace) -> None:
    with (args.output / "jobs.txt").open('r') as f:
        num_jobs_total = len([line for line in f])
    num_jobs_done = get_jobs_done(args)
    with (args.output / "num_workers_dispatched.txt").open('r') as f:
        num_workers_total = int(f.read())
    with (args.output / "workers_scheduled.txt").open('r') as f:
        num_workers_scheduled = len([line for line in f])

    with tqdm(desc="Jobs finished",
              total=num_jobs_total, initial=done_jobs, dynamic_ncols=True) as bar, \
         tqdm(desc="Workers scheduled",
              total=num_workers_total, initial=workers_scheduled, dynamic_ncols=True) as wbar:
        while done_jobs < num_jobs_total:
            new_jobs_done = get_jobs_done(args)
            bar.update(new_jobs_done - num_jobs_done)
            num_jobs_done = new_jobs_done

            with (args.output / "workers_scheduled.txt").open('r') as f):
                new_workers_scheduled = len([line for line in f])
            bar.update(new_workers_scheduled - num_workers_scheduled)
            num_workers_scheduled = new_workers_scheduled

            time.sleep(0.2)

if __name__ == "__main__":
    main(sys.argv[1:])
