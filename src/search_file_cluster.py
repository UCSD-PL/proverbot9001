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
import signal
import shutil
import functools
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, NamedTuple, Dict, Any

from tqdm import tqdm


from search_file import (add_args_to_parser,
                         get_already_done_jobs, remove_already_done_jobs,
                         project_dicts_from_args, format_arg_value)
from search_worker import ReportJob
import util

details_css = "details.css"
details_javascript = "search-details.js"
start_time = datetime.now()

def main(arg_list: List[str]) -> None:
    global start_time
    arg_parser = argparse.ArgumentParser()

    add_args_to_parser(arg_parser)
    arg_parser.add_argument("--num-workers", default=32, type=int)
    arg_parser.add_argument("--workers-output-dir", default=Path("output"),
                            type=Path)
    arg_parser.add_argument("--worker-timeout", default="6:00:00")
    arg_parser.add_argument("-p", "--partition", default="defq")
    arg_parser.add_argument("--mem", default="2G")

    args = arg_parser.parse_args(arg_list)
    if args.filenames[0].suffix == ".json":
        assert args.splits_file == None
        assert len(args.filenames) == 1
        args.splits_file = args.filenames[0]
        args.filenames = []
    base = Path(os.path.dirname(os.path.abspath(__file__)))
    assert Path(args.prelude).exists(), "Prelude directory doesn't exist!"

    os.makedirs(str(args.output_dir), exist_ok=True)
    if args.splits_file:
        with args.splits_file.open('r') as splits_f:
            project_dicts = json.loads(splits_f.read())
        for project_dict in project_dicts:
            project_output_dir = args.output_dir / project_dict["project_name"]
            if len(project_dict["test_files"]) == 0:
                continue
            os.makedirs(str(project_output_dir), exist_ok=True)
            for filename in [details_css, details_javascript]:
                destpath = args.output_dir / project_dict["project_name"] / filename
                if not destpath.exists():
                    srcpath = base.parent / 'reports' / filename
                    shutil.copyfile(srcpath, destpath)
    else:
        for filename in [details_css, details_javascript]:
            destpath = args.output_dir / filename
            if not destpath.exists():
                srcpath = base.parent / 'reports' / filename
                shutil.copyfile(srcpath, destpath)

    start_time = datetime.now()
    if args.resume:
        solved_jobs = get_already_done_jobs(args)
        try:
            with open(args.output_dir / "time_so_far.txt", 'r') as f:
                time_taken = util.read_time_taken(f.read())
                start_time = datetime.now() - time_taken
        except FileNotFoundError:
            assert len(solved_jobs) == 0, "Trying to resume but can't find a time record!"
            pass
    else:
        remove_already_done_jobs(args)
        solved_jobs = []
    os.makedirs(str(args.output_dir / args.workers_output_dir), exist_ok=True)
    get_all_jobs_cluster(args)
    with open(args.output_dir / "all_jobs.txt") as f:
        jobs = [ReportJob(*json.loads(line)) for line in f]
        assert len(jobs) > 0
    if len(solved_jobs) < len(jobs):
        if args.just_print_jobs:
            for job in jobs:
                if job not in solved_jobs:
                    print(job)
            sys.exit(0)
        setup_jobsstate(args.output_dir, jobs, solved_jobs)
        dispatch_workers(args, arg_list)
        with util.sighandler_context(signal.SIGINT, functools.partial(interrupt_early, args)):
            show_progress(args)
        cancel_workers(args)
        with open(args.output_dir / "time_so_far.txt", 'w') as f:
            time_taken = datetime.now() - start_time
            print(str(time_taken), file=f)
    else:
        assert len(solved_jobs) == len(jobs), \
          f"There are {len(solved_jobs)} solved jobs but only {len(jobs)} jobs total detected"

    if args.generate_report:
        with open(args.output_dir / "args.json", 'w') as f:
            json.dump({k: format_arg_value(v) for k, v in vars(args).items()}, f)
        cur_dir = os.path.realpath(os.path.dirname(__file__))

        for project_dict in project_dicts:
            if len(project_dict["test_files"]) == 0:
                continue
            if project_dict['project_name'] == ".":
                report_output_name = "report.out"
            else:
                report_output_name = f"{project_dict['project_name']}-report.out"
            command = [f"{cur_dir}/sbatch-retry.sh",
                            "-o", str(args.output_dir / args.workers_output_dir
                                      / report_output_name),
                            "-J", "proverbot9001-report-worker",
                            f"{cur_dir}/search_report.sh",
                            str(args.output_dir),
                            "-p", project_dict['project_name']]
            subprocess.run(command)
        with util.sighandler_context(signal.SIGINT,
                                     functools.partial(interrupt_report_early, args)):
            show_report_progress(args.output_dir, project_dicts)
        subprocess.run([f"{cur_dir}/sbatch-retry.sh",
                        "-o", str(args.output_dir / args.workers_output_dir
                                  / "index-report.out"),
                        "-J", "proverbot9001-report-worker",
                        f"{cur_dir}/search_report.sh",
                        str(args.output_dir),
                        "-i"])
        print("Generating report index...")
        while True:
            indexes_generated = int(subprocess.check_output(
                f"find {args.output_dir} -name 'index.html' | wc -l",
                shell=True, text=True))
            if indexes_generated > 0:
                break
            else:
                time.sleep(0.2)


def get_all_jobs_cluster(args: argparse.Namespace) -> None:
    if (args.output_dir / "all_jobs.txt").exists():
        return
    project_dicts = project_dicts_from_args(args)
    projfiles = [(project_dict["project_name"], filename)
                 for project_dict in project_dicts
                 for filename in project_dict["test_files"]]
    with (args.output_dir / "all_jobs.txt.partial").open("w") as f:
        pass
    with (args.output_dir / "proj_files.txt").open("w") as f:
        for projfile in projfiles:
            print(json.dumps(projfile), file=f)
    with (args.output_dir / "proj_files_taken.txt").open("w") as f:
        pass
    with (args.output_dir / "proj_files_scanned.txt").open("w") as f:
        pass
    worker_args = [f"--prelude={args.prelude}",
                   f"--output={args.output_dir}",
                   "proj_files.txt",
                   "proj_files_taken.txt",
                   "proj_files_scanned.txt",
                   "all_jobs.txt.partial"]
    if args.include_proof_relevant:
        worker_args.append("--include-proof-relevant")
    if args.proof:
        worker_args.append(f"--proof={args.proof}")
    elif args.proofs_file:
        worker_args.append(f"--proofs-file={str(args.proofs_file)}")
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    subprocess.run([f"{cur_dir}/sbatch-retry.sh",
                    "-o", str(args.output_dir / args.workers_output_dir /
                              "file-scanner-%a.out"),
                    f"--array=0-{args.num_workers-1}",
                    f"{cur_dir}/job_getting_worker.sh"] + worker_args)

    with tqdm(desc="Getting jobs", total=len(projfiles), dynamic_ncols=True) as bar:
        num_files_scanned = 0
        while num_files_scanned < len(projfiles):
            with (args.output_dir / "proj_files_scanned.txt").open('r') as f:
                new_files_scanned = len([line for line in f])
            bar.update(new_files_scanned - num_files_scanned)
            num_files_scanned = new_files_scanned
            time.sleep(0.2)

    os.rename(args.output_dir / "all_jobs.txt.partial",
              args.output_dir / "all_jobs.txt")


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
                        "-J", "proverbot9001-worker",
                        "-p", args.partition,
                        "-t", str(args.worker_timeout),
                        "--cpus-per-task", str(args.num_threads),
                        "-o", str(args.output_dir / args.workers_output_dir
                                  / "worker-%a.out"),
                        "--mem", args.mem,
                        f"--array=0-{num_dispatching_workers-1}",
                        f"{cur_dir}/search_file_cluster_worker.sh"] + rest_args)
        num_workers_left -= num_dispatching_workers

def interrupt_early(args: argparse.Namespace, *rest_args) -> None:
    cancel_workers(args)
    with open(args.output_dir / "time_so_far.txt", 'w') as f:
        time_taken = datetime.now() - start_time
        print(str(time_taken), file=f)
    sys.exit()
def cancel_workers(args: argparse.Namespace) -> None:
    subprocess.run(["scancel -u $USER -n proverbot9001-worker"], shell=True)
def interrupt_report_early(args: argparse.Namespace, *rest_args) -> None:
    subprocess.run(["scancel -u $USER -n proverbot9001-report-worker"], shell=True)
    sys.exit()

def show_progress(args: argparse.Namespace) -> None:
    num_jobs_done = len(get_already_done_jobs(args))
    with (args.output_dir / "all_jobs.txt").open('r') as f:
        num_jobs_total = len([line for line in f])
    with (args.output_dir / "num_workers_dispatched.txt").open('r') as f:
        num_workers_total = int(f.read())
    with (args.output_dir / "workers_scheduled.txt").open('r') as f:
        num_workers_scheduled = len([line for line in f])
    num_workers_alive = int(subprocess.check_output(
        f"squeue -u $USER -h -n proverbot9001-worker | wc -l", text=True, shell=True))

    with tqdm(desc="Jobs finished", total=num_jobs_total,
              initial=num_jobs_done, dynamic_ncols=True) as bar, \
         tqdm(desc="Workers scheduled", total=num_workers_total,
              initial=num_workers_scheduled, dynamic_ncols=True) as wbar:
        while num_jobs_done < num_jobs_total:
            new_jobs_done = len(get_already_done_jobs(args))
            bar.update(new_jobs_done - num_jobs_done)
            num_jobs_done = new_jobs_done

            with (args.output_dir / "workers_scheduled.txt").open('r') as f:
                new_workers_scheduled = len([line for line in f])
            new_workers_alive = int(subprocess.check_output(
                f"squeue -u $USER -h -n proverbot9001-worker | wc -l",
                text=True, shell=True))
            if new_workers_alive < num_workers_alive:
                num_workers_alive = new_workers_alive
                if num_workers_alive < (num_jobs_total - num_jobs_done):
                    util.eprint("One of the workers crashed!")
            elif new_workers_alive > num_workers_alive:
                num_workers_alive = new_workers_alive
            if num_workers_alive == 0 and num_jobs_done < num_jobs_total:
                util.eprint("All workers exited, but jobs aren't done!")
                sys.exit(1)
            wbar.update(new_workers_scheduled - num_workers_scheduled)
            num_workers_scheduled = new_workers_scheduled

            time.sleep(0.2)

def show_report_progress(report_dir: Path, project_dicts: List[Dict[str, Any]]) -> None:
    test_projects_total = len([d for d in project_dicts if len(d["test_files"]) > 0])
    num_projects_done = 0
    with tqdm(desc="Project reports generated", total=test_projects_total) as bar:
        while num_projects_done < test_projects_total:
            new_projects_done = int(subprocess.check_output(
                f"find {report_dir} -wholename '*/index.html' | wc -l",
                shell=True, text=True))
            bar.update(new_projects_done - num_projects_done)
            num_projects_done = new_projects_done
            time.sleep(0.2)

if __name__ == "__main__":
    main(sys.argv[1:])
