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
#    Copyright 2022 Alex Sanchez-Stern
#
##########################################################################

import argparse
import json
import sys
import multiprocessing
from os import environ
from typing import List

from pathlib import Path
import torch

from search_file import (add_args_to_parser, get_predictor, Worker, project_dicts_from_args)
from models.tactic_predictor import TacticPredictor
import util
from util import eprint, FileLock

def main(arg_list: List[str]) -> None:
    assert 'SLURM_ARRAY_TASK_ID' in environ
    workerid = int(environ['SLURM_ARRAY_TASK_ID'])

    multiprocessing.set_start_method('spawn')
    arg_parser = argparse.ArgumentParser()

    add_args_to_parser(arg_parser)
    arg_parser.add_argument("--num-workers", default=32, type=int)
    arg_parser.add_argument("--workers-output-dir", default=Path("output"),
                            type=Path)
    arg_parser.add_argument("--worker-timeout", default="6:00:00")
    arg_parser.add_argument("-p", "--partition", default="defq")
    arg_parser.add_argument("--mem", default="2G")
    args = arg_parser.parse_args(arg_list)
    with (args.output_dir / "workers_scheduled.txt").open('a') as f, FileLock(f):
        print(workerid, file=f)
    if args.filenames[0].suffix == ".json":
        assert args.splits_file == None
        assert len(args.filenames) == 1
        args.splits_file = args.filenames[0]
        args.filenames = []

    sys.setrecursionlimit(100000)
    if util.use_cuda:
        torch.cuda.set_device("cuda:0")
        util.cuda_device = "cuda:0"
    
    if not args.predictor and not args.weightsfile:
        print("You must specify a weightsfile or a predictor.")
        parser.print_help()
        sys.exit(1)



    workers = [multiprocessing.Process(target=run_worker,
                                       args=(args, widx, workerid))
               for widx in range(args.num_threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    eprint(f"Finished worker {workerid}")

def run_worker(args: argparse.Namespace, threadid: int, workerid: int) -> None:
    with (args.output_dir / "jobs.txt").open('r') as f:
        all_jobs = [json.loads(line) for line in f]
    
    predictor = get_predictor(args)
    
    project_dicts = project_dicts_from_args(args)
    if any(["switch" in item for item in project_dicts]):
        switch_dict = {item["project_name"]: item["switch"]
                        for item in project_dicts}
    else:
        switch_dict = None
    worker_taken_file = args.output_dir / f"worker-{workerid}-taken.txt"
    with worker_taken_file.open("w"):
        pass

    with Worker(args, threadid, predictor, switch_dict) as worker:
        while True:
            with (args.output_dir / "taken.txt").open('r+') as f, FileLock(f):
                taken_jobs = [json.loads(line) for line in f]
                current_job = None
                for job in all_jobs:
                    if job not in taken_jobs:
                        current_job = job
                        break
                if current_job:
                    print(json.dumps(current_job), file=f, flush=True)
                    with worker_taken_file.open("a") as f:
                        print(json.dumps(current_job), file=f, flush=True)
                    eprint(f"Starting job {current_job}")
                else:
                    eprint(f"Finished thread {threadid}")
                    break
            solution = worker.run_job(current_job)
            job_project, job_file, _, _ = current_job
            project_dict = [d for d in project_dicts if d["project_name"] == job_project][0]
            with (args.output_dir / job_project /
                  (util.safe_abbrev(Path(job_file), [Path(filename) for filename in
                                                     project_dict["test_files"]])
                   + "-proofs.txt")
                  ).open('a') as f, FileLock(f):
                eprint(f"Finished job {current_job}")
                print(json.dumps((current_job, solution.to_dict())), file=f)

if __name__ == "__main__":
    main(sys.argv[1:])
