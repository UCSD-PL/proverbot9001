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

import fcntl
import time
import util
import traceback
import argparse
import json
import sys
import multiprocessing
import re
from os import environ
from typing import List, Optional

from pathlib_revised import Path2
import torch

from search_file import (add_args_to_parser, get_predictor, Worker)
import coq_serapy
from coq_serapy.contexts import ProofContext
from models.tactic_predictor import TacticPredictor
from util import eprint, unwrap

def main(arg_list: List[str]) -> None:
    multiprocessing.set_start_method('spawn')
    arg_parser = argparse.ArgumentParser()

    add_args_to_parser(arg_parser)
    arg_parser.add_argument("--num-workers", default=32, type=int)
    arg_parser.add_argument("--workers-output-dir", default=Path2("output"),
                            type=Path2)
    arg_parser.add_argument("--worker-timeout", default="6:00:00")
    arg_parser.add_argument("-p", "--partition", default="defq")
    args = arg_parser.parse_args(arg_list)
    
    if 'SLURM_ARRAY_TASK_ID' in environ:
        workerid = int(environ['SLURM_ARRAY_TASK_ID'])
    else:
        assert False, 'SLURM_ARRAY_TASK_ID must be set'

    sys.setrecursionlimit(100000)
    if util.use_cuda:
        torch.cuda.set_device("cuda:0")
        util.cuda_device = "cuda:0"

    predictor = get_predictor(arg_parser, args)
    with (args.output_dir / "workers_scheduled.txt").open('a') as f, FileLock(f):
        print(workerid, file=f)
    workers = [multiprocessing.Process(target=run_worker,
                                       args=(args, widx,
                                             predictor))
               for widx in range(args.num_threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    eprint(f"Finished worker {workerid}")

def run_worker(args: argparse.Namespace, workerid: int,
               predictor: TacticPredictor) -> None:
    with (args.output_dir / "jobs.txt").open('r') as f:
        all_jobs = [json.loads(line) for line in f]

    with Worker(args, predictor) as worker:
        while True:
            with (args.output_dir / "taken.txt").open('r+') as f, FileLock(f):
                taken_jobs = [json.loads(line) for line in f]
                remaining_jobs = [job for job in all_jobs if job not in taken_jobs]
                if len(remaining_jobs) > 0:
                    current_job = remaining_jobs[0]
                    print(json.dumps(current_job), file=f, flush=True)
                    eprint(f"Starting job {current_job}")
                else:
                    break
            solution = worker.run_job(current_job)
            job_project, job_file, _, _ = current_job
            with (args.output_dir / job_project /
                  (util.safe_abbrev(Path2(job_file), args.filenames) + "-proofs.txt")
                  ).open('a') as f, FileLock(f):
                eprint(f"Finished job {current_job}")
                print(json.dumps((current_job, solution.to_dict())), file=f)
    
class FileLock:
    def __init__(self, file_handle):
        self.file_handle = file_handle

    def __enter__(self):
        while True:
            try:
                fcntl.flock(self.file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
               time.sleep(0.01)
        return self

    def __exit__(self, type, value, traceback):
        fcntl.flock(self.file_handle, fcntl.LOCK_UN)

if __name__ == "__main__":
    main(sys.argv[1:])
