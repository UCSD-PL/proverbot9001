#!/usr/bin/env python3
import argparse
import json
import sys
import multiprocessing

from pathlib import Path

from search_worker import get_files_jobs
from util import FileLock, eprint

from typing import List, cast, Tuple

def main(arg_list: List[str]) -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--prelude", default=".", type=Path)
    arg_parser.add_argument("--output", "-o", dest="output_dir",
                            help="output data folder name",
                            default="search-report",
                            type=Path)
    arg_parser.add_argument("--include-proof-relevant", action="store_true")
    arg_parser.add_argument("-j", "--num-threads", type=int, default=5)
    proofsGroup = arg_parser.add_mutually_exclusive_group()
    proofsGroup.add_argument("--proof", default=None)
    proofsGroup.add_argument("--proofs-file", default=None)
    arg_parser.add_argument("proj_files_file", type=Path)
    arg_parser.add_argument("proj_files_taken_file", type=Path)
    arg_parser.add_argument("proj_files_scanned_file", type=Path)
    arg_parser.add_argument("jobs_file", type=Path)
    args = arg_parser.parse_args()

    workers = [multiprocessing.Process(target=run_worker,
                                       args=(args,))
               for widx in range(args.num_threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

def run_worker(args: argparse.Namespace) -> None:
    with (args.output_dir / args.proj_files_file).open('r') as f:
        all_proj_files = [json.loads(line) for line in f]

    while True:
        with (args.output_dir / args.proj_files_taken_file).open('r+') as f, FileLock(f):
            taken_proj_files = [json.loads(line) for line in f]
            to_be_scanned = [tuple(proj_file) for proj_file in all_proj_files
                             if proj_file not in taken_proj_files]
            if len(to_be_scanned) > 0:
                next_proj_file = cast(Tuple[str, str], to_be_scanned[0])
                print(json.dumps(next_proj_file), file=f, flush=True)
            else:
                break
        jobs = list(set(get_files_jobs(args, [next_proj_file])))
        with (args.output_dir / args.jobs_file).open('a') as f, FileLock(f):
            for job in jobs:
                print(json.dumps(job), file=f, flush=True)
        with (args.output_dir / args.proj_files_scanned_file).open('a') as f, FileLock(f):
            print(json.dumps(next_proj_file), file=f, flush=True)
    pass

if __name__ == "__main__":
    main(sys.argv[1:])
