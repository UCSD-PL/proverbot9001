#!/usr/bin/env python
import argparse
import json
import os
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from util import FileLock
from tqdm import tqdm

from coq_serapy import Obligation
from coq_serapy.coq_backend import CoqTimeoutError
from gen_rl_tasks import RLTask
from rl import FileReinforcementWorker, tactic_prefix_is_usable
from util import eprint, safe_abbrev
from collections import Counter


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-v", "--verbose", action='count', default=0)
    argparser.add_argument("--print-timings", action='store_true')
    argparser.add_argument("--prelude", default=".", type=Path)
    argparser.add_argument("--resume", action='store_true', dest="resume")
    argparser.add_argument("--progress", action='store_true')
    argparser.add_argument("--state-dir", default="drl_taskinitobl_state", type=Path)
    argparser.add_argument("tasks_file", type=Path)
    argparser.add_argument("output_file", type=Path)
    args = argparser.parse_args()
    args.backend = "serapi"
    args.set_switch = True


    # if args.output_file.exists():
    #   if args.resume:
    #     with args.output_file.open('r') as f:
    #       done_tasks = [RLTask(**task_dict) for l in f
    #                     for task_dict, obl_dict in (json.loads(l),)]
    #     start_bar_from = len(done_tasks)
    #     tasks = sorted(list(set(tasks).difference(set(done_tasks))), key=lambda task: task.src_file)
    #     eprint(f"Resumed from file {str(args.output_file)}")
    #   else:
    #     with args.output_file.open("w") as f:
    #       pass

    workerid = int(os.environ['SLURM_ARRAY_TASK_ID'])
    eprint(f"Starting worker {workerid}")
    task_state_worker(args,workerid)


def task_state_worker(args, workerid) :

    with args.tasks_file.open("r") as f:
        tasks = [RLTask(**json.loads(line)) for line in f]

    file_all_ts_dict: Dict[Path, List[Tuple[int, RLTask]]] = {}
    for task_idx, task in enumerate(tasks):
        if Path(task.src_file) in file_all_ts_dict:
            file_all_ts_dict[Path(task.src_file)].append((task_idx, task))
        else:
            file_all_ts_dict[Path(task.src_file)] = [(task_idx,task )]

    worker = FileReinforcementWorker(args, None)
    worker.enter_instance()
    recently_done_tasks: List[RLTask] = []
    file_our_taken_dict: Dict[Path, Set[int]] = {}
    our_files_taken: Set[Path] = set()
    files_finished_this_ep: Set[Path] = set()
    skip_taken_proofs: bool = True
    while True :
        next_task_and_idx = allocate_next_task(
            args,
            file_all_ts_dict,
            our_files_taken, files_finished_this_ep,
            file_our_taken_dict, skip_taken_proofs)
        print("Allocating task")
        print(next_task_and_idx)
        if next_task_and_idx is None:
            files_finished_this_ep = set()
            if skip_taken_proofs:
                skip_taken_proofs = False
                continue
            eprint(f"Finished worker {workerid}")
            break

        next_task_idx, task = next_task_and_idx
        with (args.state_dir / "taken" / f"taken-{workerid}.txt").open('a') as f:
            print(json.dumps(task.as_dict()), file=f, flush=True)
        src_path = Path(task.src_file)
        if src_path in file_our_taken_dict:
            file_our_taken_dict[src_path].add(next_task_idx)
        else:
            file_our_taken_dict[src_path] = {next_task_idx}

        next_task_idx, task = next_task_and_idx
        with (args.state_dir / "taken" / f"taken-{workerid}.txt").open('a') as f:
            print(json.dumps(task.as_dict()), file=f, flush=True)
        src_path = Path(task.src_file)
        if src_path in file_our_taken_dict:
            file_our_taken_dict[src_path].add(next_task_idx)
        else:
            file_our_taken_dict[src_path] = {next_task_idx}

        if not tactic_prefix_is_usable(task.tactic_prefix):
            if args.verbose >= 2:
                eprint(f"Skipping job {task} with prefix {task.tactic_prefix} "
                                "because it can't purely focused")
            else:
                eprint("Skipping a job because it can't be purely focused")
        else:
            obldict = get_task_state(worker, task)
            if obldict:
                with (args.state_dir / f"finished-{workerid}.txt").open('a') as f,FileLock(f):
                    print(json.dumps((task.as_dict(),obldict)), file=f, flush=True)


        recently_done_tasks.append(task)
        with (args.state_dir / f"progress-{workerid}.txt").open('a') as f, FileLock(f):
            print(json.dumps(task.as_dict()), file=f, flush=True)

def get_task_state(worker, task) :
    try :
        if tactic_prefix_is_usable(task.tactic_prefix):
            worker.run_into_task(task.to_job(), task.tactic_prefix)
            return worker.coq.proof_context.fg_goals[0].to_dict()
        else:
            eprint("Skipping a task with unusable prefix")
            return None
    except CoqTimeoutError:
        eprint("Coqtimeout, Restarting Coq")
        worker.restart_coq()
        worker.reset_file_state()
        worker.enter_file(task.src_file)
        worker.run_into_task(task.to_job(), task.tactic_prefix)
        return worker.coq.proof_context.fg_goals[0].to_dict()

def allocate_next_task(args: argparse.Namespace,
                       file_all_tasks_dict: Dict[Path, List[Tuple[int, RLTask]]], # changed from file_all_te _dict
                       our_files_taken: Set[Path],
                       files_finished_this_ep: Set[Path],
                       file_our_taken_dict: Dict[Path, Set[int]],
                       skip_taken_proofs: bool) \
          -> Optional[Tuple[int, RLTask]]:
    all_files = list(file_all_tasks_dict.keys())
    while True:
        with (args.state_dir / "taken" / "taken-files.txt"
              ).open("r+") as f, FileLock(f):
            taken_files: Counter[Path] = Counter(Path(p.strip()) for p in f)
            if len(all_files) <= len(files_finished_this_ep):
     # This happens once we've hit the episode cap and checked every
     # file.
                return None
            least_taken_count: int = min(taken_files[filename] for filename in all_files
                                          if filename not in files_finished_this_ep)
            cur_file: Optional[Path] = None
            for src_file in all_files:
                if src_file in files_finished_this_ep:
                    continue
                if (src_file in our_files_taken or
                    taken_files[src_file] == 0 or
                    (taken_files[src_file] == least_taken_count)):
                    cur_file = src_file
                    break
            if cur_file is not None and cur_file not in our_files_taken:
                print(cur_file, file=f, flush=True)
            if cur_file is None:
                break
            next_task_and_idx = allocate_next_task_from_file_with_retry(
                args,
                cur_file, all_files,
                file_all_tasks_dict[cur_file],
                file_our_taken_dict.get(cur_file, set()),
                skip_taken_proofs)
            if next_task_and_idx is not None:
                our_files_taken.add(cur_file)
                return next_task_and_idx
            else:
                eprint(f"Couldn't find an available task for file {cur_file}, "
                       f"trying next file...",
                       guard=args.verbose >= 2)
                files_finished_this_ep.add(cur_file)
    return None

def allocate_next_task_from_file_with_retry(
            args: argparse.Namespace,
            filename: Path,
            all_files: List[Path],
            file_task_episodes: List[Tuple[int, RLTask]],
            our_taken_task_episodes: Set[int],
            skip_taken_proofs: bool,
            num_retries: int = 3) \
            -> Optional[Tuple[int, RLTask]]:
        # This is necessary because sometimes the NFS file system will give us
        # garbage back like "^@^@^@^@^@^" when we're reading files that are
        # being written.  By retrying we get the actual file contents when
        # there's a problem like that.
    for i in range(num_retries):
        try:
            return allocate_next_task_from_file(
                args, filename, all_files,
                file_task_episodes, our_taken_task_episodes,
                skip_taken_proofs)
        except json.decoder.JSONDecodeError:
            if i == num_retries - 1:
                raise
            eprint("Failed to decode, retrying", guard=args.verbose >= 1)
    return None


def allocate_next_task_from_file(args: argparse.Namespace,
                                                                 filename: Path,
                                                                 all_files: List[Path],
                                                                 file_tasks: List[Tuple[int, RLTask]],
                                                                 our_taken_tasks: Set[int],
                                                                 skip_taken_proofs: bool) \
                                                                -> Optional[Tuple[int, RLTask]]:
    filepath = args.state_dir / "taken" / ("file-" + safe_abbrev(filename, all_files) + ".txt")
    with filepath.open("r+") as f, FileLock(f):
        # Loading the taken file
        taken_tasks: Set[int] = set()
        taken_by_others_this_iter: Set[int] = set()
        for line_num, line in enumerate(f):
            try:
                task_idx, taken_this_iter = json.loads(line)
                taken_tasks.add(task_idx)
                if task_idx not in our_taken_tasks and taken_this_iter:
                    taken_by_others_this_iter.add(task_idx)
            except json.decoder.JSONDecodeError:
                eprint(f"Failed to parse line {filepath}:{line_num}: \"{line.strip()}\"")
                raise
        # Searching the tasks for a good one to take
        proofs_taken_by_others: Set[Tuple[str, str, str] ] = set()
        for file_task_idx, (task_idx, task) in enumerate(file_tasks):
            if task_idx in taken_by_others_this_iter:
                proofs_taken_by_others.add(task.to_proof_spec())
                continue
            if (task_idx in taken_tasks
                or task.to_proof_spec() in proofs_taken_by_others
                and skip_taken_proofs):
                continue
            eprint(f"Found an appropriate task-episode after searching "
                   f"{file_task_idx} task-episodes", guard=args.verbose >= 2)
            print(json.dumps((task_idx, True)), file=f, flush=True)
            return task_idx, task
    return None

if __name__ == "__main__":
    main()
