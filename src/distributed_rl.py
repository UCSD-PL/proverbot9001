#!/usr/bin/env python3

import argparse
import signal
import subprocess
import functools
import sys
import os
import json
import time
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from gen_rl_tasks import RLTask
import rl
import util

def main():
    parser = argparse.ArgumentParser(
        description="Train a state estimator using reinforcement learning"
        "to complete proofs using Proverbot9001.")
    rl.add_args_to_parser(parser)
    add_distrl_args_to_parser(parser)
    args = parser.parse_args()

    setup_jobstate(args)
    dispatch_learning_workers(args, sys.argv[1:])
    dispatch_syncing_worker(args)
    with util.sighandler_context(signal.SIGINT,
                                 functools.partial(interrupt_early, args)):
        show_progress(args)
    cancel_workers(args)

def add_distrl_args_to_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-workers", default=32, type=int)
    parser.add_argument("--workers-output-dir", default=Path("output"),
                        type=Path)
    parser.add_argument("--worker-timeout", default="6:00:00")
    parser.add_argument("--partition", default="cpu")
    parser.add_argument("--mem", default="2G")
    parser.add_argument("--state_dir", default="drl_state", type=Path)

def setup_jobstate(args: argparse.Namespace) -> None:
    if not args.state_dir.exists():
        args.state_dir.mkdir()
    taken_path = args.state_dir / "taken.txt"
    if not taken_path.exists():
        with taken_path.open('w'):
            pass

    for workerid in range(args.num_workers):
        done_path = args.state_dir / f"done-{workerid}.txt"
        if not done_path.exists():
            with done_path.open("w"):
                pass
        else:
            with done_path.open('r') as f:
                done_task_eps = [(RLTask(**task_dict), episode)
                                 for task_dict, episode in (json.loads(line),)
                                 for line in f]
        taken_path = args.state_dir / f"taken-{workerid}.txt"
        with taken_path.open("w") as f:
            for task_ep in done_task_eps:
                print(json.dumps(task_ep), f, flush=True)

    with (args.output_dir / "workers_scheduled.txt").open('w') as f:
        pass


def dispatch_learning_workers(args: argparse.Namespace,
                              rest_args: List[str]) -> None:
    with (args.state_dir / "workers_scheduled.txt").open('w'):
        pass

    cur_dir = os.path.realpath(os.path.dirname(__file__))
    subprocess.run([f"{cur_dir}/sbatch-retry.sh",
                    "-J", f"drl-worker-{args.output}",
                    "-p", args.partition,
                    "-t", str(args.worker_timeout),
                    "-o", str(args.state_dir / args.workers_output_dir
                              / "worker-%a.out"),
                    "--mem", args.mem,
                    f"--array=0-{args.num_workers-1}",
                    f"{cur_dir}/distributed_rl_learning_worker.py"] + rest_args,
                   check=False)

def dispatch_syncing_worker(args: argparse.Namespace) -> None:
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    subprocess.run([f"{cur_dir}/sbatch-retry.sh",
                    "-J", f"drl-worker-{args.output}",
                    "-p", args.partition,
                    "-t", str(args.worker_timeout),
                    "-o", str(args.state_dir / args.workers_output_dir / "worker-sync.out"),
                    "--mem", args.mem,
                    f"{cur_dir}/distributed_rl_syncing_worker.py"] + sys.argv[1:],
                   check=False)

def show_progress(args: argparse.Namespace) -> None:
    with open(args.tasks_file, 'r') as f:
        all_tasks = [RLTask(*json.loads(line)) for line in f]
        all_task_eps = [(task, episode) for episode, tasks_lits in
                        enumerate([all_tasks] * args.num_episodes)
                        for task in all_tasks]
    task_eps_done = get_task_eps_done(args)
    scheduled_workers: List[int] = []
    crashed_workers: List[int] = []
    with tqdm(desc="Task-episodes finished", total=len(all_task_eps),
              initial=len(task_eps_done), dynamic_ncols=True) as task_eps_bar, \
         tqdm(desc="Learning workers scheduled", total=args.num_workers,
              dynamic_ncols=True) as wbar:
        while len(task_eps_done) < len(all_task_eps):
            # Get the workers that are alive
            new_workers_alive = [int(wid) for wid in
                                 subprocess.check_output(
                                   "squeue -r -u$USER -h -n proverbot9001-task-worker -o%K",
                                   shell=True, text=True).strip().split("\n")
                                 if wid != ""]
            # Wait a bit between doing this and checking the tasks
            # done. This is because a worker might finish and stop
            # showing up in squeue before it's writes to the "done"
            # file are fully propagated.
            time.sleep(0.2)
            # Update the bar with the tasks that have been finished
            new_task_eps_done = get_task_eps_done(args)
            task_eps_bar.update(len(new_task_eps_done) - len(task_eps_done))
            task_eps_done = new_task_eps_done
            # Update the workers scheduled bar with workers that have been newly scheduled
            with (args.output_dir / "workers_scheduled.txt").open('r') as f:
                new_scheduled_workers = list(f)

            # Check for newly crashed workers
            for worker_id in range(args.num_workers):
                # Skip any living worker
                if worker_id in new_workers_alive:
                    continue
                # Skip any worker for which we've already reported a crash
                if worker_id in crashed_workers:
                    continue
                # Get the jobs taken by workers.
                with (args.output_dir / f"taken-{worker_id}.txt").open('r') as f:
                    taken_by_worker = [(RLTask(**task_dict), episode)
                                       for task_dict, episode in (json.loads(l),)
                                       for l in f]
                task_eps_left_behind = len(set(taken_by_worker) - set(task_eps_done))
                if task_eps_left_behind > 0:
                    util.eprint(f"Worker {worker_id} crashed! "
                                f"Left behind {task_eps_left_behind}")
                    crashed_workers.append(worker_id)

            if len(new_workers_alive) == 0:
                if len(task_eps_done) < len(all_task_eps):
                    util.eprint("All workers existed, but jobs aren't done!")
                    cancel_workers(args)
                    sys.exit(1)
            with (args.state_dir / "workers_scheduled.txt").open('r') as f:
                new_scheduled_workers = list(f)
            wbar.update(len(new_scheduled_workers) - len(scheduled_workers))
            scheduled_workers = new_scheduled_workers
    cancel_workers(args)

def get_task_eps_done(args: argparse.Namespace) -> List[Tuple[RLTask, int]]:
    task_eps_done: Tuple[RLTask, int] = []

    for workerid in range(args.num_workers):
        with (args.state_dir / f"done-{workerid}.txt").open('r') as f:
            worker_task_eps_done = [(RLTask(*task_dict), episode)
                              for task_dict, episode in (json.loads(line),)
                              for line in f]
            task_eps_done += worker_task_eps_done
    return task_eps_done

def interrupt_early(args: argparse.Namespace, *_rest_args) -> None:
    cancel_workers(args)
    sys.exit(0)

def cancel_workers(args: argparse.Namespace) -> None:
    subprocess.run([f"scancel -u$USER -n drl-worker-{args.output}"], shell=True, check=True)

if __name__ == "__main__":
    main()
