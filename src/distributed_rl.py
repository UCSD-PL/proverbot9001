#!/usr/bin/env python3

import argparse
import signal
import subprocess
import functools
import sys
import os
import json
import time
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from glob import glob

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
    num_todo_tasks = len(get_all_task_eps(args)) - len(get_task_eps_done(args))
    if num_todo_tasks == 0:
        eprint("All jobs are done! Exiting")
        return
    assert num_todo_tasks > 0, \
        (num_todo_tasks, get_all_task_eps(args), 
         get_task_eps_done(args))
    args.num_workers = min(args.num_workers, num_todo_tasks)
    dispatch_learning_workers(args, sys.argv[1:])
    dispatch_syncing_worker(args)
    with util.sighandler_context(signal.SIGINT,
                                 functools.partial(interrupt_early, args)):
        show_progress(args)

def add_distrl_args_to_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-workers", default=32, type=int)
    parser.add_argument("--workers-output-dir", default=Path("output"),
                        type=Path)
    parser.add_argument("--worker-timeout", default="6:00:00")
    parser.add_argument("--partition", default="cpu")
    parser.add_argument("--mem", default="2G")
    parser.add_argument("--state_dir", default="drl_state", type=Path)

def setup_jobstate(args: argparse.Namespace) -> None:
    resume_exists = len(glob(str(args.state_dir / "weights" / "worker-*-network-*.dat"))) > 0
    if args.resume == "ask" and resume_exists:
        print(f"Found existing worker weights in state dir {args.state_dir}. Resume?")
        response = input("[Y/n] ")
        if response.lower() in ["no", "n"]:
            resume = "no"
        else:
            resume = "yes"
    elif not resume_exists:
        assert args.resume != "yes", \
             "Can't resume because no worker weights exist " \
             "in state dir {arg.state_dir}"
        resume = "no"
    else:
        resume = args.resume

    if resume == "no" and resume_exists:
       shutil.rmtree(str(args.state_dir))
    args.state_dir.mkdir(exist_ok=True)
    (args.state_dir / args.workers_output_dir).mkdir(exist_ok=True)
    (args.state_dir / "weights").mkdir(exist_ok=True)
    taken_path = args.state_dir / "taken.txt"
    if not taken_path.exists():
        with taken_path.open('w'):
            pass

    done_task_eps = []
    for workerid in range(args.num_workers):
        worker_done_task_eps = []
        done_path = args.state_dir / f"done-{workerid}.txt"
        if not done_path.exists():
            with done_path.open("w"):
                pass
        else:
            with done_path.open('r') as f:
                worker_done_task_eps = [(RLTask(**task_dict), episode)
                                        for line in f
                                        for task_dict, episode in (json.loads(line),)]
            done_task_eps += worker_done_task_eps
        taken_path = args.state_dir / f"taken-{workerid}.txt"
        with taken_path.open("w") as f:
            pass
        progress_path = args.state_dir / f"progress-{workerid}.txt"
        with progress_path.open("w") as f:
            for task, ep in worker_done_task_eps:
                print(json.dumps((task.as_dict(), ep)), file=f, flush=True)
    with (args.state_dir / "taken.txt").open("w") as f:
        for task, ep in done_task_eps:
            print(json.dumps(((task.as_dict(), ep), False)), file=f, flush=True)

    with (args.state_dir / "workers_scheduled.txt").open('w') as f:
        pass


def dispatch_learning_workers(args: argparse.Namespace,
                              rest_args: List[str]) -> None:
    with (args.state_dir / "workers_scheduled.txt").open('w'):
        pass
    assert args.num_workers > 0, args.num_workers

    cur_dir = os.path.realpath(os.path.dirname(__file__))
    subprocess.run([f"{cur_dir}/sbatch-retry.sh",
                    "-J", f"drl-worker-{args.output_file}",
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
                    "-J", f"drl-sync-worker-{args.output_file}",
                    "-p", args.partition,
                    "-t", str(args.worker_timeout),
                    "-o", str(args.state_dir / args.workers_output_dir / "worker-sync.out"),
                    "--mem", args.mem,
                    f"{cur_dir}/distributed_rl_syncing_worker.py"] + sys.argv[1:],
                   check=False)

def get_all_task_eps(args: argparse.Namespace) -> List[Tuple[RLTask, int]]:
    with open(args.tasks_file, 'r') as f:
        all_tasks = [RLTask(*json.loads(line)) for line in f]
    return [(task, episode) for episode, tasks_lits in
            enumerate([all_tasks] * args.num_episodes)
            for task in all_tasks]


def show_progress(args: argparse.Namespace) -> None:
    all_task_eps = get_all_task_eps(args)
    task_eps_done = get_task_eps_progress(args)
    scheduled_workers: List[int] = []
    crashed_workers: List[int] = []
    with tqdm(desc="Task-episodes finished", total=len(all_task_eps),
              initial=len(task_eps_done), dynamic_ncols=True) as task_eps_bar, \
         tqdm(desc="Learning workers scheduled", total=args.num_workers,
              dynamic_ncols=True) as wbar:
        while len(task_eps_done) < len(all_task_eps):
            # Get the workers that are alive
            squeue_output = subprocess.check_output(
                f"squeue -r -u$USER -h -n drl-worker-{args.output_file} -o%K",
                shell=True, text=True)
            new_workers_alive = [int(wid) for wid in
                                 squeue_output.strip().split("\n")
                                 if wid != ""]
            # Wait a bit between doing this and checking the tasks
            # done. This is because a worker might finish and stop
            # showing up in squeue before it's writes to the "done"
            # file are fully propagated.
            time.sleep(0.2)
            # Update the bar with the tasks that have been finished
            new_task_eps_done = get_task_eps_progress(args)
            task_eps_bar.update(len(new_task_eps_done) - len(task_eps_done))
            task_eps_done = new_task_eps_done
            # Update the workers scheduled bar with workers that have been newly scheduled
            with (args.state_dir / "workers_scheduled.txt").open('r') as f:
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
                with (args.state_dir / f"taken-{worker_id}.txt").open('r') as f:
                    taken_by_worker = [(RLTask(**task_dict), episode)
                                       for l in f
                                       for task_dict, episode in (json.loads(l),)]
                with (args.state_dir / f"done-{worker_id}.txt").open('r') as f:
                    done_by_worker = [(RLTask(**task_dict), episode)
                                      for l in f
                                      for task_dict, episode in (json.loads(l),)]
                task_eps_left_behind = 0
                for task_ep in taken_by_worker:
                    if task_ep not in done_by_worker:
                        task_eps_left_behind += 1
                if task_eps_left_behind > 0:
                    util.eprint(f"Worker {worker_id} crashed! "
                                f"Left behind {task_eps_left_behind} task episodes")
                    crashed_workers.append(worker_id)

            # If all workers are dead, and we're not done with the
            # jobs, print a message and exit (we'll just exit if the
            # jobs are all done at the while condition)
            if len(new_workers_alive) == 0 and len(task_eps_done) < len(all_task_eps):
                util.eprint("All workers exited, but jobs aren't done!")
                cancel_workers(args)
                sys.exit(1)
            # If the syncing worker dies, print a message and exit
            squeue_output = subprocess.check_output(
                f"squeue -r -u$USER -h -n drl-sync-worker-{args.output_file}",
                shell=True, text=True)
            if squeue_output.strip() == "":
                util.eprint("Syncing worker died! Check the logs for more details. "
                            "Exiting...")
                cancel_workers(args)
                sys.exit(1)

            # Get the new scheduled workers, and update the worker
            # bar.
            with (args.state_dir / "workers_scheduled.txt").open('r') as f:
                new_scheduled_workers = list(f)
            wbar.update(len(new_scheduled_workers) - len(scheduled_workers))
            scheduled_workers = new_scheduled_workers

def get_task_eps_done(args: argparse.Namespace) -> List[Tuple[RLTask, int]]:
    task_eps_done: List[Tuple[RLTask, int]] = []

    for workerid in range(args.num_workers):
        with (args.state_dir / f"done-{workerid}.txt").open('r') as f:
            worker_task_eps_done = [(RLTask(**task_dict), episode)
                                    for line in f
                                    for task_dict, episode in (json.loads(line),)]
            task_eps_done += worker_task_eps_done
    return task_eps_done

def get_task_eps_progress(args: argparse.Namespace) -> List[Tuple[RLTask, int]]:
    task_eps_done: List[Tuple[RLTask, int]] = []

    for workerid in range(args.num_workers):
        with (args.state_dir / f"progress-{workerid}.txt").open('r') as f:
            worker_task_eps_done = [(RLTask(**task_dict), episode)
                                    for line in f
                                    for task_dict, episode in (json.loads(line),)]
            task_eps_done += worker_task_eps_done
    return task_eps_done

def interrupt_early(args: argparse.Namespace, *_rest_args) -> None:
    cancel_workers(args)
    sys.exit(0)

def latest_worker_save_num(args: argparse.Namespace,
                           workerid: int) -> Optional[int]:
    worker_networks = glob(f"worker-{workerid}-network-*.dat",
                           root_dir = str(args.state_dir / "weights"))
    if len(worker_networks) == 0:
        return None
    return max(int(util.unwrap(re.match(
             rf"worker-{workerid}-network-(\d+).dat",
             path)).group(1))
            for path in worker_networks)

def latest_worker_save(args: argparse.Namespace,
                       workerid: int) -> Optional[Path]:
    latest_save_num = latest_worker_save_num(args, workerid)
    if latest_save_num is None:
        return None
    return (args.state_dir / "weights" /
            f"worker-{workerid}-network-{latest_save_num}.dat")

def cancel_workers(args: argparse.Namespace) -> None:
    subprocess.run([f"scancel -u$USER -n drl-worker-{args.output_file}"], shell=True, check=True)
    subprocess.run([f"scancel -u$USER -n drl-sync-worker-{args.output_file}"], shell=True, check=True)

if __name__ == "__main__":
    main()