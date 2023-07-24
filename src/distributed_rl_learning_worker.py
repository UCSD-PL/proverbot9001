#!/usr/bin/env python3

import argparse
import sys
import os
import random
import json
from pathlib import Path

import torch

from gen_rl_tasks import RLTask
import rl
from util import nostderr, FileLock, eprint, print_time
from distributed_rl import add_distrl_args_to_parser

sys.path.append(str(Path(os.getcwd()) / "src"))

def main():
    assert 'SLURM_ARRAY_TASK_ID' in os.environ
    workerid = int(os.environ['SLURM_ARRAY_TASK_ID'])

    parser = argparse.ArgumentParser(
        description="Train a state estimator using reinforcement learning"
        "to complete proofs using Proverbot9001.")
    rl.add_args_to_parser(parser)
    add_distrl_args_to_parser(parser)
    args = parser.parse_args()

    if args.filenames[0].suffix == ".json":
        args.splits_file = args.filenames[0]
        args.filenames = None
    else:
        args.splits_file = None

    reinforce_jobs_worker(args, workerid)

def reinforce_jobs_worker(args: argparse.Namespace, workerid: int) -> None:
    worker, steps_already_done, random_state = rl.possibly_resume_rworker(args)
    random.setstate(random_state)

    assert args.tasks_file, "Can't do distributed rl without tasks right now."
    with open(args.tasks_file, 'r') as f:
        all_tasks = [RLTask(*json.loads(line)) for line in f]

    if args.curriculum:
        all_tasks = sorted(all_tasks, key=lambda t: t.target_length)

    if args.interleave:
        task_episodes = [(task, episode) for episode, tasks_list in
                         enumerate([all_tasks] * args.num_episodes)
                         for task in tasks_list]
    else:
        task_episodes = [(task, episode) for task in all_tasks
                         for episode, task in list(enumerate([task] * args.num_episodes))]

    for _ in range(steps_already_done):
        # Pytorch complains about pre-stepping the scheduler, but we
        # want it for resuming so silence that error here.
        with nostderr():
            worker.v_network.adjuster.step()

    while True:
        with (args.state_dir / "taken.txt").open("r+") as f, FileLock(f):
            taken_task_episodes = [(RLTask(**task_dict), episode)
                                   for task_dict, episode in (json.loads(line),)
                                   for line in f]
            current_task_episode = None
            for task_episode in task_episodes:
                if task_episode not in taken_task_episodes:
                    current_task_episode = task_episode
                    break
            if current_task_episode is not None:
                eprint(f"Starting task-episode {current_task_episode}")
                print(json.dumps(current_task_episode), f, flush=True)
            else:
                eprint(f"Finished worker {workerid}")
                break
        with open(args.state_dir / "taken-{workerid}.txt").open('a') as f:
            print(json.dumps(current_task_episode), f, flush=True)

        reinforce_task(args, worker, current_task_episode[0],
                       len(taken_task_episodes), len(task_episodes), workerid)

        with (args.state_dir / f"done-{workerid}.txt").open('a') as f, FileLock(f):
            print(json.dumps(current_task_episode), f, flush=True)

    if steps_already_done < len(task_episodes):
        with print_time("Saving"):
            save_state(args, worker, len(taken_task_episodes), workerid)

def reinforce_task(args: argparse.Namespace, worker: rl.ReinforcementWorker,
                   task: RLTask, step: int, num_steps_total: int,
                   workerid: int):
    cur_epsilon = args.starting_epsilon + ((step / num_steps_total) *
                                           (args.ending_epsilon -
                                            args.starting_epsilon))
    worker.run_job_reinforce(task.to_job(), task.tactic_prefix, cur_epsilon)
    if (step + 1) % args.train_train_every == 0:
        with print_time("Training", guard=args.print_timings):
            worker.train()
    save_state(args, worker, step + 1, workerid)
    if (step + 1) % args.sync_target_every == 0:
        sync_distributed_networks(args, worker)

def save_state(args: argparse.Namespace, worker: rl.ReinforcementWorker,
               step: int, workerid: int) -> None:
    with (args.state_dir / f"worker-{workerid}-network.dat").open('wb') as f:
        torch.save((worker.replay_buffer, step,
                    worker.v_network.get_state(),
                    random.getstate()), f)

def sync_distributed_networks(args: argparse.Namespace,
                              worker: rl.ReinforcementWorker):
    try:
        target_network_state = torch.load(str(args.state_dir /
                                              "common-target-network.dat"))
        worker.target_network.load_state(target_network_state)
    except FileNotFoundError:
        eprint("Skipping sync because the target network doesn't exist yet")

if __name__ == "__main__":
    main()
