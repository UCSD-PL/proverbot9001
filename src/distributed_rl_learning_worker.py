#!/usr/bin/env python3

import argparse
import sys
import os
import random
import json
import re
from typing import List, Tuple, Any
from pathlib import Path
from glob import glob

import torch

sys.path.append(str(Path(os.getcwd()) / "src"))

#pylint: disable=wrong-import-position
from gen_rl_tasks import RLTask
import rl
from util import nostderr, FileLock, eprint, print_time, unwrap
from distributed_rl import (add_distrl_args_to_parser,
                            latest_worker_save)
#pylint: enable=wrong-import-position

def main():
    assert 'SLURM_ARRAY_TASK_ID' in os.environ
    workerid = int(os.environ['SLURM_ARRAY_TASK_ID'])

    parser = argparse.ArgumentParser(
        description="Train a state estimator using reinforcement learning"
        "to complete proofs using Proverbot9001.")
    rl.add_args_to_parser(parser)
    add_distrl_args_to_parser(parser)
    args = parser.parse_args()
    with (args.state_dir / "workers_scheduled.txt").open('a') as f, FileLock(f):
        print(workerid, file=f)

    if args.filenames[0].suffix == ".json":
        args.splits_file = args.filenames[0]
        args.filenames = None
    else:
        args.splits_file = None

    reinforce_jobs_worker(args, workerid)

TaskEpisode = Tuple[RLTask, int]

def get_all_task_episodes(args: argparse.Namespace) -> List[TaskEpisode]:
    assert args.tasks_file, "Can't do distributed rl without tasks right now."
    with open(args.tasks_file, 'r') as f:
        all_tasks = [RLTask(**json.loads(line)) for line in f]

    if args.curriculum:
        all_tasks = sorted(all_tasks, key=lambda t: t.target_length)

    if args.interleave:
        task_episodes = [(task, episode) for episode, tasks_list in
                         enumerate([all_tasks] * args.num_episodes)
                         for task in tasks_list]
    else:
        task_episodes = [(task, episode) for task in all_tasks
                         for episode, task in list(enumerate([task] * args.num_episodes))]

    return task_episodes

def reinforce_jobs_worker(args: argparse.Namespace,
                          workerid: int) -> None:
    worker, steps_already_done, random_state = \
        possibly_resume_rworker(args, workerid)
    random.setstate(random_state)

    task_episodes = get_all_task_episodes(args)

    for _ in range(steps_already_done):
        # Pytorch complains about pre-stepping the scheduler, but we
        # want it for resuming so silence that error here.
        with nostderr():
            worker.v_network.adjuster.step()

    worker_step = 1
    recently_done_task_eps: List[Tuple[RLTask, int]] = []
    while True:
        with (args.state_dir / "taken.txt").open("r+") as f, FileLock(f):
            taken_task_episodes = [(RLTask(**task_dict), episode)
                                   for line in f
                                   for task_dict, episode in (json.loads(line),)]
            current_task_episode = None
            for task_episode in task_episodes:
                if task_episode not in taken_task_episodes:
                    current_task_episode = task_episode
                    break
            if current_task_episode is not None:
                eprint(f"Starting task-episode {current_task_episode}")
                task, episode = current_task_episode
                print(json.dumps((vars(task), episode)), file=f, flush=True)
            else:
                eprint(f"Finished worker {workerid}")
                break
        with (args.state_dir / f"taken-{workerid}.txt").open('a') as f:
            print(json.dumps((vars(task), episode)), file=f, flush=True)

        cur_epsilon = args.starting_epsilon + \
            ((len(taken_task_episodes) / len(task_episodes)) *
             (args.ending_epsilon - args.starting_epsilon))

        reinforce_task(args, worker, current_task_episode[0],
                       worker_step, cur_epsilon)
        recently_done_task_eps.append((task, episode))
        if worker_step % args.sync_target_every == 0:
            sync_distributed_networks(args, worker_step, workerid, worker)
            save_replay_buffer(args, worker, workerid)

            with (args.state_dir / f"progress-{workerid}.txt").open('a') as f:
                print(json.dumps((vars(task), episode)),
                      file=f, flush=True)
            sync_done(args, workerid, recently_done_task_eps)
            recently_done_task_eps = []

        worker_step += 1

    if steps_already_done < len(task_episodes):
        save_state(args, worker, worker_step, workerid)
        save_replay_buffer(args, worker, workerid)
        sync_done(args, workerid, recently_done_task_eps)

def sync_done(args: argparse.Namespace,
              workerid: int,
              recently_done_task_eps: List[TaskEpisode]) -> None:
    with (args.state_dir / f"done-{workerid}.txt").open('a') as f:
        for task, episode in recently_done_task_eps:
            print(json.dumps((vars(task), episode)),
                  file=f, flush=True)

def reinforce_task(args: argparse.Namespace, worker: rl.ReinforcementWorker,
                   task: RLTask, step: int, cur_epsilon):
    worker.run_job_reinforce(task.to_job(), task.tactic_prefix, cur_epsilon)
    if step % args.train_every == 0:
        with print_time("Training", guard=args.print_timings):
            worker.train()

def save_replay_buffer(args: argparse.Namespace,
                       worker: rl.ReinforcementWorker,
                       workerid: int) -> None:
    torch.save(worker.replay_buffer,
               args.state_dir / f"buffer-{workerid}.dat")

def save_state(args: argparse.Namespace, worker: rl.ReinforcementWorker,
               step: int, workerid: int) -> None:
    save_num = step // args.sync_target_every
    with (args.state_dir / "weights" /
          f"worker-{workerid}-network-{save_num}.dat.tmp").open('wb') as f:
        torch.save((worker.replay_buffer, step,
                    worker.v_network.get_state(),
                    random.getstate()), f)
    save_path = str(args.state_dir / "weights" / f"worker-{workerid}-network-{save_num}.dat")
    os.rename(save_path + ".tmp", save_path)

def possibly_resume_rworker(args: argparse.Namespace,
                            workerid: int) \
        -> Tuple[rl.ReinforcementWorker, int, Any]:
    worker_save = latest_worker_save(args, workerid)
    predictor = rl.MemoizingPredictor(rl.get_predictor(args))
    if worker_save is not None:
        replay_buffer, steps_already_done, network_state, random_state = \
            torch.load(str(worker_save))
        random.setstate(random_state)
        print(f"Resuming from existing weights of {steps_already_done} steps")
        v_network = rl.VNetwork(None, args.learning_rate,
                                args.batch_step, args.lr_step)
        target_network = rl.VNetwork(None, args.learning_rate,
                                     args.batch_step, args.lr_step)
        v_network.load_state(network_state)
        target_network.load_state(network_state)
        # This ensures that the target and obligation will share a cache for coq2vec encodings
        target_network.obligation_encoder = v_network.obligation_encoder
    else:
        steps_already_done = 0
        replay_buffer = None
        random_state = random.getstate()
        with print_time("Building models"):
            v_network = rl.VNetwork(args.coq2vec_weights, args.learning_rate,
                                    args.batch_step, args.lr_step)
            target_network = rl.VNetwork(args.coq2vec_weights, args.learning_rate,
                                         args.batch_step, args.lr_step)
            # This ensures that the target and obligation will share a cache for coq2vec encodings
            target_network.obligation_encoder = v_network.obligation_encoder
    worker = rl.ReinforcementWorker(args, predictor, v_network, target_network,
                                    rl.switch_dict_from_args(args),
                                    initial_replay_buffer = replay_buffer)
    return worker, steps_already_done, random_state

def load_latest_target_network(args: argparse.Namespace,
                               worker: rl.ReinforcementWorker) -> None:
    target_networks = glob("common-target-network-*.dat",
                           root_dir = str(args.state_dir / "weights"))
    if len(target_networks) == 0:
        eprint("Skipping sync because the target network doesn't exist yet")
        return
    target_network_save_nums = [
        int(unwrap(re.match(r"common-target-network-(\d+).dat", path)).group(1))
        for path in target_networks]
    latest_target_network_path = str(
        args.state_dir / "weights" /
        f"common-target-network-{max(target_network_save_nums)}.dat")
    target_network_state = torch.load(latest_target_network_path)
    worker.target_v_network.network.load_state_dict(target_network_state)

def sync_distributed_networks(args: argparse.Namespace, step: int,
                              workerid: int,
                              worker: rl.ReinforcementWorker) -> None:
    save_state(args, worker, step, workerid)
    load_latest_target_network(args, worker)

if __name__ == "__main__":
    main()
