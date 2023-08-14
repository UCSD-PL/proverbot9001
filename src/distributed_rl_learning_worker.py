#!/usr/bin/env python3

import argparse
import sys
import os
import random
import json
import re
import time
from typing import List, Tuple, Any, Optional, Set, Dict, Iterable
from pathlib import Path
from glob import glob
from collections import Counter

import torch

sys.path.append(str(Path(os.getcwd()) / "src"))

#pylint: disable=wrong-import-position
from gen_rl_tasks import RLTask
import rl
from util import (nostderr, FileLock, eprint,
                  print_time, unwrap, safe_abbrev)
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
        print(workerid, file=f, flush=True)

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
    file_all_tes_dict: Dict[Path, List[Tuple[int, TaskEpisode]]] = {}
    for task_ep_idx, (task, episode) in enumerate(task_episodes):
        if Path(task.src_file) in file_all_tes_dict:
            file_all_tes_dict[Path(task.src_file)].append((task_ep_idx, (task, episode)))
        else:
            file_all_tes_dict[Path(task.src_file)] = [(task_ep_idx, (task, episode))]

    for _ in range(steps_already_done):
        # Pytorch complains about pre-stepping the scheduler, but we
        # want it for resuming so silence that error here.
        with nostderr():
            worker.v_network.adjuster.step()

    worker_step = 1
    recently_done_task_eps: List[TaskEpisode] = []
    file_our_taken_dict: Dict[Path, Set[int]] = {}
    our_files_taken: Set[Path] = set()
    max_episode: int = 0
    files_finished_this_ep: Set[Path] = set()
    while True:
        next_task_and_idx = allocate_next_task(args,
                                               file_all_tes_dict,
                                               our_files_taken, files_finished_this_ep,
                                               file_our_taken_dict, max_episode)
        if next_task_and_idx is None:
            if max_episode == args.num_episodes - 1:
                eprint(f"Finished worker {workerid}")
                break
            else:
                assert max_episode < args.num_episodes - 1
                max_episode += 1
            files_finished_this_ep = set()
            continue
        next_task_idx, (task, episode) = next_task_and_idx
        with (args.state_dir / "taken" / f"taken-{workerid}.txt").open('a') as f:
            print(json.dumps((task.as_dict(), episode)), file=f, flush=True)
        src_path = Path(task.src_file)
        if src_path in file_our_taken_dict:
            file_our_taken_dict[src_path].add(next_task_idx)
        else:
            file_our_taken_dict[src_path] = {next_task_idx}

        cur_epsilon = args.starting_epsilon + \
            ((get_num_tasks_taken(args, list(file_all_tes_dict.keys()))
              / len(task_episodes)) *
             (args.ending_epsilon - args.starting_epsilon))

        reinforce_task(args, worker, task,
                       worker_step, cur_epsilon)
        recently_done_task_eps.append((task, episode))
        with (args.state_dir / f"progress-{workerid}.txt").open('a') as f, FileLock(f):
            print(json.dumps((task.as_dict(), episode)),
                  file=f, flush=True)
        if worker_step % args.sync_target_every == 0:
            with print_time("Syncing", guard=args.print_timings):
                sync_distributed_networks(args, worker_step, workerid, worker)
                save_replay_buffer(args, worker, workerid)

                sync_done(args, workerid, recently_done_task_eps)
                assert len(recently_done_task_eps) == args.sync_target_every
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
            print(json.dumps((task.as_dict(), episode)),
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
        with print_time("Building models", guard=args.print_timings):
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

def allocate_next_task(args: argparse.Namespace,
                       file_all_tes_dict: Dict[Path, List[Tuple[int, TaskEpisode]]],
                       our_files_taken: Set[Path],
                       files_finished_this_ep: Set[Path],
                       file_our_taken_dict: Dict[Path, Set[int]],
                       max_episode: int) \
                      -> Optional[Tuple[int, TaskEpisode]]:
    all_files = list(file_all_tes_dict.keys())
    while True:
        with (args.state_dir / "taken" / "taken-files.txt"
              ).open("r+") as f, FileLock(f):
            taken_files: Counter[Path] = Counter(Path(p.strip()) for p in f)
            if len(all_files) <= len(files_finished_this_ep):
                assert max_episode == args.num_episodes - 1
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
                    (taken_files[src_file] == least_taken_count
                     and max_episode == args.num_episodes - 1)):
                    cur_file = src_file
                    break
            if cur_file is not None:
                print(cur_file, file=f, flush=True)
        if cur_file is None:
            break
        next_te_and_idx = allocate_next_task_from_file_with_retry(
          args,
          cur_file, all_files,
          file_all_tes_dict[cur_file],
          file_our_taken_dict.get(cur_file, set()),
          max_episode)
        if next_te_and_idx is not None:
            our_files_taken.add(cur_file)
            return next_te_and_idx
        else:
            eprint(f"Couldn't find an available task for file {cur_file}, "
                   f"trying next file...",
                   guard=args.verbose >= 2)
            files_finished_this_ep.add(cur_file)
    assert max_episode < args.num_episodes - 1
    # This is the case when we've exhausted all the tasks less than max_episode
    # in the files we've taken, but max_episode isn't high enough to justify
    # checking other files.
    return None

def allocate_next_task_from_file_with_retry(
      args: argparse.Namespace,
      filename: Path,
      all_files: List[Path],
      file_task_episodes: List[Tuple[int, TaskEpisode]],
      our_taken_task_episodes: Set[int],
      max_episode: int,
      num_retries: int = 3) \
      -> Optional[Tuple[int, TaskEpisode]]:
    # This is necessary because sometimes the NFS file system will give us
    # garbage back like "^@^@^@^@^@^" when we're reading files that are
    # being written.  By retrying we get the actual file contents when
    # there's a problem like that.
    for i in range(num_retries):
        try:
            return allocate_next_task_from_file(
              args, filename, all_files,
              file_task_episodes, our_taken_task_episodes,
              max_episode)
        except json.decoder.JSONDecodeError:
            if i == num_retries - 1:
                raise
            eprint("Failed to decode, retrying", guard=args.verbose >= 1)
    return None
def allocate_next_task_from_file(args: argparse.Namespace,
                                 filename: Path,
                                 all_files: List[Path],
                                 file_task_episodes: List[Tuple[int, TaskEpisode]],
                                 our_taken_task_episodes: Set[int],
                                 max_episode: int) \
                                -> Optional[Tuple[int, TaskEpisode]]:
    filepath = args.state_dir / "taken" / ("file-" + safe_abbrev(filename, all_files) + ".txt")
    with filepath.open("r+") as f, FileLock(f):
        # Loading the taken file
        taken_task_episodes: Set[int] = set()
        taken_by_others_this_iter: Set[int] = set()
        for line_num, line in enumerate(f):
            try:
                te_idx, taken_this_iter = json.loads(line)
                taken_task_episodes.add(te_idx)
                if te_idx not in our_taken_task_episodes and taken_this_iter:
                    taken_by_others_this_iter.add(te_idx)
            except json.decoder.JSONDecodeError:
                eprint(f"Failed to parse line {filepath}:{line_num}: \"{line.strip()}\"")
                raise
        # Searching the tasks for a good one to take
        proof_eps_taken_by_others: Set[Tuple[Tuple[str, str, str], int]] = set()
        for file_task_idx, (task_ep_idx, (task, episode)) in enumerate(file_task_episodes):
            if episode > max_episode:
                break
            if task_ep_idx in taken_by_others_this_iter:
                proof_eps_taken_by_others.add((task.to_proof_spec(), episode))
                continue
            if (task_ep_idx in taken_task_episodes
                or (task.to_proof_spec(), episode) in proof_eps_taken_by_others):
                continue
            eprint(f"Found an appropriate task-episode after searching "
                   f"{file_task_idx} task-episodes", guard=args.verbose >= 2)
            print(json.dumps((task_ep_idx, True)), file=f)
            return task_ep_idx, (task, episode)
    return None

def get_num_tasks_taken(args: argparse.Namespace, all_files: List[Path]) -> int:
    tasks_taken = 0
    for filename in all_files:
        with (args.state_dir / "taken" /
              ("file-" + safe_abbrev(filename, all_files) + ".txt")).open("r") as f:
            tasks_taken += sum(1 for _ in f)
    return tasks_taken

if __name__ == "__main__":
    main()
