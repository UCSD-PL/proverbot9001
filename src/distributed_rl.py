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
from typing import List, Tuple, Optional, Dict
from glob import glob

from tqdm import tqdm, trange

from gen_rl_tasks import RLTask
import rl
import util
import torch

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a state estimator using reinforcement learning"
        "to complete proofs using Proverbot9001.")
    rl.add_args_to_parser(parser)
    add_distrl_args_to_parser(parser)
    args = parser.parse_args()

    task_eps_done = setup_jobstate(args)
    all_task_eps = get_all_task_episodes(args)
    num_workers_actually_needed = min(len(all_task_eps) - len(task_eps_done),
                                      args.num_actors)
    hostname = dispatch_learning_server(args)
    dispatch_actors(args, num_workers_actually_needed, sys.argv[1:], hostname)
    with util.sighandler_context(signal.SIGINT,
                                 functools.partial(interrupt_early, args)):
        show_progress(args, num_workers_actually_needed)

    if args.verifyvval:
        cur_dir = os.path.realpath(os.path.dirname(__file__))
        subprocess.run([f"srun",
                        "--pty",
                        "-J", "drl-verify-worker",
                        "-p", "cpu",
                        "python",
                        f"{cur_dir}/rl.py",
                        "--supervised-weights", args.weightsfile,
                        "--coq2vec-weights", args.coq2vec_weights,
                        "--tasks-file", args.tasks_file,
                        "--prelude", args.prelude,
                        "-o", args.output_file,
                        "--gamma", str(args.gamma),
                        "-n", "0",
                        "--resume", "yes",
                        "--verifyvval",
                        ] +
                        args.filenames)

def add_distrl_args_to_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-actors", default=32, type=int)
    parser.add_argument("--workers-output-dir", default=Path("output"),
                        type=Path)
    parser.add_argument("--worker-timeout", default="6:00:00")
    parser.add_argument("--partition", default="cpu")
    parser.add_argument("--learning-partition", default="gpu")
    parser.add_argument("--mem", default="2G")
    parser.add_argument("--state_dir", default="drl_state", type=Path)
    parser.add_argument("--keep-latest", default=3, type=int)
    parser.add_argument("--port", default=9000)

def check_resume(args: argparse.Namespace) -> None:
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

    if resume == "no":
        shutil.rmtree(str(args.state_dir))

def make_initial_filestructure(args: argparse.Namespace) -> None:
    args.state_dir.mkdir(exist_ok=True)
    (args.state_dir / args.workers_output_dir).mkdir(exist_ok=True)
    (args.state_dir / "weights").mkdir(exist_ok=True)
    (args.state_dir / "taken").mkdir(exist_ok=True)
    taken_path = args.state_dir / "taken" / "taken-files.txt"
    if not taken_path.exists():
        with taken_path.open('w'):
            pass
    with (args.state_dir / "actors_scheduled.txt").open('w'):
        pass

    (args.state_dir / "shorter_proofs").mkdir(exist_ok=True)
    all_files = get_all_files(args)
    for filename in all_files:
        shorter_path = (args.state_dir / "shorter_proofs" /
                        (util.safe_abbrev(filename, all_files) + ".json"))
        if not shorter_path.exists():
            with shorter_path.open('w') as f:
                pass

def get_file_taken_tasks(args: argparse.Namespace) -> Dict[Path, List[Tuple[RLTask, int]]]:
    file_taken_dict: Dict[Path, List[Tuple[RLTask, int]]] = {}
    for workerid in trange(args.num_actors, desc="Loading done task episodes", leave=False):
        done_path = args.state_dir / f"done-{workerid}.txt"
        if not done_path.exists():
            with done_path.open("w"):
                pass
            worker_done_task_eps = []
        else:
            with done_path.open('r') as f:
                worker_done_task_eps = [(RLTask(**task_dict), episode)
                                        for line in f
                                        for task_dict, episode in (json.loads(line),)]
            for task, ep in worker_done_task_eps:
                if task.src_file in file_taken_dict:
                    file_taken_dict[Path(task.src_file)].append((task, ep))
                else:
                    file_taken_dict[Path(task.src_file)] = [(task, ep)]

        taken_path = args.state_dir / "taken" / f"taken-{workerid}.txt"
        with taken_path.open("w") as f:
            pass
        progress_path = args.state_dir / f"progress-{workerid}.txt"
        with progress_path.open("w") as f:
            for task, ep in worker_done_task_eps:
                print(json.dumps((task.as_dict(), ep)), file=f, flush=True)
    return file_taken_dict

def write_done_tasks_to_taken_files(args: argparse.Namespace,
                                    file_done_task_eps: Dict[Path, List[Tuple[RLTask, int]]])\
                                    -> None:
    all_task_eps = get_all_task_episodes(args)
    task_eps_idx_dict = {task_ep: idx for idx, task_ep in enumerate(all_task_eps)}
    all_files = get_all_files(args)
    for fidx, filename in enumerate(tqdm(all_files,
                                         desc="Writing file taken files", leave=False)):
        with (args.state_dir / "taken" /
              ("file-" + util.safe_abbrev(filename,
                                          all_files) + ".txt")).open("w") as f:
            for tidx, task_ep in enumerate(file_done_task_eps.get(filename, [])):
                try:
                    task_ep_idx = task_eps_idx_dict[task_ep]
                except KeyError:
                    util.eprint(f"File number {fidx}, task number {tidx}")
                    task, _ep = task_ep
                    for dict_task, _ in task_eps_idx_dict.keys():
                        if task.to_proof_spec() == dict_task.to_proof_spec():
                            util.eprint("There is a task with a matching proof spec!")
                            break
                    raise
                print(json.dumps((task_ep_idx, False)), file=f, flush=True)

# Returns the list of done tasks too because it can be slow to get them and we
# need them to set up job state anyway.
def setup_jobstate(args: argparse.Namespace) -> List[Tuple[RLTask, int]]:
    check_resume(args)
    make_initial_filestructure(args)

    file_taken_dict = get_file_taken_tasks(args)

    write_done_tasks_to_taken_files(args, file_taken_dict)

    done_task_eps = []
    for _filename, file_done_task_eps in file_taken_dict.items():
        done_task_eps += file_done_task_eps

    return done_task_eps

def dispatch_actors(args: argparse.Namespace,
                    num_actors_to_dispatch: int,
                    rest_args: List[str],
                    server_hostname: str) -> None:
    with (args.state_dir / "workers_scheduled.txt").open('w'):
        pass
    assert num_actors_to_dispatch > 0, num_actors_to_dispatch

    cur_dir = os.path.realpath(os.path.dirname(__file__))
    subprocess.run([f"{cur_dir}/sbatch-retry.sh",
                    "-J", f"drl-actor-{args.output_file}",
                    "-p", args.partition,
                    "-t", str(args.worker_timeout),
                    "-o", str(args.state_dir / args.workers_output_dir
                              / "actor-%a.out"),
                    "--mem", args.mem,
                    f"--array=0-{num_actors_to_dispatch-1}",
                    f"{cur_dir}/distributed_rl_acting_worker.py",
                    "--tasks-file", args.tasks_file,
                    "--state-dir", args.state_dir,
                    "-s", str(args.steps_per_episode),
                    "-n", str(args.num_episodes),
                    "-p", str(args.num_predictions),
                    "--coq2vec-weights", str(args.coq2vec_weights),
                    "--starting-epsilon", str(args.starting_epsilon),
                    "--ending-epsilon", str(args.ending_epsilon),
                    "-H", server_hostname,
                    "-P", str(args.port),
                    "--num-actors", str(args.num_actors),
                    ] + (["--curriculum"] if args.curriculum else [])
                   + (["--no-interleave"] if not args.interleave else [])
                   + ["--blacklist-tactic={tactic}" for tactic
                      in args.blacklisted_tactics]
                   + (["-" + "v"*args.verbose] if args.verbose > 0 else [])
                   + args.filenames,
                   check=False)

def dispatch_learning_server(args: argparse.Namespace) -> str:
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    hidden_size = torch.load(args.coq2vec_weights, map_location="cpu")[5]
    num_hyps = 5
    encoding_size = hidden_size * (num_hyps + 1)
    server_jobname = f"drl-learner-{args.output_file}"
    subprocess.run([f"{cur_dir}/sbatch-retry.sh",
                    "-J", server_jobname,
                    "-p", args.learning_partition,
                    "-t", str(args.worker_timeout),
                    "-o", str(args.state_dir / args.workers_output_dir / "learner.out"),
                    "--mem", args.mem,
                    f"{cur_dir}/distributed_rl_learning_server.py",
                    "-e", str(encoding_size),
                    "-n", str(args.num_actors),
                    "-p", str(args.port),
                    "-l", str(args.learning_rate),
                    "--window-size", str(args.window_size),
                    "--train-every", str(args.train_every)]
                   + (["--allow-partial-batches"] if args.allow_partial_batches 
                      else []),
                   check=False)
    while not is_server_running(server_jobname):
        time.sleep(0.2)

    return server_node_string(server_jobname)

def is_server_running(jobname: str) -> bool:
  status_string = subprocess.check_output(
    f"squeue -u$USER -h -n {jobname} -o %t",
    shell=True, text=True)
  if status_string.strip() == "R":
    return True
  return False

def server_node_string(jobname: str) -> str:
  return subprocess.check_output(
    f"squeue -u$USER -h -n {jobname} -o %N",
    shell=True, text=True).strip()

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

def get_all_files(args: argparse.Namespace) -> List[Path]:
    with open(args.tasks_file, 'r') as f:
        return list({Path(json.loads(line)["src_file"]) for line in f})

def show_progress(args: argparse.Namespace, num_actors_dispatched: int) -> None:
    all_task_eps = get_all_task_episodes(args)
    num_task_eps_progress = get_num_task_eps_progress(args)
    num_task_eps_done = 0
    scheduled_actors: List[int] = []
    crashed_actors: List[int] = []
    with tqdm(desc="Task-episodes finished", total=len(all_task_eps),
              initial=num_task_eps_progress, dynamic_ncols=True) as task_eps_bar, \
         tqdm(desc="Actors scheduled", total=num_actors_dispatched,
              dynamic_ncols=True) as wbar:
        while num_task_eps_done < len(all_task_eps):
            num_task_eps_done = get_num_task_eps_done(args)
            # Update the bar with the tasks that have been finished
            new_num_task_eps_progress = get_num_task_eps_progress(args)
            task_eps_bar.update(new_num_task_eps_progress - num_task_eps_progress)
            num_task_eps_progress = new_num_task_eps_progress
            # Update the actors scheduled bar with actors that have been newly scheduled
            with (args.state_dir / "actors_scheduled.txt").open('r') as f:
                new_scheduled_actors = list(f)

            num_actors_alive = check_for_crashed_actors(args, crashed_actors)
            # If all actors are dead, and we're not done with the
            # jobs, print a message and exit (we'll just exit if the
            # jobs are all done at the while condition)
            if num_actors_alive == 0 and num_task_eps_progress < len(all_task_eps):
                util.eprint("All actors exited, but jobs aren't done!")
                cancel_workers(args)
                sys.exit(1)
            check_for_learning_worker(args)

            # Get the new scheduled actors, and update the worker
            # bar.
            with (args.state_dir / "actors_scheduled.txt").open('r') as f:
                new_scheduled_actors = list(f)
            wbar.update(len(new_scheduled_actors) - len(scheduled_actors))
            scheduled_actors = new_scheduled_actors

def check_for_learning_worker(args: argparse.Namespace) -> None:
    # If the syncing worker dies, print a message and exit
    squeue_output = subprocess.check_output(
        f"squeue -r -u$USER -h -n drl-learner-{args.output_file}",
        shell=True, text=True)
    if squeue_output.strip() == "":
        util.eprint("Learning server died! Check the logs for more details. "
                    "Exiting...")
        cancel_workers(args)
        sys.exit(1)

def check_for_crashed_actors(args: argparse.Namespace, 
                             crashed_actors: List[int]) -> int:
    # Get the workers that are alive
    squeue_output = subprocess.check_output(
        f"squeue -r -u$USER -h -n drl-actor-{args.output_file} -o%K",
        shell=True, text=True)
    new_workers_alive = [int(wid) for wid in
                         squeue_output.strip().split("\n")
                         if wid != ""]
    # Wait a bit between doing this and checking the tasks
    # done. This is because a worker might finish and stop
    # showing up in squeue before it's writes to the "done"
    # file are fully propagated.
    time.sleep(0.1)
    # Check for newly crashed workers
    for worker_id in range(args.num_actors):
        # Skip any living worker
        if worker_id in new_workers_alive:
            continue
        # Skip any worker for which we've already reported a crash
        if worker_id in crashed_actors:
            continue
        # Get the jobs taken by workers.
        with (args.state_dir / "taken" / f"taken-{worker_id}.txt").open('r') as f:
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
            crashed_actors.append(worker_id)
    return len(new_workers_alive)

def get_num_task_eps_done(args: argparse.Namespace) -> int:
    num_task_eps_done: int = 0

    for workerid in range(args.num_actors):
        with (args.state_dir / f"done-{workerid}.txt").open('r') as f:
            num_task_eps_done += len(f.readlines())
    return num_task_eps_done

def get_num_task_eps_progress(args: argparse.Namespace) -> int:
    num_task_eps_done = 0

    for workerid in range(args.num_actors):
        with (args.state_dir / f"progress-{workerid}.txt").open('r') as f:
            num_task_eps_done += sum((1 for _ in f))
    return num_task_eps_done

def interrupt_early(args: argparse.Namespace, *_rest_args) -> None:
    cancel_workers(args)
    sys.exit(0)

def latest_worker_save_num(args: argparse.Namespace,
                           workerid: int) -> Optional[int]:
    root_dir = str(args.state_dir / "weights")
    current_working_directory = os.getcwd()
    os.chdir(root_dir)
    worker_networks = glob(f"worker-{workerid}-network-*.dat")
    os.chdir(current_working_directory)
    #worker_networks = glob(str(args.state_dir / "weights" / "worker-{workerid}-network-*.dat"),
    #                       root_dir = str(args.state_dir / "weights"))

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
    subprocess.run([f"scancel -u$USER -n drl-actor-{args.output_file}"],
                   shell=True, check=True)
    subprocess.run([f"scancel -u$USER -n drl-learner-{args.output_file}"],
                   shell=True, check=True)

if __name__ == "__main__":
    main()
