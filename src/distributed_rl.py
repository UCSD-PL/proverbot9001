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
from typing import List, Tuple, Optional, Dict, Any, OrderedDict
from glob import glob

from tqdm import tqdm, trange

from gen_rl_tasks import RLTask
from search_file import get_all_jobs
from search_worker import project_dicts_from_args
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

    if args.filenames[0].suffix == ".json":
        args.splits_file = args.filenames[0]
    else:
        args.splits_file = None

    all_task_eps = get_all_task_episodes(args)
    task_eps_done = setup_jobstate(args, all_task_eps)
    num_workers_actually_needed = min(len(all_task_eps) - len(task_eps_done),
                                      args.num_actors)
    if num_workers_actually_needed > 0:
        dispatch_learner_and_actors(args, num_workers_actually_needed)
        with util.sighandler_context(signal.SIGINT,
                                     functools.partial(interrupt_early, args)):
            show_progress(args, all_task_eps, num_workers_actually_needed)
        cancel_workers(args)
    elif num_workers_actually_needed < 0:
        util.eprint(f"WARNING: there are {len(task_eps_done)} tasks eps already done, but only {len(all_task_eps)} task eps total! "
               "This means that something didn't resume properly, or you resumed with a smaller task set")
    build_final_save(args, len(all_task_eps))

    if args.verifyvval:
        cur_dir = os.path.realpath(os.path.dirname(__file__))
        verify_args = ([f"srun",
                 "--pty",
                 "-J", "drl-verify-worker",
                 "-p", "cpu",
                 "python",
                 f"{cur_dir}/rl.py",
                 "--supervised-weights", str(args.weightsfile),
                 "--coq2vec-weights", str(args.coq2vec_weights),
                 "--tasks-file", str(args.tasks_file),
                 "--prelude", str(args.prelude),
                 "-o", str(args.output_file),
                 "--gamma", str(args.gamma),
                 "-n", "0",
                 "--resume", "yes",
                 "--verifyvval",
                 ] +
                 [str(p) for p in args.filenames])
        args_string = " ".join(verify_args)
        util.eprint(f"Running as \"{args_string}\"")
        subprocess.run(verify_args)

def add_distrl_args_to_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-actors", default=32, type=int)
    parser.add_argument("--workers-output-dir", default=Path("output"),
                        type=Path)
    parser.add_argument("--worker-timeout", default="6:00:00")
    parser.add_argument("--partition", default="cpu")
    parser.add_argument("--learning-partition", default="gpu")
    parser.add_argument("--mem", default="2G")
    parser.add_argument("--state-dir", default="drl_state", type=Path)
    parser.add_argument("--keep-latest", default=3, type=int)
    parser.add_argument("--sync-workers-every", type=int, default=16)

def check_resume(args: argparse.Namespace) -> None:
    resume_exists = len(glob(str(args.state_dir / "weights" / "common-v-network-*.dat"))) > 0
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

    if resume == "no" and args.state_dir.exists():
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

    (args.state_dir / "learner_scheduled.txt").unlink(missing_ok=True)

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
    for done_path in (Path(p) for p in glob(str(args.state_dir / f"done-*.txt"))):
        with done_path.open('r') as f:
            worker_done_task_eps = [(RLTask(**task_dict), episode)
                                    for line in f
                                    for task_dict, episode in (json.loads(line),)]
        for task, ep in worker_done_task_eps:
            if Path(task.src_file) in file_taken_dict:
                file_taken_dict[Path(task.src_file)].append((task, ep))
            else:
                file_taken_dict[Path(task.src_file)] = [(task, ep)]

    for workerid in range(args.num_actors):
        taken_path = args.state_dir / "taken" / f"taken-{workerid}.txt"
        done_path = args.state_dir / f"done-{workerid}.txt"
        with taken_path.open("w") as f:
            pass
        if not done_path.exists():
            with done_path.open("w"):
                pass
    return file_taken_dict

def write_done_tasks_to_taken_files(args: argparse.Namespace,
                                    all_task_eps: List[Tuple[RLTask, int]],
                                    file_done_task_eps: Dict[Path, List[Tuple[RLTask, int]]])\
                                    -> None:
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
def setup_jobstate(args: argparse.Namespace, all_task_eps: List[Tuple[RLTask, int]]) -> List[Tuple[RLTask, int]]:
    check_resume(args)
    make_initial_filestructure(args)

    file_taken_dict = get_file_taken_tasks(args)

    write_done_tasks_to_taken_files(args, all_task_eps, file_taken_dict)

    done_task_eps = []
    for _filename, file_done_task_eps in file_taken_dict.items():
        done_task_eps += file_done_task_eps

    return done_task_eps

def dispatch_learner_and_actors(args: argparse.Namespace, num_actors: int):
    with (args.state_dir / "workers_scheduled.txt").open('w'):
        pass
    assert num_actors > 0, num_actors
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    hidden_size = torch.load(args.coq2vec_weights, map_location="cpu")[5]
    num_hyps = 5
    encoding_size = hidden_size * (num_hyps + 1)
    server_jobname = f"drl-learner-{args.output_file}"
    actor_jobname = f"drl-actor-{args.output_file}"

    actor_job_args = (["-J", actor_jobname,
                   "-p", args.partition,
                   "-t", str(args.worker_timeout),
                   "--mem", args.mem,
                   "--kill-on-bad-exit"])
    actor_script_args = ([
                   "python", f"{cur_dir}/distributed_rl_acting_worker.py",
                   "--prelude", str(args.prelude),
                   "--state-dir", str(args.state_dir),
                   "-s", str(args.steps_per_episode),
                   "-n", str(args.num_episodes),
                   "-p", str(args.num_predictions),
                   "--coq2vec-weights", str(args.coq2vec_weights),
                   "--supervised-weights", str(args.weightsfile),
                   "--starting-epsilon", str(args.starting_epsilon),
                   "--ending-epsilon", str(args.ending_epsilon),
                   "--smoothing-factor", str(args.smoothing_factor),
                   "--backend", args.backend,
                   ] + (["--curriculum"] if args.curriculum else []) +
                   (["--no-interleave"] if not args.interleave else []) + 
                   (["--progress"] if args.progress else []) +
                   (["--tasks-file", str(args.tasks_file)]
                    if args.tasks_file is not None else []) +
                   (["--include-proof-relevant"] if args.include_proof_relevant else []) +
                   ["--blacklist-tactic={tactic}" for tactic
                    in args.blacklisted_tactics] +
                   (["-" + "v"*args.verbose] if args.verbose > 0 else []) + 
                   (["-t"] if args.print_timings else []) +
                   [str(f) for f in args.filenames])
    learner_args = (["-J", server_jobname,
                     "-p", args.learning_partition,
                     "-t", str(args.worker_timeout),
                     "-o", str(args.state_dir / args.workers_output_dir / "learner.out"),
                     "--mem", args.mem,
                     "--gres=gpu:1",
                     "--kill-on-bad-exit",
                     "python", f"{cur_dir}/distributed_rl_learning_server.py",
                     "--state-dir", str(args.state_dir),
                     "-e", str(encoding_size),
                     "-l", str(args.learning_rate),
                     "-b", str(args.batch_size),
                     "-g", str(args.gamma),
                     "--window-size", str(args.window_size),
                     "--train-every", str(args.train_every),
                     "--sync-target-every", str(args.sync_target_every),
                     "--sync-workers-every", str(args.sync_workers_every),
                     "--keep-latest", str(args.keep_latest),
                   ] + (["--allow-partial-batches"] if args.allow_partial_batches 
                      else []))
    total_args = ["srun"] + learner_args
    for workerid in range(num_actors):
        total_args += ([":"] + actor_job_args +
                       ["-o", str(args.state_dir / args.workers_output_dir
                             / f"actor-{workerid}.out")] +
                       actor_script_args + ["-w", str(workerid)])
    args_string = " ".join(total_args)
    subprocess.Popen(total_args, stderr=subprocess.DEVNULL)

TaskEpisode = Tuple[RLTask, int]
def get_all_task_episodes(args: argparse.Namespace) -> List[TaskEpisode]:
    if args.tasks_file is not None:
        with open(args.tasks_file, 'r') as f:
            all_tasks = [RLTask(**json.loads(line)) for line in f]
    else:
        all_tasks = [RLTask.from_job(job) for job in get_all_jobs(args)]

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
    if args.tasks_file:
        with open(args.tasks_file, 'r') as f:
            return list({Path(json.loads(line)["src_file"]) for line in f})
    else:
        project_dicts = project_dicts_from_args(args)
        return [Path(filename) for project_dict in project_dicts
                for filename in project_dict["test_files"]]

def show_progress(args: argparse.Namespace, all_task_eps: List[Tuple[RLTask, int]], num_actors_dispatched: int) -> None:
    num_task_eps_progress = get_num_task_eps_done(args)
    num_task_eps_done = 0
    scheduled_actors: List[int] = []
    crashed_actors: List[int] = []
    learner_is_scheduled = False
    with tqdm(desc="Task-episodes finished", total=len(all_task_eps),
              initial=num_task_eps_progress, dynamic_ncols=True) as task_eps_bar, \
         tqdm(desc="Actors scheduled", total=num_actors_dispatched,
              dynamic_ncols=True) as wbar:
        while num_task_eps_done < len(all_task_eps):
            num_task_eps_done = get_num_task_eps_done(args)
            # Update the bar with the tasks that have been finished
            new_num_task_eps_progress = get_num_task_eps_done(args)
            task_eps_bar.update(new_num_task_eps_progress - num_task_eps_progress)
            num_task_eps_progress = new_num_task_eps_progress

            num_actors_alive = check_for_crashed_actors(args, crashed_actors)
            # If all actors are dead, and we're not done with the
            # jobs, print a message and exit (we'll just exit if the
            # jobs are all done at the while condition)
            if num_actors_alive == 0 and len(scheduled_actors) == num_actors_dispatched and num_task_eps_progress < len(all_task_eps):
                util.eprint("All actors exited, but jobs aren't done!")
                cancel_workers(args)
                sys.exit(1)
            if learner_is_scheduled:
                check_for_learning_worker(args)
            else:
                learner_is_scheduled = (args.state_dir / "learner_scheduled.txt").exists()
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
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    squeue_output = subprocess.check_output(
        f"{cur_dir}/squeue-retry.sh -r -u$USER -h -n drl-actor-{args.output_file} -OHetJobOffset",
        shell=True, text=True)
    new_workers_alive = [int(wid) - 1 for wid in
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
    return len(new_workers_alive) - len(crashed_actors)

def get_num_task_eps_done(args: argparse.Namespace) -> int:
    num_task_eps_done: int = 0

    for done_path in (Path(p) for p in glob(str(args.state_dir / f"done-*.txt"))):
        with done_path.open('r') as f:
            num_task_eps_done += len(f.readlines())
    return num_task_eps_done

def interrupt_early(args: argparse.Namespace, *_rest_args) -> None:
    cancel_workers(args)
    sys.exit(0)

def latest_common_save_num(args: argparse.Namespace) -> Optional[int]:
    root_dir = str(args.state_dir / "weights")
    cwd = os.getcwd()
    os.chdir(root_dir)
    worker_networks = glob("common-v-network-*.dat")
    os.chdir(cwd)
    if len(worker_networks) == 0:
        return None
    return max(int(util.unwrap(re.match(
             r"common-v-network-(\d+).dat",
             path)).group(1))
            for path in worker_networks)

def latest_common_save(args: argparse.Namespace) -> Optional[Path]:
    latest_save_num = latest_common_save_num(args)
    if latest_save_num is None:
        return None
    return (args.state_dir / "weights" /
            f"common-v-network-{latest_save_num}.dat")
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
    subprocess.run([f"scancel -u$USER -n drl-actor-{args.output_file} -n drl-learner-{args.output_file}"],
                   shell=True, check=True)

def build_final_save(args: argparse.Namespace, steps_done: int) -> None:
    save_path = latest_common_save(args)
    assert save_path is not None, \
      "We've reached the end of training, but no common weights are found " \
      "in the weights directory!"
    common_network_weights_dict = torch.load(str(save_path), map_location="cpu")
    obl_encoder_state = torch.load(args.coq2vec_weights, map_location="cpu")
    v_network_state: Tuple[dict, Any, OrderedDict[Any, torch.FloatTensor]] = \
                           (common_network_weights_dict, obl_encoder_state, OrderedDict())
    with args.output_file.open('wb') as f:
        torch.save((False, None,
                    steps_done, v_network_state, v_network_state,
                    get_shorter_proofs_dict(args), None), f)

def get_shorter_proofs_dict(args: argparse.Namespace) -> Dict[RLTask, int]:
    dict_entries = []
    all_files = get_all_files(args)
    for filename in all_files:
        shorter_proofs_path = (args.state_dir / "shorter_proofs" /
                               (util.safe_abbrev(filename, all_files) + ".json"))
        with shorter_proofs_path.open("r") as f, util.FileLock(f):
            dict_entries += [(RLTask(**task_dict), shorter_length)
                             for l in f for task_dict, shorter_length in (json.loads(l),)]

    return dict(dict_entries)

if __name__ == "__main__":
    main()
