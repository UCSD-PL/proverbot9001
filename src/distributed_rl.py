#!/usr/bin/env python3

import argparse
import signal
import subprocess
import functools
import sys
import os
import stat
import json
import time
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, OrderedDict
from glob import glob
from dataclasses import dataclass

from tqdm import tqdm, trange

from gen_rl_tasks import RLTask
from search_file import get_all_jobs
from search_worker import project_dicts_from_args, get_predictor
import rl
import util
import torch

num_task_eps_done: int

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
    args.catch_interrupts = True
    args.progress = True
    distributed_rl(args)

def distributed_rl(args: argparse.Namespace):
    check_for_duplicate_run(args)
    all_task_eps = get_all_task_episodes(args)
    num_task_eps_done = setup_jobstate(args, all_task_eps)
    num_workers_actually_needed = min(len(all_task_eps) - num_task_eps_done,
                                      args.num_actors)
    if num_workers_actually_needed > 0:
        if args.start_from is not None:
            _, _, _, (_, _, _, training_args), _, _, _ = \
              torch.load(str(args.start_from), map_location="cpu")
            assert training_args.gamma == args.gamma,\
               "Used a different gamma during training of start_from "\
               "weights than we're using now! "\
               f"{training_args.gamma} start_from vs {args.gamma}"
            assert args.hidden_size == training_args.hidden_size and\
                   args.num_layers == training_args.num_layers,\
               "The provided start_from network doesn't have the same "\
               "architecture as you requested! "\
               f"{training_args.hidden_size} hidden size in start_from vs "\
               f"{args.hidden_size} hidden size requested; "\
               f"{training_args.num_layers} num layers in start_from vs "\
               f"{args.num_layers} num layers requested"
        hidden_size = args.hidden_size
        num_layers = args.num_layers
        dispatch_learner_and_actors(args, num_workers_actually_needed,
                                    hidden_size, num_layers)
        with util.sighandler_context(signal.SIGINT,
                                     functools.partial(interrupt_early, args),
                                     guard=args.catch_interrupts):
            show_progress(args, all_task_eps, num_workers_actually_needed)
    else:
        util.eprint(f"WARNING: there are {num_task_eps_done} tasks eps already done, but only {len(all_task_eps)} task eps total! "
               "This means that something didn't resume properly, or you resumed with a smaller task set")
    time.sleep(5)
    # wait_for_learning_server(args)
    cancel_workers(args)
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
        # util.eprint(f"Running as \"{args_string}\"")
        subprocess.run(verify_args)

def add_distrl_args_to_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-actors", default=32, type=int)
    parser.add_argument("--workers-output-dir", default=Path("output"),
                        type=Path)
    parser.add_argument("--worker-timeout", default="6:00:00")
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--mem", default="2G")
    parser.add_argument("--state-dir", default="drl_state", type=Path)
    parser.add_argument("--keep-latest", default=3, type=int)
    parser.add_argument("--start-after", type=int, default=None)
    parser.add_argument("--ignore-after", type=int, default=None)
    parser.add_argument("--loss-smoothing", type=int, default=1)
    parser.add_argument("--dump-negative-examples", type=Path, default=None)
    parser.add_argument("--dump-replay-buffer", type=Path, default=None)
    parser.add_argument("--no-reset-on-sync", action='store_false', dest='reset_on_sync')
    parser.add_argument("--decrease-lr-on-reset", action='store_true', dest='decrease_lr_on_reset')
    parser.add_argument("--qos", type=str, default=None)
    parser.add_argument("--lr-reset", type=bool, default=False)
    parser.add_argument("--loss", choices=["simple", "log"],
                        default="simple")

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
    #Clearing out taken/file-prooffile for the next step
    for filename in tqdm(all_files,
            desc="Clearing all taken/file-prooffile", leave=False,
            disable=not args.progress):
        with (args.state_dir / "taken" /
              ("file-" + util.safe_abbrev(filename,
                                          all_files) + ".txt")).open("w") as f:
            pass


def prepare_taken_prooffiles(args: argparse.Namespace,
                             all_task_eps: List[Tuple[RLTask, int]])\
                             -> int:

    num_te_encountered = 0
    done_paths = [Path(p) for p in glob(str(args.state_dir / f"done-*.txt"))]
    if len(done_paths) > 0 :
        task_eps_idx_dict = {task_ep: idx for idx, task_ep in enumerate(all_task_eps)}
        all_files = get_all_files(args)

        for done_path in tqdm(done_paths,
                              desc="Preparing taken/file-prooffile for resuming",
                              leave=False, disable=not args.progress):
            file_taken_dict: Dict[Path, List[Tuple[RLTask, int]]] = {}
            with done_path.open('r') as f:
                for line in f :
                    num_te_encountered += 1
                    task_dict, epsiode = json.loads(line)
                    task = RLTask(**task_dict)
                    if Path(task.src_file) in file_taken_dict:
                        file_taken_dict[Path(task.src_file)].append((task, epsiode))
                    else:
                        file_taken_dict[Path(task.src_file)] = [(task, epsiode)]

            write_done_tasks_to_taken_files(args, all_files,task_eps_idx_dict, file_taken_dict )

    for workerid in range(args.num_actors):
        taken_path = args.state_dir / "taken" / f"taken-{workerid}.txt"
        done_path = args.state_dir / f"done-{workerid}.txt"
        with taken_path.open("w") as f:
            pass
        if not done_path.exists():
            with done_path.open("w"):
                pass

    return num_te_encountered

def write_done_tasks_to_taken_files(args : argparse.Namespace,
                                    all_files: List[Path],
                                    task_eps_idx_dict: Dict[Tuple[RLTask,int],int],
                                    file_done_task_eps: Dict[Path, List[Tuple[RLTask, int]]])\
                                    -> None:

    for fidx, filename in enumerate(tqdm(file_done_task_eps.keys(),
                                         desc="For current Done file Writing file taken files",
                                         leave=False,
                                         disable=not args.progress)):
        with (args.state_dir / "taken" /
              ("file-" + util.safe_abbrev(filename,
                                          all_files) + ".txt")).open("a") as f:
            for tidx, task_ep in enumerate(file_done_task_eps.get(filename, [])):
                try:
                    task_ep_idx = task_eps_idx_dict[task_ep]
                except KeyError:
                    util.eprint(f"File {fidx}, task number {tidx} does not exist in task eps indx dict")
                    task, _ep = task_ep
                    for dict_task, _ in task_eps_idx_dict.keys():
                        if task.to_proof_spec() == dict_task.to_proof_spec():
                            util.eprint("There is a task with a matching proof spec!")
                            break
                    raise
                print(json.dumps((task_ep_idx, False)), file=f, flush=True)

# Returns the number of done tasks too because it can be slow to get them and we
# need them to set up job state anyway.
def setup_jobstate(args: argparse.Namespace, all_task_eps: List[Tuple[RLTask, int]]) -> int:
    check_resume(args)
    make_initial_filestructure(args)
    num_task_eps_done = prepare_taken_prooffiles(args,all_task_eps)
    return num_task_eps_done

def dispatch_learner_and_actors(args: argparse.Namespace, num_actors: int,
                                hidden_size: int, num_layers: int):
    predictor: FeaturesPolyargPredictor = get_predictor(args) # type: ignore
    tactic_vocab_size = predictor.prev_tactic_vocab_size
    assert num_actors > 0, num_actors
    cur_dir = os.path.realpath(os.path.dirname(__file__))

    actor_args = ([
                   "--prelude", str(args.prelude),
                   "--state-dir", str(args.state_dir),
                   "-s", str(args.steps_per_episode),
                   "-n", str(args.num_episodes),
                   "-p", str(args.num_predictions),
                   "--hidden-size", str(hidden_size),
                   "--tactic-embedding-size", str(args.tactic_embedding_size),
                   "--num-layers", str(num_layers),
                   "--coq2vec-weights", str(args.coq2vec_weights),
                   "--supervised-weights", str(args.weightsfile),
                   "--starting-epsilon", str(args.starting_epsilon),
                   "--ending-epsilon", str(args.ending_epsilon),
                   "--smoothing-factor", str(args.smoothing_factor),
                   "--backend", args.backend,
                   "--optimizer", args.optimizer,
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
    learner_args = (["--state-dir", str(args.state_dir),
                     "--coq2vec-weights", str(args.coq2vec_weights),
                     "-l", str(args.learning_rate),
                     "-b", str(args.batch_size),
                     "-g", str(args.gamma),
                     "--hidden-size", str(hidden_size),
                     "--tactic-embedding-size", str(args.tactic_embedding_size),
                     "--tactic-vocab-size", str(tactic_vocab_size),
                     "--num-layers", str(num_layers),
                     "--window-size", str(args.window_size),
                     "--train-every", str(args.train_every),
                     "--sync-target-every", str(args.sync_target_every),
                     "--keep-latest", str(args.keep_latest),
                     "--optimizer", args.optimizer,
                     "--scheduler", args.scheduler,
                     "--loss-smoothing", str(args.loss_smoothing),
                     "--learning-rate-decay", str(args.learning_rate_decay),
                     "--loss", args.loss,
                   ] + (["--allow-partial-batches"] if args.allow_partial_batches
                      else [])
                     + (["--start-from", str(args.start_from)]
                        if args.start_from is not None else [])
                     + (["--no-reset-on-sync"]
                        if not args.reset_on_sync else [])
                     + (["--decrease-lr-on-reset"]
                        if args.decrease_lr_on_reset else [])
                     + (["--ignore-after", str(args.ignore_after)]
                        if args.ignore_after is not None else [])
                     + (["--start-after", str(args.start_after)]
                        if args.start_after is not None else [])
                     + (["--learning-rate-step", str(args.learning_rate_step)]
                        if args.learning_rate_step is not None else [])
                     + (["--verifyv-every", str(args.verifyv_every)]
                        if args.verifyv_every is not None else [])
                     + (["-v"] * args.verbose)
                     + (["--dump-negative-examples",
                         str(args.dump_negative_examples)]
                        if args.dump_negative_examples is not None
                        else [])
                     + (["--dump-replay-buffer",
                        str(args.dump_replay_buffer)]
                        if args.dump_replay_buffer is not None
                        else []))
    total_args = ["srun",
                  "-p", args.partition,
                  "--gpus-per-task=1",
                  "-t", args.worker_timeout,
                  "--mem-per-cpu", args.mem,
                  "-n", str(num_actors + 1),
                  "-c", "1",
                  "-J", f"drl-all-{args.output_file}",
                  f"{cur_dir}/dist_dispatch.sh",
                  f"{args.state_dir}/output/",
                  "\" " + " ".join(learner_args) + " \"",
                  "\" " + " ".join(actor_args) + " \""]
    str_args = " ".join(total_args)
    util.eprint(f"Running as {str_args}")
    subprocess.Popen(total_args, stderr=subprocess.DEVNULL)

TaskEpisode = Tuple[RLTask, int]
def get_all_task_episodes(args: argparse.Namespace) -> List[TaskEpisode]:
    if args.tasks_file is not None:
        with open(args.tasks_file, 'r') as f:
            all_tasks = []
            i = 0
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
    global num_task_eps_done
    num_task_eps_progress = get_num_task_eps_done(args)
    num_task_eps_done = 0
    scheduled_actors: List[int] = []
    crashed_actors: List[int] = []
    learner_is_scheduled = False
    start_time = time.time()
    actor_end_time = None
    with tqdm(desc="Task-episodes finished", total=len(all_task_eps),
              initial=num_task_eps_progress, dynamic_ncols=True,
              disable=not args.progress) as task_eps_bar, \
         tqdm(desc="Actors scheduled", total=num_actors_dispatched,
              dynamic_ncols=True, disable=not args.progress) as wbar:
        while num_task_eps_done < len(all_task_eps):
            num_task_eps_done = get_num_task_eps_done(args)
            # Update the bar with the tasks that have been finished
            new_num_task_eps_progress = get_num_task_eps_done(args)
            task_eps_bar.update(new_num_task_eps_progress - num_task_eps_progress)
            num_task_eps_progress = new_num_task_eps_progress

            if actor_end_time is not None:
                assert time.time() - actor_end_time < timedelta(minutes=2), \
                    f"It's been more than two minutes since the actors exited, "\
                    f"and we're still at {num_task_eps_done} out of {len(all_task_eps)}. "\
                    "Exiting..."

            if learner_is_scheduled:
                if actor_end_time is None:
                    job_id_output = subprocess.check_output(
                      [f"./src/squeue-retry.sh -tR -u$USER "\
                       f"-n drl-all-{args.output_file} -o%A -h"],
                      shell=True, text=True).split("\n", maxsplit=1)[0].strip()
                    if job_id_output == "":
                        actor_end_time = time.time()
            else:
                learner_is_scheduled = (args.state_dir / "learner_scheduled.txt").exists()
            if timedelta(seconds=time.time() - start_time) >= parse_timeout(args.worker_timeout) - timedelta(minutes=1):
                interrupt_early(args)
            # Get the new scheduled actors, and update the worker
            # bar.
            with (args.state_dir / "actors_scheduled.txt").open('r') as f:
                new_scheduled_actors = list(f)
            wbar.update(len(new_scheduled_actors) - len(scheduled_actors))
            scheduled_actors = new_scheduled_actors

def parse_timeout(timestring: str) -> timedelta:
    try:
        t = datetime.strptime(timestring, "%H:%M:%S")
    except ValueError:
        t = datetime.strptime(timestring, "%M:%S")
    return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)

def wait_for_learning_server(args: argparse.Namespace) -> None:
    squeue_all_output = subprocess.check_output(
      [f"squeue -u$USER -tR -n drl-all-{args.output_file} -o%A -h"],
      shell=True, text=True).split("\n")[0].strip()
    if squeue_all_output == "":
        return
    job_id = int(squeue_all_output)
    while True:
        squeue_step_output = subprocess.check_output(
            f"squeue -r -tR -u$USER -h -s{job_id}.0",
            shell=True, text=True)
        if squeue_step_output.strip() == "":
            break
        time.sleep(0.1)

def get_num_task_eps_done(args: argparse.Namespace) -> int:
    num_task_eps_done: int = 0

    for done_path in (Path(p) for p in glob(str(args.state_dir / f"done-*.txt"))):
        with done_path.open('r') as f:
            num_task_eps_done += len(f.readlines())
    return num_task_eps_done

def interrupt_early(args: argparse.Namespace, *_rest_args) -> None:
    util.eprint("Interrupting but saving results (interrupt again to cancel)")
    interrupt_learning_server(args)
    # wait_for_learning_server(args)
    time.sleep(5)
    cancel_workers(args)
    build_final_save(args, num_task_eps_done)
    sys.exit(1)

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

def interrupt_learning_server(args: argparse.Namespace) -> None:
    # job_node = subprocess.check_output(
    #   f"squeue -h -n drl-all-{args.output_file} -o %N | head -n 1",
    #   shell=True, text=True).split("\n")[0].strip()
    # assert job_node != "", \
    #   "Can't save final weights because learning server is already dead!"
    # multi_node_match = re.match("([a-zA-Z0-9]+)\[([a-zA-Z0-9]+,.*)\]", job_node)
    # if multi_node_match:
    #   job_node = multi_node_match.group(1) + multi_node_match.group(2)
    # assert "[" not in job_node
    job_id_output = subprocess.check_output(
      [f"squeue -u$USER -tR -n drl-all-{args.output_file} -o%A -h"],
      shell=True, text=True).split("\n")[0].strip()
    assert job_id_output != ""
    job_id = int(job_id_output)
    subprocess.check_output(f"scancel -u$USER -sSIGINT {job_id}",
      shell=True, text=True)
    #output = subprocess.check_output(
    #  f"ssh {job_node} scontrol listpids {job_id} | tail -n+2 "
    #  "| awk '($3 ~/0/) && ($4 ~/-/) { print $1 }' "
    #  f"| ssh {job_node} xargs kill -2",
    #  shell=True, text=True)
    #assert output.strip() == "", output.strip()
    pass

def cancel_workers(args: argparse.Namespace) -> None:
    subprocess.run([f"scancel -u$USER -n drl-all-{args.output_file}"],
                   shell=True, check=True)

def build_final_save(args: argparse.Namespace, steps_done: int) -> None:
    save_path = latest_common_save(args)
    predictor: FeaturesPolyargPredictor = get_predictor(args) # type: ignore
    tactic_vocab_size = predictor.prev_tactic_vocab_size
    args.tactic_vocab_size = tactic_vocab_size
    assert save_path is not None, \
      "We've reached the end of training, but no common weights are found " \
      "in the weights directory!"
    common_network_weights_dict = torch.load(str(save_path), map_location="cpu")
    obl_encoder_state = torch.load(args.coq2vec_weights, map_location="cpu")
    v_network_state: Tuple[dict, Any,
                           OrderedDict[Any, torch.FloatTensor],
                           argparse.Namespace] = \
                           (common_network_weights_dict, obl_encoder_state,
                            OrderedDict(), args)
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

@dataclass
class EObligation:
  contents: torch.FloatTensor
  def __hash__(self) -> int:
    return hash(tuple(self.contents.view(-1).tolist()))
  def __eq__(self, other: object) -> bool:
    if not isinstance(other, EObligation):
      return False
    return bool(torch.all(self.contents == other.contents))

def check_for_duplicate_run(args: argparse.Namespace) -> None:
    output = subprocess.check_output(
      [f"squeue -u$USER -tR -n drl-all-{args.output_file} -h"],
      shell=True, text=True)
    assert output.strip() == "", \
      "There's already an instance running with that output file! "\
      "Having two would mess up job control"

if __name__ == "__main__":
    main()
