import argparse
import json
import coq2vec
import random
import torch
import os
from typing import (List, Optional, Dict, Tuple, Union, Any, Set,
                    Sequence, TypeVar, Callable)

from util import (nostderr, FileLock, eprint,
                  print_time, unwrap, safe_abbrev)
from pathlib import Path
from search_worker import get_predictor
from search_file import get_all_jobs
from gen_rl_tasks import RLTask
from rl import ReinforcementWorker, MemoizingPredictor, switch_dict_from_args, VNetwork, ReplayBuffer, tactic_prefix_is_usable
from distributed_rl_learning_worker import TaskEpisode, get_all_task_episodes
from distributed_rl_eval import get_all_tasks
from collections import Counter

def main() -> None:
    eprint("Starting main")
    parser = argparse.ArgumentParser(
        description="Evaluation worker - The rl agent"
        "to complete proofs using Proverbot9001.")
    parser.add_argument("--prelude", default=".", type=Path)
    parser.add_argument("--rl-weights", "-r",
                        help="output data folder name",
                        default="data/rl_weights.dat",
                        type=Path)
    parser.add_argument("--verbose", "-v", help="verbose output",
                        action="count", default=0)
    parser.add_argument("--progress", "-P", help="show progress of files",
                        action='store_true')
    parser.add_argument("--print-timings", "-t", action='store_true')
    parser.add_argument("--no-set-switch", dest="set_switch", action='store_false')
    parser.add_argument("--include-proof-relevant", action="store_true")
    parser.add_argument("--backend", choices=['serapi', 'lsp', 'auto'], default='auto')
    parser.add_argument('filenames', help="proof file name (*.v)",
                        nargs='+', type=Path)
    proofsGroup = parser.add_mutually_exclusive_group()
    proofsGroup.add_argument("--proof", default=None)
    proofsGroup.add_argument("--proofs-file", default=None)
    parser.add_argument("--tasks-file", default=None)
    parser.add_argument('--supervised-weights', type=Path, dest="weightsfile")
    parser.add_argument("--coq2vec-weights", type=Path)
    parser.add_argument("--max-sertop-workers", default=16, type=int)
    parser.add_argument("--max-attempts", default=16, type=int)
    parser.add_argument("--blacklist-tactic", action="append",
                        dest="blacklisted_tactics")
    parser.add_argument("-s", "--steps-per-episode", default=16, type=int)
    parser.add_argument("--resume", choices=["no", "yes", "ask"], default="ask")
    evalGroup = parser.add_mutually_exclusive_group()
    evalGroup.add_argument("--evaluate", action="store_true")
    evalGroup.add_argument("--evaluate-baseline", action="store_true")
    evalGroup.add_argument("--evaluate-random-baseline", action="store_true")
    parser.add_argument("--state-dir", default="drl_eval_state", type=Path)
    
    
    args = parser.parse_args()
    args.output_dir = args.state_dir

    
    (args.state_dir).mkdir(exist_ok=True, parents=True)

    if args.filenames[0].suffix == ".json":
        args.splits_file = args.filenames[0]
        args.filenames = None
    else:
        args.splits_file = None

    workerid = int(os.environ['SLURM_ARRAY_TASK_ID'])
    jobid = int(os.environ['SLURM_ARRAY_JOB_ID'])
    eprint(f"Starting worker {workerid}")
    evaluation_worker(args,workerid,jobid)




def evaluation_worker(args: argparse.Namespace, workerid: int, jobid: int) -> None:
            
    tasks = get_all_tasks(args)
    file_all_ts_dict: Dict[Path, List[Tuple[int, RLTask]]] = {}
    for task_idx, task in enumerate(tasks):
        if Path(task.src_file) in file_all_ts_dict:
            file_all_ts_dict[Path(task.src_file)].append((task_idx, task))
        else:
            file_all_ts_dict[Path(task.src_file)] = [(task_idx,task )]

    predictor = get_predictor(args) #MemoizingPredictor(get_predictor(args)) #commenting out Memoizing predictor for eval for code review as it doesn't seem useful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.evaluate_random_baseline:
        is_distributed, replay_buffer, steps_already_done, \
          network_state, tnetwork_state, shorter_proofs_dict, random_state = \
            torch.load(str(args.rl_weights), map_location=device)
        if random_state is not None:
            random.setstate(random_state)
    else:
        replay_buffer = None
    if args.evaluate_random_baseline:
        # Giving dummy values for learning rate, batch step, and lr step because we
        # won't be training this network, so it won't matter.
        v_network = VNetwork(args, args.coq2vec_weights, predictor, device)
        target_network = VNetwork(args, args.coq2vec_weights,  predictor, device)
        target_network.obligation_encoder = v_network.obligation_encoder
    else:
        # Giving dummy values for learning rate, batch step, and lr step because we
        # won't be training this network, so it won't matter.
        v_network = VNetwork(args, None, predictor, device)
        target_network = VNetwork(args, None, predictor, device)
        target_network.obligation_encoder = v_network.obligation_encoder
        v_network.load_state(network_state)
        target_network.load_state(tnetwork_state)
    # rl.py expects these arguments for reinforcement workers, when replay
    # buffer is None (which it is for loading from distributed training), but
    # since we're not going to use it at all we can pass a dummy value
    args.window_size = 0
    args.allow_partial_batches = False
    worker = ReinforcementWorker(args, predictor, v_network, target_network, switch_dict_from_args(args),
                                 initial_replay_buffer = replay_buffer)
    
    proofs_completed = 0
    recently_done_tasks: List[RLTask] = []
    file_our_taken_dict: Dict[Path, Set[int]] = {}
    our_files_taken: Set[Path] = set()
    files_finished_this_ep: Set[Path] = set()
    skip_taken_proofs: bool = True
    while True :
        next_task_and_idx = allocate_next_task(args,
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

        
        if not tactic_prefix_is_usable(task.tactic_prefix):
            if args.verbose >= 2:
                eprint(f"Skipping job {task} with prefix {task.tactic_prefix} "
                       "because it can't purely focused")
            else:
                eprint("Skipping a job because it can't be purely focused")
        else :
            if worker.evaluate_job(task.to_job(), task.tactic_prefix):
                with (args.state_dir / f"finished-{workerid}.txt").open('a') as f,FileLock(f):
                    print(json.dumps(task.as_dict()), file=f, flush=True)


        recently_done_tasks.append(task)
        with (args.state_dir / f"progress-{workerid}.txt").open('a') as f, FileLock(f):
            print(json.dumps(task.as_dict()), file=f, flush=True)

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

if __name__ == "__main__" :
    main()
