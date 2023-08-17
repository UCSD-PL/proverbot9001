import argparse
import json
import random
import torch
import os
from typing import (List, Optional, Dict, Tuple, Union, Any, Set,
                    Sequence, TypeVar, Callable)

from util import eprint, FileLock
from pathlib import Path
from search_worker import get_predictor
from search_file import get_all_jobs
from gen_rl_tasks import RLTask
from rl import ReinforcementWorker, MemoizingPredictor, switch_dict_from_args, VNetwork, ReplayBuffer
from distributed_rl_learning_worker import TaskEpisode, get_all_task_episodes, allocate_next_task_from_file_with_retry


def main():
    eprint("Starting main")
    parser = argparse.ArgumentParser(
        description="Evaluation worker - The rl agent"
        "to complete proofs using Proverbot9001.")
    parser.add_argument("--prelude", default=".", type=Path)
    parser.add_argument("--output", "-o", dest="output_file",
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
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--no-interleave", dest="interleave", action="store_false")
    parser.add_argument('--supervised-weights', type=Path, dest="weightsfile")
    parser.add_argument("--coq2vec-weights", type=Path)
    parser.add_argument("--max-sertop-workers", default=16, type=int)
    parser.add_argument("-l", "--learning-rate", default=2.5e-4, type=float)
    parser.add_argument("-g", "--gamma", default=0.9, type=float)
    parser.add_argument("--starting-epsilon", default=0, type=float)
    parser.add_argument("--ending-epsilon", default=1.0, type=float)
    parser.add_argument("-s", "--steps-per-episode", default=16, type=int)
    parser.add_argument("-n", "--num-episodes", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-w", "--window-size", default=2560)
    parser.add_argument("-p", "--num-predictions", default=5, type=int)
    parser.add_argument("--batch-step", default=25, type=int)
    parser.add_argument("--lr-step", default=0.8, type=float)
    parser.add_argument("--batches-per-proof", default=1, type=int)
    parser.add_argument("--train-every", default=1, type=int)
    parser.add_argument("--print-loss-every", default=None, type=int)
    parser.add_argument("--curriculum",action="store_true")
    parser.add_argument("--sync-target-every",
                        help="Sync target network to v network every <n> episodes",
                        default=10, type=int)
    parser.add_argument("--allow-partial-batches", action='store_true')
    parser.add_argument("--blacklist-tactic", action="append",
                        dest="blacklisted_tactics")
    parser.add_argument("--resume", choices=["no", "yes", "ask"], default="ask")
    parser.add_argument("--num-eval-workers", type=int, default=5)
    evalGroup = parser.add_mutually_exclusive_group()
    evalGroup.add_argument("--evaluate", action="store_true")
    evalGroup.add_argument("--evaluate-baseline", action="store_true")
    parser.add_argument("--state_dir", default="drl_state/eval", type=Path)
    
    
    args = parser.parse_args()

    
    (args.state_dir).mkdir(exist_ok=True, parents=True)

    if args.filenames[0].suffix == ".json":
        args.splits_file = args.filenames[0]
        args.filenames = None
    else:
        args.splits_file = None

    workerid = int(os.environ['SLURM_ARRAY_TASK_ID'])
    jobid = int(os.environ['SLURM_ARRAY_JOB_ID'])
    # workerid = 1
    with (args.state_dir / "taken.txt").open("w") as f:
        pass


    eprint(f"Starting worker {workerid}")
    evaluation_worker(args,workerid,jobid)
    eprint(f"Finished worker {workerid}")




def evaluation_worker(args: argparse.Namespace, workerid, jobid) :
            
    task_episodes = get_all_task_episodes(args)
    file_all_tes_dict: Dict[Path, List[Tuple[int, TaskEpisode]]] = {}
    for task_ep_idx, (task, episode) in enumerate(task_episodes):
        if Path(task.src_file) in file_all_tes_dict:
            file_all_tes_dict[Path(task.src_file)].append((task_ep_idx, (task, episode)))
        else:
            file_all_tes_dict[Path(task.src_file)] = [(task_ep_idx, (task, episode))]

    predictor = MemoizingPredictor(get_predictor(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replay_buffer, steps_already_done, network_state, tnetwork_state, random_state = \
        torch.load(str(args.output_file), map_location=device)
    random.setstate(random_state)
    v_network = VNetwork(None, args.learning_rate,
                            args.batch_step, args.lr_step)
    target_network = VNetwork(None, args.learning_rate,
                                args.batch_step, args.lr_step)
    target_network.obligation_encoder = v_network.obligation_encoder
    v_network.load_state(network_state)
    target_network.load_state(tnetwork_state)
    worker = ReinforcementWorker(args, predictor, v_network, target_network, switch_dict_from_args(args),
                                 initial_replay_buffer = replay_buffer)
    
    proofs_completed = 0
    recently_done_task_eps: List[TaskEpisode] = []
    file_our_taken_dict: Dict[Path, Set[int]] = {}
    our_files_taken: Set[Path] = set()
    max_episode: int = 0
    files_finished_this_ep: Set[Path] = set()
    while True :
        with (args.state_dir / f"taken.txt").open("r+") as f, FileLock(f):
            taken_task_episodes = [((RLTask(**task_dict), episode), taken_this_iter)
                                   for line in f
                                   for (task_dict, episode), taken_this_iter in (json.loads(line),)]
            # current_task_episode = get_next_task(task_episodes, taken_task_episodes,
            #                                      our_taken_task_eps)
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
        with (args.state_dir / f"taken-{workerid}.txt").open('a') as f:
            print(json.dumps((task.as_dict(), episode)), file=f, flush=True)
        src_path = Path(task.src_file)
        if src_path in file_our_taken_dict:
            file_our_taken_dict[src_path].add(next_task_idx)
        else:
            file_our_taken_dict[src_path] = {next_task_idx}


        if worker.evaluate_job(task.to_job(), task.tactic_prefix):
            with (args.state_dir / f"finished-{workerid}.txt").open('a') as f:
                print(json.dumps((task.as_dict(), episode)), file=f, flush=True)
        
        recently_done_task_eps.append((task, episode))

    return
    





if __name__ == "__main__" :
    main()