#!/usr/bin/env python3

import argparse
import sys
import os
import time
import re
from pathlib import Path
from typing import List, Tuple

from glob import glob

import torch

sys.path.append(str(Path(os.getcwd()) / "src"))

#pylint: disable=wrong-import-position
import rl
from util import eprint, unwrap
from distributed_rl import (add_distrl_args_to_parser,
                            get_all_task_eps, get_num_task_eps_done,
                            latest_worker_save_num)
#pylint: enable=wrong-import-position

def main():
    print("running main?")
    parser = argparse.ArgumentParser()
    rl.add_args_to_parser(parser)
    add_distrl_args_to_parser(parser)
    args = parser.parse_args()

    sync_worker_target_networks(args)

def sync_worker_target_networks(args: argparse.Namespace) -> None:
    print("running sync worker")
    retry_delay_secs = 0.5
    next_save_num = get_resumed_save_num(args) + 1
    last_weights_versions: List[Tuple[int, int]] = []
    all_task_eps = get_all_task_eps(args)
    while True:
        num_task_eps_done = get_num_task_eps_done(args)
        worker_weights_versions = get_latest_worker_weights_versions(args)
        if worker_weights_versions == last_weights_versions:
            time.sleep(retry_delay_secs)
            continue
        worker_weights = load_worker_weights(args, worker_weights_versions)
        result_params = {}
        for key in worker_weights[0]:
            result_params[key] = sum((weights_dict[key] for weights_dict
                                      in worker_weights)) / len(worker_weights)
        eprint(f"Saving weights with versions {worker_weights_versions} "
               f"to save num {next_save_num}")
        save_path = str(args.state_dir / "weights" /
                        f"common-target-network-{next_save_num}.dat")
        torch.save(result_params, save_path + ".tmp")
        os.rename(save_path + ".tmp", save_path)
        delete_old_worker_weights(args, worker_weights_versions)
        delete_old_common_weights(args)
        next_save_num += 1
        if num_task_eps_done >= len(all_task_eps):
            break
    print("no reason to quit before this line")
    eprint("Saving final weights and cleaning up")
    os.rename(save_path, args.output_file)

def get_latest_worker_weights_versions(args: argparse.Namespace) \
        -> List[Tuple[int, int]]:
    save_nums = []
    for workerid in range(args.num_workers):
        save_num = latest_worker_save_num(args, workerid)
        if save_num is not None:
            save_nums.append((workerid, save_num))
    return save_nums

def load_worker_weights(args: argparse.Namespace,
                        worker_weights_versions: List[Tuple[int, int]]) \
        -> List[dict]:
    worker_weights = []
    for workerid, save_num in worker_weights_versions:
        latest_worker_save_path = (
            args.state_dir / "weights" /
            f"worker-{workerid}-network-{save_num}.dat")
        eprint(f"Loading worker weights from {latest_worker_save_path}")
        for i in range(3):
            try:
                latest_worker_save = torch.load(latest_worker_save_path)
                break
            except OSError:
                eprint("Failed to load worker file because of stale handle, trying again")
        _replay_buffer, _step, (weights, _obl_encoder_state, _obl_enccoder_cache), \
            _random_state = latest_worker_save
        worker_weights.append(weights)

    return worker_weights

def delete_old_common_weights(args: argparse.Namespace) -> None:
    current_working_directory = os.getcwd()
    root_dir = str(args.state_dir / "weights")
    os.chdir(root_dir)
    
    common_network_paths= glob(f"common-target-network-*.dat")
    os.chdir(current_working_directory)

    #common_network_paths= glob(f"common-target-network-*.dat",
    #                            root_dir = str(args.state_dir / "weights"))
    common_save_nums = [int(unwrap(re.match(rf"common-target-network-(\d+).dat",
                                            path)).group(1))
                        for path in common_network_paths]
    latest_worker_save_num = max(common_save_nums)
    for save_num in common_save_nums:
        if save_num > latest_worker_save_num - args.keep_latest:
            continue
        old_save_path = (args.state_dir / "weights" /
                         f"common-target-network-{save_num}.dat")
        old_save_path.unlink()

def delete_old_worker_weights(args: argparse.Namespace,
                              cur_versions: List[Tuple[int, int]]) -> None:
    for workerid, latest_save_num in cur_versions:
        if latest_save_num == 1:
            continue
        current_working_directory = os.getcwd()
        root_dir = str(args.state_dir / "weights")
        os.chdir(root_dir)
        
        worker_network_paths= glob(f"worker-{workerid}-network-*.dat")
        os.chdir(current_working_directory)
        #worker_network_paths = glob(f"worker-{workerid}-network-*.dat",
        #                            root_dir = str(args.state_dir / "weights"))
        worker_save_nums = [int(unwrap(re.match(rf"worker-{workerid}-network-(\d+).dat",
                                                path)).group(1))
                            for path in worker_network_paths]
        old_worker_save_nums = [save_num for save_num in worker_save_nums
                                if save_num != latest_save_num]
        for save_num in old_worker_save_nums:
            old_save_path = (args.state_dir / "weights" /
                             f"worker-{workerid}-network-{save_num}.dat")
            old_save_path.unlink()
def get_resumed_save_num(args: argparse.Namespace) -> int:
    cwd = os.getcwd()
    os.chdir(str(args.state_dir / "weights"))
    common_networks = glob("common-target-network-*.dat")
    os.chdir(cwd)
    if len(common_networks) == 0:
        return 0

    return max(int(unwrap(re.match(
               rf"common-target-network-(\d+).dat",
               path)).group(1))
               for path in common_networks)

if __name__ == "__main__":
    main()
