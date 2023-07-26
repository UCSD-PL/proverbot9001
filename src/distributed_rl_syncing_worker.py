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
from util import eprint, print_time, unwrap
from distributed_rl import add_distrl_args_to_parser
#pylint: enable=wrong-import-position

def main():
    parser = argparse.ArgumentParser()
    rl.add_args_to_parser(parser)
    add_distrl_args_to_parser(parser)
    args = parser.parse_args()

    sync_worker_target_networks(args)

def sync_worker_target_networks(args: argparse.Namespace) -> None:
    retry_delay_secs = 1
    next_save_num = 0
    last_weights_versions: List[Tuple[int, int]] = []
    while True:
        worker_weights_versions = get_latest_worker_weights_versions(args)
        if worker_weights_versions == last_weights_versions:
            time.sleep(retry_delay_secs)
            continue
        worker_weights = load_worker_weights(args, worker_weights_versions)
        result_params = {}
        for key in worker_weights[0]:
            result_params[key] = sum((weights_dict[key] for weights_dict
                                      in worker_weights)) / len(worker_weights)
        eprint(f"Saving weights with versinos {worker_weights_versions}")
        with print_time("Saving weights"):
            torch.save(result_params,
                       args.state_dir / "weights" / f"common-target-network-{next_save_num}.dat")
            delete_old_worker_weights(args, worker_weights_versions)
        next_save_num += 1

def get_latest_worker_weights_versions(args: argparse.Namespace) \
        -> List[Tuple[int, int]]:
    save_nums = []
    for workerid in range(args.num_workers):
        worker_networks = glob(f"worker-{workerid}-network-*.dat",
                               root_dir = str(args.state_dir / "weights"))
        if len(worker_networks) == 0:
            continue
        latest_worker_save_num = max(
            int(unwrap(re.match(rf"worker-{workerid}-network-(\d+).dat",
                                path)).group(1))
            for path in worker_networks)
        save_nums.append((workerid, latest_worker_save_num))
    return save_nums

def load_worker_weights(args: argparse.Namespace,
                        worker_weights_versions: List[Tuple[int, int]]) \
        -> List[dict]:
    worker_weights = []
    for workerid, save_num in worker_weights_versions:
        latest_worker_save = torch.load(
            args.state_dir / "weights" /
            f"worker-{workerid}-network-{save_num}.dat")
        _replay_buffer, _step, (weights, _obl_encoder_state, _obl_enccoder_cache), \
            _random_state = latest_worker_save
        worker_weights.append(weights)

    return worker_weights

def delete_old_worker_weights(args: argparse.Namespace,
                              cur_versions: List[Tuple[int, int]]) -> None:
    for workerid, save_num in cur_versions:
        if save_num == 1:
            continue
        (args.state_dir / "weights" /
         f"worker-{workerid}-network-{save_num - 1}.dat").unlink()

if __name__ == "__main__":
    main()
