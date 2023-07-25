#!/usr/bin/env python3

import argparse
import sys
import os
import time
from pathlib import Path
from typing import List

import torch

sys.path.append(str(Path(os.getcwd()) / "src"))

import rl
from util import eprint, print_time
from distributed_rl import add_distrl_args_to_parser


def main():
    parser = argparse.ArgumentParser()
    rl.add_args_to_parser(parser)
    add_distrl_args_to_parser(parser)
    args = parser.parse_args()

    sync_worker_target_networks(args)

def sync_worker_target_networks(args: argparse.Namespace) -> None:
    next_save_num = 1
    while True:
        try:
            worker_weights = load_worker_weights(args, next_save_num)
        except FileNotFoundError:
            eprint(f"At {time.ctime()}, waiting for weights of save number {next_save_num}")
            time.sleep(0.25)
            continue
        with print_time("Averaging weights"):
            result_params = {}
            for key in worker_weights[0]:
                result_params[key] = sum((weights_dict[key] for weights_dict
                                          in worker_weights)) / args.num_workers
        eprint("Saving weights")
        with print_time("Saving weights"):
            torch.save(result_params,
                       args.state_dir / "weights" / f"common-target-network-{next_save_num}.dat")
            delete_worker_weights(args, next_save_num)
        next_save_num += 1

def load_worker_weights(args: argparse.Namespace, save_num: int) -> List[dict]:
    worker_weights = []
    for workerid in range(args.num_workers):
        worker_save = torch.load(str(
            args.state_dir / "weights" / f"worker-{workerid}-network-{save_num}.dat"))
        replay_buffer, step, weights_dict, random_state = worker_save
        worker_weights.append(weights_dict[0])
    return worker_weights
def delete_worker_weights(args: argparse.Namespace, save_num: int) -> None:
    for workerid in range(args.num_workers):
        (args.state_dir / "weights" / f"worker-{workerid}-network-{save_num}.dat").unlink()

if __name__ == "__main__":
    main()
