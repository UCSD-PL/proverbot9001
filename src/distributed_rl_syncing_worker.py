#!/usr/bin/env python3

import argparse
import sys
import os
import time
from pathlib import Path

import torch

import rl
from util import eprint
from distributed_rl import add_distrl_args_to_parser

sys.path.append(str(Path(os.getcwd()) / "src"))

def main():
    parser = argparse.ArgumentParser()
    rl.add_args_to_parser(parser)
    add_distrl_args_to_parser(parser)
    args = parser.parse_args()

    sync_worker_target_networks(args)

def sync_worker_target_networks(args: argparse.Namespace) -> None:
    while True:
        worker_weights = [torch.load(str(args.state_dir / f"worker-{workerid}-network.dat"))
                          for workerid in range(args.num_workers)]
        result_params = {}
        for key in worker_weights[0]:
            result_params[key] = sum((weights_dict[key] for weights_dict
                                      in worker_weights)) / args.num_workers
        torch.save(result_params, args.state_dir / "common-target-network.dat")
        time.sleep(10)
        eprint("Saving weights")

if __name__ == "__main__":
    main()
