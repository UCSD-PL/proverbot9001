#!/usr/bin/env python
import argparse
import json
import random
import time
import contextlib
import math
from pathlib import Path
from operator import itemgetter
from typing import (
    List,
    Optional,
    Dict,
    Tuple,
    Union,
    Any,
    Set,
    Sequence,
    TypeVar,
    Callable,
)

from util import unwrap, eprint, print_time, nostderr
import subprocess
with print_time("Importing torch"):
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch import optim
    import torch.optim.lr_scheduler as scheduler
    from models.tactic_predictor import TacticPredictor, Prediction

from tqdm import tqdm

import coq_serapy
from coq_serapy.contexts import (
    FullContext,
    truncate_tactic_context,
    Obligation,
    TacticContext,
    ProofContext,
)
import coq2vec
import os
with print_time("Importing search code"):
    from search_file import get_all_jobs
    from search_worker import ReportJob, Worker, get_predictor
    from search_strategies import completed_proof

from util import unwrap, eprint, print_time, nostderr
from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
RUN_ROOT='/home/shizhuo2_illinois_edu/proverbot9001/src'

def main():
    eprint("Starting main")
    parser = argparse.ArgumentParser(
        description="Train a state estimator using reinforcement learning"
        "to complete proofs using Proverbot9001."
    )
    parser.add_argument("--prelude", default=".", type=Path)
    parser.add_argument(
        "--output_file","-o",type=Path
    )
    parser.add_argument(
        "--verbose", "-v", help="verbose output", action="count", default=0
    )
    parser.add_argument(
        "--progress", "-P", help="show progress of files", action="store_true"
    )
    parser.add_argument("--print_timings", "-t", action="store_true")
    parser.add_argument("--set_switch",  action="store_true")
    parser.add_argument("--include_proof_relevant", action="store_true")
    parser.add_argument("--backend", choices=["serapi", "lsp", "auto"], default="auto")
    parser.add_argument("--filenames", help="proof file name (*.v)", nargs="+", type=Path)
    parser.add_argument("--splits_file",type=str)
    proofsGroup = parser.add_mutually_exclusive_group()
    proofsGroup.add_argument("--proof", default=None)
    proofsGroup.add_argument("--proofs_file", default=None)
    parser.add_argument("--tasks_file", default=None)
    parser.add_argument("--test_file", default=None)
    parser.add_argument("--interleave",  action="store_true")
    parser.add_argument("--weightsfile", type=Path)
    parser.add_argument("--coq2vec_weights", type=Path)
    parser.add_argument("--max_sertop_workers", default=16, type=int)
    parser.add_argument("-l", "--learning_rate", default=2.5e-4, type=float)
    parser.add_argument("-g", "--gamma", default=0.9, type=float)
    parser.add_argument("--starting_epsilon", default=0, type=float)
    parser.add_argument("--ending_epsilon", default=1.0, type=float)
    parser.add_argument("-s", "--steps_per_episode", default=16, type=int)
    parser.add_argument("-n", "--num_episodes", default=1, type=int)
    parser.add_argument("-b", "--batch_size", default=64, type=int)
    parser.add_argument("-w", "--window_size", default=2560)
    parser.add_argument("-p", "--num_predictions", default=5, type=int)
    parser.add_argument("--batch_step", default=25, type=int)
    parser.add_argument("--lr_step", default=0.8, type=float)
    parser.add_argument("--batches_per_proof", default=1, type=int)
    parser.add_argument("--train_every", default=1, type=int)
    parser.add_argument("--print_loss_every", default=None, type=int)
    parser.add_argument(
        "--sync_target_every",
        help="Sync target network to v network every <n> episodes",
        default=-1,
        type=int,
    )
    parser.add_argument("--allow_partial_batches", action="store_true")
    parser.add_argument(
        "--blacklisted_tactics", action="append"
    )
    parser.add_argument("--resume", choices=["no", "yes", "ask"], default="ask")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--log_root", type=Path, default=Path("logs"))
    parser.add_argument("--result_root", type=Path, default=Path("result"))
    parser.add_argument("--exp_name", type=Path, default=Path("search"))
    parser.add_argument(
        "--num_trails",
        default=50,
        type=int,
        help="Number of trails for hyperparameter search",
    )
    parser.add_argument(
        "--num_cpus",
        default=1,
        type=int,
        help="Number of CPUs available for a hyperparameter search",
    )
    parser.add_argument(
        "--num_gpus",
        default=0,
        type=int,
        help="Number of GPUs available for a hyperparameter search",
    )
    args = parser.parse_args()
    ## commented temporarily
    # if args.filenames[0].suffix == ".json":
    #     args.splits_file = args.filenames[0]
    #     args.filenames = None
    # else:
    #     args.splits_file = None
    args.splits_file = None
    tuning(args)

def config_fnames(logdir: str = 'logs'):
    # Check if output.log and error.log exist
    new_output_name = 'output1.log'
    index = 1
    if os.path.exists(os.path.join(logdir,'output1.log')):
        # Find an available name for the new output file
        index = 2
        while os.path.exists(os.path.join(logdir,f'output{index}.log')):
            index += 1
        new_output_name = f'output{index}.log'
    new_error_name = f'error{index}.log'
    bash_file = f'run{index}.sh'
    return new_output_name, new_error_name,bash_file
def tuning(args) -> None:
    """
    This function is used for tuning hyperparameters with Ray Tune.
    """

    def objective(config, args):
        """
        This function is passed to Ray Tune to optimize hyperparameters.
        config: A dictionary of hyperparameters to optimize from ray tune.
        args: The arguments passed to the main function.
        """

        setattr(args, "gamma", config["gamma"])
        setattr(args, "starting_epsilon", config["starting_epsilon"])
        setattr(args, "batch_step", config["batch_step"])
        setattr(args, "lr_step", config["lr_step"])
        setattr(args, "batches_per_proof", config["batches_per_proof"])
        setattr(args, "sync_target_every", config["sync_target_every"])
        setattr(args, "num_cpus", 1)
        setattr(args, "num_gpus", 0)
        
        logdir = os.path.join(args.log_root, args.exp_name)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        resultdir = os.path.join(args.result_root, args.exp_name)
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)
        log_fname, err_fname,bash_fname = config_fnames(logdir)
        script_dir = os.path.join(logdir, bash_fname)
        # Add the necessary arguments to the sbatch command
        sbatch_command = []
        sbatch_command.extend([
            '#!/bin/bash',          # Specify the interpreter to be bash
            f'#SBATCH --job-name={args.exp_name}',  # Specify the job name
            '#SBATCH --ntasks=1',  # Specify the number of tasks (processes) to run, default is 1
            '#SBATCH --time=0-6:00:00',  # Specify the time limit for the job
            f'#SBATCH --output={logdir}/{log_fname}',  # Specify the output log file
            f'#SBATCH --error={logdir}/{err_fname}',    # Specify the error log file
            f'python {RUN_ROOT}/rl_tuning_distributed_worker.py \\',               # Specify the command to run (Python interpreter)          # Specify the Python script to run
        ])

        # Add the config key-value pairs as command-line arguments for run.py
        for key, value in vars(args).items():
            if isinstance(value, bool):  # Check if the argument value is a boolean
                if value:
                    sbatch_command.append(f"    --{key} \\")  # Include the flag if it's True
            elif value:
                if key == 'num_gpus' or key == 'num_cpus':
                    sbatch_command.append(f"    --{key}=1 \\")  # Include other arguments with values
                else:
                    sbatch_command.append(f"    --{key}={value} \\")  # Include other arguments with values

        with open(script_dir, 'w') as script_file:
            script_file.write('\n'.join(sbatch_command))
            print(script_dir)

        # Execute the sbatch command
        subprocess.run(['sbatch', script_dir])
        



    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "gamma": tune.uniform(0.1, 0.99),
        "starting_epsilon": tune.uniform(0, 1),
        "batch_step": tune.randint(1, 30),
        "lr_step": tune.uniform(0.8, 1),
        "batches_per_proof": tune.randint(10, 50),
        "sync_target_every": tune.randint(1, 250),
    } # Define the search space for hyperparameters
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(objective, args=args),
            {"cpu": args.num_cpus, "gpu": args.num_gpus},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(num_samples=args.num_trails),
    )

    tuner.fit()


if __name__ == "__main__":
    main()
