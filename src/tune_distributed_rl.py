#!/usr/bin/env python
import argparse
import math
import os
from pathlib import Path
from typing import Dict, Union

from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch

from rl import optimizers
from distributed_rl import distributed_rl
from util import eprint

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-actors", default=8, type=int)
  parser.add_argument("--workers-output-dir", default=Path("output"),
                      type=Path)
  parser.add_argument("--worker-timeout", default="6:00:00")
  parser.add_argument("--partition", default="gpu")
  parser.add_argument("--mem", default="2G")
  parser.add_argument("--state-dir", default="drl_state", type=Path)
  parser.add_argument("--keep-latest", default=3, type=int)
  parser.add_argument("--loss-smoothing", type=int, default=1)
  parser.add_argument("--no-reset-on-sync", action='store_false', dest='reset_on_sync')
  parser.add_argument("--loss", choices=["simple", "log"],
                      default="simple")
  parser.add_argument("--prelude", default=".", type=Path)
  parser.add_argument("--output", "-o", dest="output_file",
                      help="output data folder name",
                      default="data/rl_weights.dat",
                      type=Path)
  parser.add_argument("--verbose", "-v", help="verbose output",
                      action="count", default=0)
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
  parser.add_argument("-s", "--steps-per-episode", default=16, type=int)
  parser.add_argument("-n", "--num-episodes", default=1, type=int)
  parser.add_argument("-b", "--batch-size", default=64, type=int)
  parser.add_argument("-p", "--num-predictions", default=5, type=int)
  parser.add_argument("--allow-partial-batches", action='store_true')
  parser.add_argument("--optimizer", choices=optimizers.keys(),
                      default=list(optimizers.keys())[0])
  parser.add_argument("--start-from", type=Path, default=None)
  parser.add_argument("--print-loss-every", default=None, type=int)
  args = parser.parse_args()
  args.ignore_after = None
  args.dump_negative_examples = None
  args.dump_replay_buffer = None
  args.progress = False
  args.print_timings = False
  args.blacklisted_tactics = []
  args.print_loss_every = None
  args.resume = "no"
  args.save_every = 100
  args.evaluate = False
  args.evaluate_baseline = False
  args.evaluate_random_baseline = False
  args.curriculum = False
  args.verifyv_every = 100 # This is need to get the validation error to tune on.
  args.verifyv_steps = False
  args.verifyvval = False
  args.train_every = 1
  args.catch_interrupts = False
  args.progress = False
  args.decrease_lr_on_reset = True
  tune_params(args)

def tune_params(args: argparse.Namespace) -> None:
  cwd = os.getcwd()
  def objective(config: Dict[str, Union[float, int]]):
    run_dir = os.getcwd()
    os.chdir(cwd)
    train_args = argparse.Namespace(**vars(args))
    train_args.sync_workers_every = config["sync-workers-every"]
    train_args.learning_rate = config["learning-rate"]
    train_args.learning_rate_step = config["learning-rate-step"]
    train_args.learning_rate_decay = config["learning-rate-decay"]
    train_args.gamma = config["gamma"]
    train_args.starting_epsilon = config["starting-epsilon"]
    train_args.ending_epsilon = config["ending-epsilon"]
    train_args.smoothing_factor = config["smoothing-factor"]
    train_args.window_size = config["window-size"]
    train_args.hidden_size = int(math.sqrt(config["internal-connections"] / (config["num-layers"])))
    train_args.tactic_embedding_size = config["tactic-embedding-size"]
    train_args.num_layers = config["num-layers"]
    train_args.sync_target_every = config["sync-target-every"]
    train_args.state_dir = Path(run_dir) / args.state_dir

    distributed_rl(train_args)
    with open(Path(run_dir) / args.state_dir / "latest_error.txt", "r") as f:
      error = float(f.read())
    session.report({"vvalue_error": error})

  search_space={
                "sync-workers-every": tune.lograndint(1, 1000),
                "learning-rate": tune.loguniform(1e-7, 1e-1),
                "learning-rate-decay": tune.uniform(0.1, 1.0),
                "learning-rate-step": tune.lograndint(1, 1000),
                "gamma": tune.choice([0.7]),
                "starting-epsilon": tune.choice([0.1]),
                "ending-epsilon": tune.choice([1.0]),
                "smoothing-factor": tune.choice([1]),
                "window-size": tune.choice([2560]),
                "internal-connections":
                  tune.lograndint(1024, 645500000),
                "tactic-embedding-size": tune.choice([32]),
                "num-layers": tune.randint(1, 10), 
                "sync-target-every": tune.lograndint(1, 1000),
                }
  algo = OptunaSearch()
  tuner = tune.Tuner(tune.with_parameters(objective),
                     tune_config=tune.TuneConfig(
                       metric="vvalue_error",
                       mode="min",
                       search_alg=algo,
                       max_concurrent_trials=3,
                       num_samples=256),
                     param_space=search_space)
  results = tuner.fit()
  print("Best config is:", results.get_best_result().config)

if __name__ == "__main__":
  main()
