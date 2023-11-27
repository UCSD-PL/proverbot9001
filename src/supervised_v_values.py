#!/usr/bin/env python

import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Union

import torch
import torch.cuda
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim.lr_scheduler as scheduler
from tqdm import tqdm

from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch

import coq2vec
from coq_serapy import Obligation

from rl import (VNetwork, optimizers, FileReinforcementWorker,
                VModel, prev_tactic_from_prefix)
from gen_rl_tasks import RLTask
from util import timeSince, print_time, unwrap
from search_worker import get_predictor
from models.features_polyarg_predictor import FeaturesPolyargPredictor

def main() -> None:
  argparser = argparse.ArgumentParser()
  argparser.add_argument("--prelude", default=".", type=Path)
  argparser.add_argument("-l", "--learning-rate", type=float, default=6e-5)
  argparser.add_argument("--learning-rate-decay", type=float, default=0.9)
  argparser.add_argument("--learning-rate-step", type=int, default=5)
  argparser.add_argument("--batch-size", type=int, default=16)
  argparser.add_argument("--optimizer", choices=optimizers.keys(),
                         default=list(optimizers.keys())[0])
  argparser.add_argument('--supervised-weights', type=Path, dest="weightsfile")
  argparser.add_argument("--print-final-outputs", action='store_true')
  argparser.add_argument("--gamma", type=float, default=0.7)
  argparser.add_argument("--num-epochs",type=int, default=16)
  argparser.add_argument("--hidden-size", type=int, default=128)
  argparser.add_argument("--num-layers", type=int, default=3)
  argparser.add_argument("--tactic-embedding-size", type=int, default=32)
  argparser.add_argument("--print-every", type=int, default=5)
  argparser.add_argument("-v", "--verbose", action='count', default=0)
  argparser.add_argument("--print-timings", action='store_true')
  argparser.add_argument("--encoder-weights", type=Path)
  argparser.add_argument("-o", "--output", type=Path)
  argparser.add_argument("--shorter-proofs-from", type=Path, default=None)
  argparser.add_argument("--mode", choices=["train", "tune"], default="train")
  argparser.add_argument("obl_tasks_file", type=Path)
  args = argparser.parse_args()

  args.backend = "serapi"
  args.set_switch = True

  if args.mode == "train":
    train(args)
  else:
    assert args.mode == "tune"
    tune_hyperparams(args)

def tune_hyperparams(args: argparse.Namespace) -> None:
  def objective(config: Dict[str, Union[float, int]]):
    train_args = argparse.Namespace(**vars(args))
    train_args.learning_rate = config["learning-rate"]
    train_args.learning_rate_decay = config["learning-rate-decay"]
    train_args.learning_rate_step = config["learning-rate-step"]
    train_args.gamma = config["gamma"]
    train_args.hidden_size = config["hidden-size"]
    train_args.num_layers = config["num-layers"]
    session.report({"loss": train(train_args)})
  search_space={"learning-rate": tune.loguniform(1e-12, 1e-1),
                "learning-rate-decay": tune.uniform(0.1, 1.0),
                "learning-rate-step": tune.lograndint(1, args.num_epochs // 2),
                "gamma": tune.uniform(0.2, 0.9),
                "hidden-size": tune.lograndint(64, 32768),
                "num-layers": tune.randint(1, 8)}
  algo = OptunaSearch()
  tuner = tune.Tuner(tune.with_resources(
                       tune.with_parameters(objective), {"cpu": 1, "gpu": 1}),
                     tune_config=tune.TuneConfig(
                       metric="loss",
                       mode="min",
                       search_alg=algo,
                       num_samples=256),
                     run_config=air.RunConfig(stop={"training_iteration": 5}),
                     param_space=search_space)
  results = tuner.fit()
  print("Best config is:", results.get_best_result().config)
  pass

def train(args: argparse.Namespace) -> float:
  device = "cuda" if torch.cuda.is_available() else "cpu"
  with print_time("Loading tasks"):
    with args.obl_tasks_file.open("r") as f:
      obl_tasks = [(RLTask(**task_dict), Obligation.from_dict(obl_dict))
                   for l in f for task_dict, obl_dict in (json.loads(l),)]
    tasks = [task for task, obl in obl_tasks]
    obls = [obl for task, obl in obl_tasks]
  with print_time("Building model"):
    predictor = get_predictor(args)
    assert isinstance(predictor, FeaturesPolyargPredictor)
    tactic_vocab_size = predictor.prev_tactic_vocab_size
    v_network = VNetwork(args.encoder_weights, predictor, args.learning_rate,
                         args.learning_rate_step, args.learning_rate_decay,
                         args.optimizer, tactic_vocab_size,
                         args.tactic_embedding_size, args.hidden_size,
                         args.num_layers)
    assert isinstance(v_network.network, VModel)
  #with print_time(f"Tokenizing {len(obls)} states"):
  #  tokenized_states = [v_network.obligation_encoder.obligation_to_seqs(obl)
  #                      for obl in obls]
  with print_time(f"Encoding {len(obls)} states"):
    with torch.no_grad():
      encoded_states = unwrap(v_network.obligation_encoder).\
        obligations_to_vectors_cached(obls).to(device)
    prev_tactics = [prev_tactic_from_prefix(task.tactic_prefix)
                    if len(task.tactic_prefix) > 0 else "Proof"
                    for task in tasks]
    prev_tactics_encoded = torch.LongTensor(
      [predictor.prev_tactic_stem_idx(prev_tactic)
       for prev_tactic in prev_tactics]).to(device)
  if args.shorter_proofs_from is not None:
    with args.shorter_proofs_from.open('rb') as f:
      _, _, _, _, _, shorter_proofs_dict, _ = torch.load(f, map_location="cpu")
    target_v_values = torch.FloatTensor(
      [args.gamma ** (shorter_proofs_dict[t] if t in shorter_proofs_dict 
                      else t.target_length) for t in tasks]).to(device)
    assert any(t in shorter_proofs_dict for t in tasks), \
      "Didn't find any shorter proofs for our tasks"
    for t in tasks:
      if t in shorter_proofs_dict and shorter_proofs_dict[t] < t.target_length:
        print(f"Found shorter length {shorter_proofs_dict[t]} for task {t}.")
  else:
    target_v_values = torch.FloatTensor(
      [args.gamma ** t.target_length for t in tasks]).to(device)

  dataloader = data.DataLoader(data.TensorDataset(encoded_states,
                                                  prev_tactics_encoded,
                                                  target_v_values),
                               batch_size = args.batch_size,
                               num_workers = 0, shuffle=True, drop_last=True)

  adjuster = scheduler.StepLR(v_network.optimizer, args.learning_rate_step,
                              args.learning_rate_decay)

  training_start = time.time()
  num_batches = len(encoded_states) // args.batch_size

  for epoch in range(args.num_epochs):
    print(f"Epoch {epoch} (learning rate "
          f"{v_network.optimizer.param_groups[0]['lr']:.5e})")
    epoch_loss = 0.

    for batch_num, (contexts_batch, prev_tactics_batch, target_batch) in \
        enumerate(dataloader, start=1):
      assert isinstance(target_batch, torch.Tensor), type(target_batch)
      assert target_batch.is_floating_point(), target_batch.type
      v_network.optimizer.zero_grad()
      actual = v_network.network(contexts_batch, prev_tactics_batch).view(-1)
      loss = F.mse_loss(actual, target_batch)
      loss.backward()
      v_network.optimizer.step()
      epoch_loss += (loss.item() / (num_batches * args.batch_size))

      if batch_num % args.print_every == 0:
        items_processed = batch_num * args.batch_size + \
          epoch * len(encoded_states)
        progress = items_processed / (len(encoded_states) * args.num_epochs)
        print(f"{timeSince(training_start, progress)} "
              f"({items_processed:7} {progress * 100:5.2f}%) "
              f"{epoch_loss / batch_num:.8f}")
    adjuster.step()

  with args.output.open('wb') as f:
    torch.save((False, None, 0, v_network.get_state(), v_network.get_state(),
                {}, None), f)

  threshold = 0.1
  erroneous_count = 0
  if args.print_final_outputs:
    actual = v_network.network(encoded_states, prev_tactics_encoded).view(-1)
    for actual_value, expected_value in zip(actual, target_v_values):
      if abs(actual_value - expected_value) > threshold:
        erroneous_count += 1
        status_string = "(Error)"
      else:
        status_string = ""
      print(actual_value.item(), "vs", expected_value.item(), status_string)
    print(f"{erroneous_count} of {len(encoded_states)} samples have the wrong v value")
  return epoch_loss

if __name__ == "__main__":
  main()
