#!/usr/bin/env python

import argparse
import time
import json
from pathlib import Path
from typing import List

import torch
import torch.cuda
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim.lr_scheduler as scheduler
from tqdm import tqdm

import coq2vec
from coq_serapy import Obligation

from rl import VNetwork, optimizers, FileReinforcementWorker
from gen_rl_tasks import RLTask
from util import timeSince, print_time

def main() -> None:
  argparser = argparse.ArgumentParser()
  argparser.add_argument("--prelude", default=".", type=Path)
  argparser.add_argument("-l", "--learning-rate", type=float, default=6e-5)
  argparser.add_argument("--learning-rate-decay", type=float, default=0.9)
  argparser.add_argument("--learning-rate-step", type=int, default=5)
  argparser.add_argument("--batch-size", type=int, default=16)
  argparser.add_argument("--optimizer", choices=optimizers.keys(),
                         default=list(optimizers.keys())[0])
  argparser.add_argument("--print-final-outputs", action='store_true')
  argparser.add_argument("--gamma", type=float, default=0.7)
  argparser.add_argument("--num-epochs",type=int, default=16)
  argparser.add_argument("--hidden-size", type=int, default=128)
  argparser.add_argument("--num-layers", type=int, default=3)
  argparser.add_argument("--print-every", type=int, default=5)
  argparser.add_argument("-v", "--verbose", action='count', default=0)
  argparser.add_argument("--print-timings", action='store_true')
  argparser.add_argument("--encoder-weights", type=Path)
  argparser.add_argument("-o", "--output", type=Path)
  argparser.add_argument("obl_tasks_file", type=Path)
  args = argparser.parse_args()

  args.backend = "serapi"
  args.set_switch = True

  train(args)

def train(args: argparse.Namespace) -> None:
  device = "cuda" if torch.cuda.is_available() else "cpu"
  with print_time("Loading tasks"):
    with args.obl_tasks_file.open("r") as f:
      obl_tasks = [(RLTask(**task_dict), Obligation.from_dict(obl_dict))
                   for l in f for task_dict, obl_dict in (json.loads(l),)]
    tasks = [task for task, obl in obl_tasks]
    obls = [obl for task, obl in obl_tasks]
  with print_time("Building model"):
    v_network = VNetwork(args.encoder_weights, args.learning_rate,
                         args.learning_rate_step, args.learning_rate_decay,
                         args.optimizer, args.hidden_size, args.num_layers)
  #with print_time(f"Tokenizing {len(obls)} states"):
  #  tokenized_states = [v_network.obligation_encoder.obligation_to_seqs(obl)
  #                      for obl in obls]
  with print_time(f"Encoding {len(obls)} states"):
    with torch.no_grad():
      encoded_states = v_network.obligation_encoder.\
        obligations_to_vectors_cached(obls).to(device)
  target_v_values = torch.FloatTensor(
    [args.gamma ** t.target_length for t in tasks]).to(device)

  dataloader = data.DataLoader(data.TensorDataset(encoded_states, 
                                                  target_v_values),
                               batch_size = args.batch_size,
                               num_workers = 0, shuffle=True, drop_last=True)

  adjuster = scheduler.StepLR(v_network.optimizer, args.learning_rate_step,
                              args.learning_rate_decay)

  training_start = time.time()
  num_batches = len(encoded_states) // args.batch_size

  for epoch in range(args.num_epochs):
    print(f"Epoch {epoch} (learning rate "
          f"{v_network.optimizer.param_groups[0]['lr']:.6f})")
    epoch_loss = 0.

    for batch_num, (input_batch, target_batch) in \
        enumerate(dataloader, start=1):
      v_network.optimizer.zero_grad()
      actual = v_network.network(input_batch).view(-1)
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
    actual = v_network.network(encoded_states).view(-1)
    for actual_value, expected_value in zip(actual, target_v_values):
      if abs(actual_value - expected_value) > threshold:
        erroneous_count += 1
      print(actual_value.item(), "vs", expected_value.item())
    print(f"{erroneous_count} of {len(encoded_states)} samples have the wrong v value")

if __name__ == "__main__":
  main()
