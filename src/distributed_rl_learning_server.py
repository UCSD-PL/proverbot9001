#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sys
import random
import math
import subprocess
import re
from glob import glob

from threading import Thread, Lock, Event
from pathlib import Path
from typing import List, Set, Optional, Dict, Tuple, Sequence
from socket import gethostname

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import torch.distributed as dist

# pylint: disable=wrong-import-position
sys.path.append(str(Path(os.getcwd()) / "src"))
from rl import model_setup
from util import eprint, unwrap
# pylint: enable=wrong-import-position

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--state-dir", type=Path, default="drl_state")
  parser.add_argument("-e", "--encoding-size", type=int, required=True)
  parser.add_argument("-l", "--learning-rate", default=5e-6, type=float)
  parser.add_argument("-b", "--batch-size", default=64, type=int)
  parser.add_argument("-g", "--gamma", default=0.9, type=float)
  parser.add_argument("--allow-partial-batches", action='store_true')
  parser.add_argument("--window-size", type=int, default=2560)
  parser.add_argument("--train-every", type=int, default=8)
  parser.add_argument("--sync-target-every", type=int, default=32)
  parser.add_argument("--keep-latest", default=3, type=int)
  args = parser.parse_args()

  serve_parameters(args)

def serve_parameters(args: argparse.Namespace, backend='mpi') -> None:
  dist.init_process_group(backend)
  eprint(f"Connection established")
  v_network: nn.Module = model_setup(args.encoding_size)
  target_network: nn.Module = model_setup(args.encoding_size)
  target_network.load_state_dict(v_network.state_dict())
  optimizer: optim.Optimizer = optim.RMSprop(v_network.parameters(),
                                             lr=args.learning_rate)
  replay_buffer = EncodedReplayBuffer(args.window_size,
                                      args.allow_partial_batches)
  signal_change = Event()
  buffer_thread = BufferPopulatingThread(replay_buffer, signal_change, 
                                         args.encoding_size)
  buffer_thread.start()

  steps_last_trained = 0
  steps_last_synced = 0
  common_network_version = 0

  while signal_change.wait():
    signal_change.clear()
    if replay_buffer.buffer_steps - steps_last_trained >= args.train_every:
      steps_last_trained = replay_buffer.buffer_steps
      train(args, v_network, target_network, optimizer, replay_buffer)
      send_new_weights(args, v_network, common_network_version)
      common_network_version += 1
    if replay_buffer.buffer_steps - steps_last_synced >= args.sync_target_every:
      eprint("Syncing target network")
      steps_last_synced = replay_buffer.buffer_steps
      target_network.load_state_dict(v_network.state_dict())

def train(args: argparse.Namespace, v_model: nn.Module,
          target_model: nn.Module,
          optimizer: optim.Optimizer,
          replay_buffer: EncodedReplayBuffer) -> None:
  samples = replay_buffer.sample(args.batch_size)
  if samples is None:
    return
  eprint(f"Got {len(samples)} samples to train")
  inputs = torch.cat([start_obl.view(1, args.encoding_size)
                      for start_obl, _action_records in samples], dim=0)
  num_resulting_obls = [[len(resulting_obls)
                         for _action, resulting_obls in action_records]
                        for _start_obl, action_records in samples]
  all_resulting_obls = [obl for _start_obl, action_records in samples
                        for _action, resulting_obls in action_records
                        for obl in resulting_obls]
  with torch.no_grad():
    all_obls_tensor = torch.cat([obl.view(1, args.encoding_size) for obl in
                                 all_resulting_obls], dim=0)
    eprint(all_obls_tensor)
    all_obl_scores = target_model(all_obls_tensor)
  outputs = []
  cur_row = 0
  for resulting_obl_lens in num_resulting_obls:
    action_outputs = []
    for num_obls in resulting_obl_lens:
      selected_obl_scores = all_obl_scores[cur_row:cur_row+num_obls]
      action_outputs.append(args.gamma * math.prod(selected_obl_scores))
      cur_row += num_obls
    outputs.append(max(action_outputs))
  actual_values = v_model(inputs).view(len(samples))
  target_values = torch.FloatTensor(outputs)
  loss = F.mse_loss(actual_values, target_values)
  eprint(f"Loss: {loss}")
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

class BufferPopulatingThread(Thread):
  replay_buffer: EncodedReplayBuffer
  signal_change: Event
  def __init__(self, replay_buffer: EncodedReplayBuffer, 
               signal_change: Event, encoding_size: int) -> None:
    self.replay_buffer = replay_buffer
    self.signal_change = signal_change
    self.encoding_size = encoding_size
    super().__init__()
    pass
  def run(self) -> None:
    while True:
      newest_prestate_sample: torch.FloatTensor = \
        torch.zeros(self.encoding_size, dtype=torch.float32) #type: ignore
      sending_worker = dist.recv(tensor=newest_prestate_sample, tag=0)
      newest_action_sample = torch.zeros(1, dtype=int)
      dist.recv(tensor=newest_action_sample, src=sending_worker, tag=1)
      number_of_poststates = torch.zeros(1, dtype=int)
      dist.recv(tensor=number_of_poststates, src=sending_worker, tag=2)
      post_states = []
      for _ in range(number_of_poststates.item()):
        newest_poststate_sample: torch.FloatTensor = \
          torch.zeros(self.encoding_size, dtype=torch.float32) #type: ignore
        dist.recv(tensor=newest_poststate_sample, src=sending_worker, tag=3)
        post_states.append(newest_poststate_sample)
      self.replay_buffer.add_transition(
        (newest_prestate_sample, int(newest_action_sample.item()),
         post_states))
      self.replay_buffer.buffer_steps += 1
      self.signal_change.set()
    pass

EObligation = torch.FloatTensor
ETransition = Tuple[int, Sequence[EObligation]]
EFullTransition = Tuple[EObligation, int, List[EObligation]]
class EncodedReplayBuffer:
  buffer_steps: int
  lock: Lock
  _contents: Dict[EObligation, Tuple[int, Set[ETransition]]]
  window_size: int
  window_end_position: int
  allow_partial_batches: bool
  def __init__(self, window_size: int,
               allow_partial_batches: bool) -> None:
    self.window_size = window_size
    self.window_end_position = 0
    self.allow_partial_batches = allow_partial_batches
    self._contents = {}
    self.lock = Lock()
    self.buffer_steps = 0

  def sample(self, batch_size: int) -> \
        Optional[List[Tuple[EObligation, Set[ETransition]]]]:
    with self.lock:
      sample_pool: List[Tuple[EObligation, Set[ETransition]]] = []
      for obl, (last_updated, transitions) in self._contents.copy().items():
        if last_updated <= self.window_end_position - self.window_size:
          del self._contents[obl]
        else:
          sample_pool.append((obl, transitions))
      if len(sample_pool) >= batch_size:
        return random.sample(sample_pool, batch_size)
      if self.allow_partial_batches and len(sample_pool) > 0:
        return sample_pool
      return None

  def add_transition(self, transition: EFullTransition) -> None:
    with self.lock:
      from_obl, action, _ = transition
      to_obls = tuple(transition[2])
      self._contents[from_obl] = \
        (self.window_end_position,
         {(action, to_obls)} |
         self._contents.get(from_obl, (0, set()))[1])
      self.window_end_position += 1

def send_new_weights(args: argparse.Namespace, v_network: nn.Module, version: int) -> None:
  save_path = str(args.state_dir / "weights" / f"common-v-network-{version}.dat")
  torch.save(v_network.state_dict(), save_path + ".tmp")
  os.rename(save_path + ".tmp", save_path)
  delete_old_common_weights(args)

def delete_old_common_weights(args: argparse.Namespace) -> None:
  cwd = os.getcwd()
  root_dir = str(args.state_dir / "weights")
  os.chdir(root_dir)
  common_network_paths = glob("common-v-network-*.dat")
  os.chdir(cwd)

  common_save_nums = [int(unwrap(re.match(r"common-v-network-(\d+).dat", path)).group(1))
                      for path in common_network_paths]
  latest_common_save_num = max(common_save_nums)
  for save_num in common_save_nums:
    if save_num > latest_common_save_num - args.keep_latest:
      continue
    old_save_path = (args.state_dir / "weights" /
                     f"common-v-network-{save_num}.dat")
    old_save_path.unlink()

if __name__ == "__main__":
  main()
