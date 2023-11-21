#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import sys
import random
import math
import re
from glob import glob

from threading import Thread, Lock, Event
from pathlib import Path
from typing import List, Set, Optional, Dict, Tuple, Sequence
from dataclasses import dataclass

print("Finished std imports", file=sys.stderr)

import torch
print("Finished main torch import", file=sys.stderr)
from torch import nn
import torch.nn.functional as F
from torch import optim
import torch.distributed as dist
import torch.optim.lr_scheduler as scheduler

print("Finished torch imports", file=sys.stderr)
# pylint: disable=wrong-import-position
sys.path.append(str(Path(os.getcwd()) / "src"))
from rl import optimizers, ReplayBuffer, EObligation, VModel
print("Imported rl model setup", file=sys.stderr)
from util import eprint, unwrap, print_time
# pylint: enable=wrong-import-position
eprint("Finished imports")


def main() -> None:
  eprint("Starting main")
  parser = argparse.ArgumentParser()
  parser.add_argument("--state-dir", type=Path, default="drl_state")
  parser.add_argument("-e", "--encoding-size", type=int, required=True)
  parser.add_argument("-l", "--learning-rate", default=5e-6, type=float)
  parser.add_argument("-b", "--batch-size", default=64, type=int)
  parser.add_argument("-g", "--gamma", default=0.9, type=float)
  parser.add_argument("--hidden-size", type=int, default=128)
  parser.add_argument("--tactic-embedding-size", type=int, default=32)
  parser.add_argument("--tactic-vocab-size", type=int)
  parser.add_argument("--num-layers", type=int, default=3)
  parser.add_argument("--allow-partial-batches", action='store_true')
  parser.add_argument("--window-size", type=int, default=2560)
  parser.add_argument("--train-every", type=int, default=8)
  parser.add_argument("--sync-target-every", type=int, default=32)
  parser.add_argument("--keep-latest", default=3, type=int)
  parser.add_argument("--sync-workers-every", type=int, default=16)
  parser.add_argument("--optimizer", choices=optimizers.keys(), default=list(optimizers.keys())[0])
  parser.add_argument("--verifyv-every", type=int, default=None)
  parser.add_argument("--start-from", type=Path, default=None)
  parser.add_argument("--ignore-after", type=int, default=None)
  parser.add_argument("--loss-smoothing", type=int, default=1)
  parser.add_argument("--learning-rate-step", type=int, default=None)
  parser.add_argument("--learning-rate-decay", type=float, default=0.8)
  parser.add_argument("--reset-on-updated-sample", action='store_true')
  parser.add_argument("--no-reset-on-sync", action='store_false', dest='reset_on_sync')
  args = parser.parse_args()

  with (args.state_dir / "learner_scheduled.txt").open('w') as f:
      print("1", file=f, flush=True)
  serve_parameters(args)

vsample_changed: bool = False

def serve_parameters(args: argparse.Namespace, backend='mpi') -> None:
  global vsample_changed
  eprint("Establishing connection")
  dist.init_process_group(backend)
  eprint("Connection established")
  assert torch.cuda.is_available(), "Training node doesn't have CUDA available!" # type: ignore
  device = "cuda"
  v_network: VModel = VModel(args.encoding_size, args.tactic_vocab_size,
                             args.tactic_embedding_size,
                             args.hidden_size,
                             args.num_layers).to(device)
  if args.start_from is not None:
    _, _, _, network_state, \
      tnetwork_state, shorter_proofs_dict, _ = \
        torch.load(str(args.start_from), map_location=device)
    eprint(f"Loading initial weights from {args.start_from}")
    inner_network_state, _encoder_state, _obl_cache, \
      _hidden_size, _num_layers = network_state
    v_network.load_state_dict(inner_network_state)
  target_network: VModel = VModel(args.encoding_size, args.tactic_vocab_size,
                             args.tactic_embedding_size,
                             args.hidden_size,
                             args.num_layers).to(device)
  target_network.load_state_dict(v_network.state_dict())
  optimizer: optim.Optimizer = optimizers[args.optimizer](v_network.parameters(),
                                                          lr=args.learning_rate)
  adjuster = scheduler.StepLR(optimizer, args.learning_rate_step,
                              args.learning_rate_decay)
  replay_buffer = EncodedReplayBuffer(args.window_size,
                                      args.allow_partial_batches)
  signal_change = Event()
  buffer_thread = BufferPopulatingThread(replay_buffer,
                                         signal_change, args.encoding_size,
                                         args.ignore_after)
  buffer_thread.start()

  steps_last_trained = 0
  steps_last_synced_target = 0
  steps_last_synced_workers = 0
  common_network_version = 0
  iters_trained = 0
  last_iter_verified = 0
  loss_buffer: List[torch.FloatTensor] = []

  while signal_change.wait():
    signal_change.clear()
    if replay_buffer.buffer_steps - steps_last_trained >= args.train_every:
      steps_last_trained = replay_buffer.buffer_steps
      loss = train(args, v_network, target_network, optimizer, replay_buffer)
      if args.learning_rate_step is not None and loss is not None:
        adjuster.step()
      if loss is not None:
        if len(loss_buffer) == args.loss_smoothing:
          loss_buffer = loss_buffer[1:] + [loss]
        else:
          eprint(f"Loss buffer is only {len(loss_buffer)} long")
          loss_buffer.append(loss)
      if len(loss_buffer) == args.loss_smoothing:
        smoothed_loss = sum(loss_buffer) / args.loss_smoothing
        lr = optimizer.param_groups[0]['lr']
        eprint(f"Loss: {smoothed_loss} (learning rate {lr:.3e})")
      if vsample_changed and args.reset_on_updated_sample:
        vsample_changed = False
        optimizer = optimizers[args.optimizer](v_network.parameters(),
                                                                lr=args.learning_rate)
        adjuster = scheduler.StepLR(optimizer, args.learning_rate_step,
                                    args.learning_rate_decay)
        eprint("Resetting the optimizer and adjuster")
      iters_trained += 1
    if replay_buffer.buffer_steps - steps_last_synced_target >= args.sync_target_every:
      eprint(f"Syncing target network at step {replay_buffer.buffer_steps} ({replay_buffer.buffer_steps - steps_last_synced_target} steps since last synced)")
      steps_last_synced_target = replay_buffer.buffer_steps
      if args.ignore_after is not None and replay_buffer.buffer_steps > args.ignore_after:
        eprint("Skipping sync because we're ignoring samples now")
      else:
        target_network.load_state_dict(v_network.state_dict())
      if args.reset_on_sync:
        optimizer = optimizers[args.optimizer](v_network.parameters(),
                                                                lr=args.learning_rate)
        adjuster = scheduler.StepLR(optimizer, args.learning_rate_step,
                                    args.learning_rate_decay)
        eprint("Resetting the optimizer and adjuster")
    if replay_buffer.buffer_steps - steps_last_synced_workers >= args.sync_workers_every:
      steps_last_synced_workers = replay_buffer.buffer_steps
      send_new_weights(args, v_network, common_network_version)
      common_network_version += 1
    if args.verifyv_every is not None and \
       iters_trained - last_iter_verified >= args.verifyv_every:
      print_vvalue_errors(args.gamma, v_network, buffer_thread.verification_states)
      last_iter_verified = iters_trained

def train(args: argparse.Namespace, v_model: VModel,
          target_model: nn.Module,
          optimizer: optim.Optimizer,
          replay_buffer: EncodedReplayBuffer) -> Optional[torch.FloatTensor]:
  samples = replay_buffer.sample(args.batch_size)
  if samples is None:
    return None
  eprint(f"Got {len(samples)} samples to train at step {replay_buffer.buffer_steps}")

  local_contexts_encoded = torch.cat([start_obl.local_context
                                               .view(1, args.encoding_size)
                                     for start_obl, _action_records
                                     in samples], dim=0)
  prev_tactic_indices = torch.LongTensor([start_obl.previous_tactic
                                          for start_obl, _ in samples]).to("cuda")
  for prev_tactic_index in prev_tactic_indices:
    assert prev_tactic_index < v_model.prev_tactic_vocab_size,\
      prev_tactic_index
  num_resulting_obls = [[len(resulting_obls)
                         for _action, resulting_obls in action_records]
                        for _start_obl, action_records in samples]
  all_resulting_obls = [obl for _start_obl, action_records in samples
                        for _action, resulting_obls in action_records
                        for obl in resulting_obls]
  if len(all_resulting_obls) > 0:
    with torch.no_grad():
      resulting_local_contexts_tensor = \
          torch.cat([obl.local_context.view(1, args.encoding_size)
                     for obl in all_resulting_obls], dim=0)
      resulting_prev_tactics_tensor = \
          torch.LongTensor([obl.previous_tactic for obl
                            in all_resulting_obls]).to("cuda")
      for prev_tactic_index in resulting_prev_tactics_tensor:
          assert prev_tactic_index < v_model.prev_tactic_vocab_size,\
      prev_tactic_index
      all_obl_scores = target_model(resulting_local_contexts_tensor,
                                    resulting_prev_tactics_tensor)
  else:
      all_obl_scores = torch.FloatTensor([])
  outputs = []
  cur_row = 0
  for resulting_obl_lens in num_resulting_obls:
    if len(resulting_obl_lens) == 0:
      outputs.append(0)
      continue
    action_outputs = []
    for num_obls in resulting_obl_lens:
      selected_obl_scores = all_obl_scores[cur_row:cur_row+num_obls]
      action_outputs.append(args.gamma * math.prod(selected_obl_scores))
      cur_row += num_obls
    outputs.append(max(action_outputs))
  actual_values = v_model(local_contexts_encoded,
                          prev_tactic_indices).view(len(samples))
  device = "cuda"
  target_values = torch.FloatTensor(outputs).to(device)
  loss: torch.FloatTensor = F.mse_loss(actual_values, target_values)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss

class BufferPopulatingThread(Thread):
  replay_buffer: EncodedReplayBuffer
  verification_states: Dict[EObligation, int]
  signal_change: Event
  ignore_after: Optional[int]
  def __init__(self, replay_buffer: EncodedReplayBuffer,
               signal_change: Event, encoding_size: int, ignore_after: Optional[int] = None) -> None:
    self.replay_buffer = replay_buffer
    self.signal_change = signal_change
    self.encoding_size = encoding_size
    self.verification_states = {}
    self.ignore_after = ignore_after
    super().__init__()
    pass
  def run(self) -> None:
    while True:
      send_type = torch.zeros(1, dtype=int)
      sending_worker = dist.recv(tensor=send_type, tag=0)
      if send_type.item() == 0:
        self.receive_experience_sample(sending_worker)
      elif send_type.item() == 1:
        self.receive_verification_sample(sending_worker)
      else:
        assert send_type.item() == 2
        self.receive_negative_sample(sending_worker)

  def receive_experience_sample(self, sending_worker: int) -> None:
    device = "cuda"
    newest_prev_tactic_sample = torch.zeros(1, dtype=int)
    dist.recv(tensor=newest_prev_tactic_sample, src=sending_worker, tag=1)
    newest_prestate_sample: torch.FloatTensor = \
      torch.zeros(self.encoding_size, dtype=torch.float32) #type: ignore
    dist.recv(tensor=newest_prestate_sample, src=sending_worker, tag=2)
    prestate_hash = hash(tuple(newest_prestate_sample.view(-1).tolist() +
                               [newest_prev_tactic_sample.item()]))
    eprint(f"Experience hash is {prestate_hash}")
    newest_hashed_action_sample = torch.zeros(1, dtype=int)
    dist.recv(tensor=newest_hashed_action_sample, src=sending_worker, tag=3)
    newest_encoded_action_sample = torch.zeros(1, dtype=int)
    dist.recv(tensor=newest_encoded_action_sample, src=sending_worker, tag=4)
    number_of_poststates = torch.zeros(1, dtype=int)
    dist.recv(tensor=number_of_poststates, src=sending_worker, tag=5)
    post_states = []
    for _ in range(number_of_poststates.item()):
      newest_poststate_sample: torch.FloatTensor = \
        torch.zeros(self.encoding_size, dtype=torch.float32) #type: ignore
      dist.recv(tensor=newest_poststate_sample, src=sending_worker, tag=6)
      post_states.append(EObligation(newest_poststate_sample.to(device),
                                     newest_encoded_action_sample.item()))
    if self.ignore_after is not None and self.replay_buffer.buffer_steps >= self.ignore_after:
        eprint("Ignoring a sample, but training anyway")
    else:
        from_obl = EObligation(newest_prestate_sample.to(device),
                               newest_prev_tactic_sample.item())
        eprint(f"From obl hash is {hash(from_obl)}")
        self.replay_buffer.add_transition(
          (from_obl,
           int(newest_hashed_action_sample.item()),
           post_states))
    self.replay_buffer.buffer_steps += 1
    self.signal_change.set()

  def receive_verification_sample(self, sending_worker: int) -> None:
    global vsample_changed
    device = "cuda"
    newest_prev_tactic_sample = torch.zeros(1, dtype=int)
    dist.recv(tensor=newest_prev_tactic_sample, src=sending_worker, tag=10)
    state_sample_buffer: torch.FloatTensor = \
      torch.zeros(self.encoding_size, dtype=torch.float32)  #type: ignore
    dist.recv(tensor=state_sample_buffer, src=sending_worker, tag=11)
    target_steps: torch.LongTensor = torch.zeros(1, dtype=int) #type: ignore
    dist.recv(tensor=target_steps, src=sending_worker, tag=12)
    state_sample = EObligation(state_sample_buffer.to(device),
                               newest_prev_tactic_sample.item())
    if state_sample in self.verification_states:
      assert target_steps.item() <= self.verification_states[state_sample], \
        "Got sent a target value  less than the previously expected value for this state!"
      eprint("Updating existing verification sample")
      vsample_changed = True
    else:
      eprint("Adding new verification sample")
    self.verification_states[state_sample] = target_steps.item()

  def receive_negative_sample(self, sending_worker: int) -> None:
    device = "cuda"
    newest_prev_tactic_sample = torch.zeros(1, dtype=int)
    dist.recv(tensor=newest_prev_tactic_sample, src=sending_worker, tag=20)
    state_sample_buffer: torch.FloatTensor = \
      torch.zeros(self.encoding_size, dtype=torch.float32) # type: ignore
    dist.recv(tensor=state_sample_buffer, src=sending_worker, tag=21)
    state_hash = hash(tuple(state_sample_buffer.view(-1).tolist()))
    eprint(f"negative sample hash is {state_hash}")
    self.replay_buffer.add_negative_sample(
      EObligation(state_sample_buffer.to(device),
                  newest_prev_tactic_sample.item()))
    self.replay_buffer.buffer_steps += 1
    self.signal_change.set()

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
      eprint(f"ReplayBuffer has {len(sample_pool)} valid items")
      if len(sample_pool) >= batch_size:
        return random.sample(sample_pool, batch_size)
      if self.allow_partial_batches and len(sample_pool) > 0:
        return sample_pool
      return None

  def add_transition(self, transition: EFullTransition) -> None:
    with self.lock:
      from_obl, action, _ = transition
      eprint(f"Adding positive transition from {hash(from_obl)}")
      to_obls = tuple(transition[2])
      existing_entry = self._contents.get(from_obl, (0, set()))
      assert from_obl not in self._contents or len(existing_entry[1]) > 0
      for existing_action, existing_to_obls in existing_entry[1]:
          assert action != existing_action or to_obls == existing_to_obls,\
            f"From state {hash(from_obl)}, taking action has {action}, " \
            f"resulted in obls {[hash(obl) for obl in to_obls]}, " \
            "but in the past it resulted in obls " \
            f"{[hash(obl) for obl in existing_to_obls]}."
      self._contents[from_obl] = \
        (self.window_end_position,
         {(action, to_obls)} | existing_entry[1])
      self.window_end_position += 1
  def add_negative_sample(self, state: EObligation) -> None:
    with self.lock:
      assert state not in self._contents or len(self._contents[state][1]) == 0, \
        f"State {hash(state)} already had sample {self._contents[state]}, but we're marking it as negative now"
      self._contents[state] = (self.window_end_position, set())
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

def print_vvalue_errors(gamma: float, vnetwork: nn.Module,
                        verification_samples: Dict[EObligation, int]):
  device = "cuda"
  items = list(verification_samples.items())
  predicted_v_values = vnetwork(
    torch.cat([obl.local_context.view(1, -1) for obl, _ in items], dim=0),
    torch.LongTensor([obl.previous_tactic for obl, _ in items]).to(device)).view(-1)
  predicted_steps = torch.log(predicted_v_values) / math.log(gamma)
  target_steps: FloatTensor = torch.tensor([steps for _, steps in items]).to(device) #type: ignore
  step_errors = torch.abs(predicted_steps - target_steps)
  total_error = torch.sum(step_errors).item()
  avg_error = total_error / len(items)
  eprint(f"Average V Value error across {len(items)} initial states: {avg_error:.6f}")

if __name__ == "__main__":
  main()
