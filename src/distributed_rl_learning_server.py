#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import sys
import random
import math
import re
import json
import coq2vec
import hashlib
import time
import shutil
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
  parser.add_argument("--coq2vec-weights", type=Path)
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
  parser.add_argument("--dump-negative-examples", type=Path, default=None)
  parser.add_argument("--dump-replay-buffer", type=Path, default=None)
  parser.add_argument("--ignore-after", type=int, default=None)
  parser.add_argument("--loss-smoothing", type=int, default=1)
  parser.add_argument("--learning-rate-step", type=int, default=None)
  parser.add_argument("--learning-rate-decay", type=float, default=0.8)
  parser.add_argument("--reset-on-updated-sample", action='store_true')
  parser.add_argument("--no-reset-on-sync", action='store_false', dest='reset_on_sync')
  parser.add_argument("--verbose", "-v", help="verbose output", action="count", default=0)
  parser.add_argument("--loss", choices=["simple", "log"],
                      default="simple")
  args = parser.parse_args()
  torch.manual_seed(0)
  random.seed(0)
  torch.use_deterministic_algorithms(True)

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
  term_encoder = coq2vec.CoqTermRNNVectorizer()
  cur_dir = os.path.realpath(os.path.dirname(__file__))
  term_encoder.load_state(torch.load(args.coq2vec_weights,
                          map_location=device))
  num_hyps = 5
  obligation_encoder = coq2vec.CoqContextVectorizer(
    term_encoder, num_hyps)
  encoding_size = unwrap(obligation_encoder.term_encoder
                                           .hidden_size) *\
                  (obligation_encoder.max_num_hypotheses + 1)
  v_network: VModel = VModel(encoding_size, args.tactic_vocab_size,
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
  target_network = VModel(encoding_size, args.tactic_vocab_size,
                             args.tactic_embedding_size,
                             args.hidden_size,
                             args.num_layers).to(device)
  target_network.load_state_dict(v_network.state_dict())
  optimizer: optim.Optimizer = optimizers[args.optimizer](v_network.parameters(),
                                                          lr=args.learning_rate)
  adjuster = scheduler.StepLR(optimizer, args.learning_rate_step,
                              args.learning_rate_decay)
  replay_buffer = EncodedReplayBuffer(args.window_size,
                                      args.allow_partial_batches,
                                      args.verbose)
  true_target_buffer = TrueTargetBuffer(args.allow_partial_batches)
  signal_change = Event()
  buffer_thread = BufferPopulatingThread(
    replay_buffer, true_target_buffer,
    signal_change, obligation_encoder,
    args.verbose, args.ignore_after)
  buffer_thread.start()

  steps_last_trained = 0
  steps_last_synced_target = 0
  steps_last_synced_workers = 0
  common_network_version = 0
  iters_trained = 0
  last_iter_verified = 0
  loss_buffer: List[torch.FloatTensor] = []

  time_started_waiting = time.time()
  while True:
  # while signal_change.wait():
    # signal_change.clear()
    # eprint(f"Waited {time.time() - time_started_waiting:.4f}s for signal")
    # if replay_buffer.buffer_steps - steps_last_trained >= args.train_every:
    with print_time(f"Training iter {iters_trained}"):
      steps_last_trained = replay_buffer.buffer_steps
      loss = train(args, v_network, target_network, optimizer, replay_buffer, true_target_buffer)
      if args.learning_rate_step is not None and loss is not None:
        adjuster.step()
      if loss is not None:
        if len(loss_buffer) == args.loss_smoothing:
          loss_buffer = loss_buffer[1:] + [loss]
        else:
          eprint(f"Loss buffer is only {len(loss_buffer)} long",
                 guard=args.verbose >= 1)
          loss_buffer.append(loss)
        iters_trained += 1
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
        eprint("Resetting the optimizer and adjuster",
               guard=args.verbose >= 1)
      if args.dump_negative_examples is not None:
        with args.dump_negative_examples.open('w') as f:
          negative_examples = replay_buffer.get_negative_samples()
          eprint(f"Dumping {len(negative_examples)} negative examples",
                 guard=args.verbose >= 1)
          for obl in negative_examples:
             print(json.dumps(obl.to_dict()), file=f)
      if args.dump_replay_buffer is not None:
        with open(str(args.dump_replay_buffer) + ".tmp", 'w') as f:
          samples = replay_buffer.get_buffer_contents()
          eprint(f"Dumping {len(samples)} examples",
                 guard=args.verbose >= 1)
          for obl, action, next_obls in samples:
             print(json.dumps((obl.to_dict(), action, 
                               [next_obl.to_dict() for next_obl
                                in next_obls] if next_obls is not None else None)), file=f)
        shutil.copyfile(str(args.dump_replay_buffer) + ".tmp", str(args.dump_replay_buffer)) 
    with print_time("Syncing"):
      if replay_buffer.buffer_steps - steps_last_synced_target >= args.sync_target_every:
        eprint(f"Syncing target network at step {replay_buffer.buffer_steps} "
               f"({replay_buffer.buffer_steps - steps_last_synced_target} "
               "steps since last synced)", guard=args.verbose >= 1)
        steps_last_synced_target = replay_buffer.buffer_steps
        if args.ignore_after is not None and replay_buffer.buffer_steps > args.ignore_after:
          eprint("Skipping sync because we're ignoring samples now", guard=args.verbose >= 1)
        else:
          target_network.load_state_dict(v_network.state_dict())
        if args.reset_on_sync:
          optimizer = optimizers[args.optimizer](v_network.parameters(),
                                                                  lr=args.learning_rate)
          adjuster = scheduler.StepLR(optimizer, args.learning_rate_step,
                                      args.learning_rate_decay)
          eprint("Resetting the optimizer and adjuster", guard=args.verbose >= 1)
      if replay_buffer.buffer_steps - steps_last_synced_workers >= args.sync_workers_every:
        steps_last_synced_workers = replay_buffer.buffer_steps
        send_new_weights(args, v_network, common_network_version)
        common_network_version += 1
    if args.verifyv_every is not None and \
       iters_trained - last_iter_verified >= args.verifyv_every and \
       len(buffer_thread.verification_states) > 0:
      with print_time("Verifying"):
        print_vvalue_errors(args.gamma, v_network, buffer_thread.verification_states)
        last_iter_verified = iters_trained
    time_started_waiting = time.time()

def train(args: argparse.Namespace, v_model: VModel,
          target_model: nn.Module,
          optimizer: optim.Optimizer,
          replay_buffer: EncodedReplayBuffer,
          originaltargetbuffer: TrueTargetBuffer) -> Optional[torch.FloatTensor]:
  
  original_target_samples = originaltargetbuffer.sample(args.batch_size//2)
  replay_buffer_samples = replay_buffer.sample(args.batch_size//2)
  if (replay_buffer_samples is None) and (original_target_samples is None):
    eprint("No samples yet in both replay buffer or original target buffer. "
           "Skipping training", guard=args.verbose >= 1)
    return None
  eprint(f"Got {len(replay_buffer_samples) if replay_buffer_samples else 0} "
         f"samples to train from replay buffer at step {replay_buffer.buffer_steps} "
         f"and {len(original_target_samples) if original_target_samples else 0} "
         f"to train from Original target buffer", guard=args.verbose >= 1)
  

  if original_target_samples:
    original_target_local_contexts_encoded = torch.cat([start_obl.local_context
                                               .view(1, -1)
                                     for start_obl, _
                                     in original_target_samples], dim=0)
    original_target_prev_tactic_indices = torch.LongTensor([start_obl.previous_tactic
                                          for start_obl, _ in original_target_samples]).to("cuda")
    original_target_output = [args.gamma**target for obl,target in original_target_samples]
    original_target_obls = [start_obl for start_obl, _
                            in original_target_samples]
  else:
    original_target_local_contexts_encoded = torch.FloatTensor([]).to('cuda')
    original_target_prev_tactic_indices = torch.LongTensor([]).to('cuda')
    original_target_output = []
    original_target_obls = []
    
  if replay_buffer_samples:
    replay_buffer_local_contexts_encoded = torch.cat([start_obl.local_context
                                               .view(1, -1)
                                     for start_obl, _action_records
                                     in replay_buffer_samples], dim=0)
    replay_buffer_prev_tactic_indices = torch.LongTensor([start_obl.previous_tactic
                                          for start_obl, _ in replay_buffer_samples]).to("cuda")
    replay_buffer_obls = [start_obl for start_obl, _
                          in replay_buffer_samples]
    num_resulting_obls = [[len(resulting_obls)
                          for _action, resulting_obls in action_records]
                          for _start_obl, action_records in replay_buffer_samples]
    all_resulting_obls = [obl for _start_obl, action_records in replay_buffer_samples
                          for _action, resulting_obls in action_records
                          for obl in resulting_obls]
    if len(all_resulting_obls) > 0:
      with torch.no_grad():
        resulting_local_contexts_tensor = \
          torch.cat([obl.local_context.view(1, -1)
                     for obl in all_resulting_obls], dim=0)
        resulting_prev_tactics_tensor = \
            torch.LongTensor([obl.previous_tactic for obl
                              in all_resulting_obls]).to("cuda")
        all_obl_scores = target_model(resulting_local_contexts_tensor,
                                      resulting_prev_tactics_tensor)
    else:
        all_obl_scores = torch.FloatTensor([]) 
    replay_buffer_sample_outputs = []
    cur_row = 0
    for starting_obl, resulting_obl_lens in \
        zip(replay_buffer_obls, num_resulting_obls):
      if len(resulting_obl_lens) == 0:
        replay_buffer_sample_outputs.append(0)
        continue
      action_outputs = []
      for num_obls in resulting_obl_lens:
        selected_obl_scores = [
          obl_score.item() for obl_score in
          all_obl_scores[cur_row:cur_row+num_obls]]
        if args.verbose >= 1:
          eprint(f"For obl {starting_obl.context_hash()}, "
                 f"tactic {starting_obl.previous_tactic}, "
                 f"multiplying scores of obls:")
          for obl, obl_score in zip(all_resulting_obls[cur_row:cur_row+num_obls],
                                    all_obl_scores[cur_row:cur_row+num_obls]) :
              eprint(f"{obl.context_hash()}, {obl.previous_tactic}: "
                     f"{obl_score.item()}")
        action_outputs.append(args.gamma * math.prod(selected_obl_scores))
        if args.verbose >= 1:
          eprint(f"And gamma for new score {action_outputs[-1]}")
        cur_row += num_obls
      replay_buffer_sample_outputs.append(max(action_outputs))
  else :
    replay_buffer_local_contexts_encoded = torch.FloatTensor([]).to('cuda')
    replay_buffer_prev_tactic_indices = torch.LongTensor([]).to('cuda')
    replay_buffer_sample_outputs = []
    replay_buffer_obls = []


  local_context_input = torch.cat( (original_target_local_contexts_encoded, replay_buffer_local_contexts_encoded ) )
  prev_tactics_input = torch.cat( (original_target_prev_tactic_indices, replay_buffer_prev_tactic_indices) )
  outputs = replay_buffer_sample_outputs + original_target_output 

  actual_values = v_model(local_context_input, prev_tactics_input).view(len(replay_buffer_sample_outputs) + len(original_target_output))
  device = "cuda"
  target_values = torch.FloatTensor(outputs).to(device)

  assert len(local_context_input) == len(prev_tactics_input)
  assert len(prev_tactics_input) == len(outputs)

  loss: torch.FloatTensor
  if args.loss == "simple":
    loss = F.mse_loss(actual_values, target_values)
  else:
    assert args.loss == "log"
    loss = F.mse_loss(torch.log(actual_values), torch.log(target_values))
  if args.verbose >= 1:
    eprint("Training obligations to values:")
    for context, prev_tactic, output, actual_value,\
        in zip(original_target_obls + replay_buffer_obls,
               prev_tactics_input, outputs, actual_values):
      # local_loss = F.mse_loss(actual_value, torch.FloatTensor([output]).to(device)[0])
      eprint(f"{context.context_hash()}, {prev_tactic.item()}: "
             f"{actual_value.item():.6f} -> {output:.6f} ")
             #f"(Loss {local_loss:.6f})")
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss

class BufferPopulatingThread(Thread):
  replay_buffer: EncodedReplayBuffer
  verification_states: Dict[EObligation, int]
  target_training_buffer: TrueTargetBuffer
  signal_change: Event
  ignore_after: Optional[int]
  obligation_encoder: coq2vec.CoqContextVectorizer
  encoding_size: int
  max_term_length: int
  verbose: int
  def __init__(self, replay_buffer: EncodedReplayBuffer,
               target_training_buffer: TrueTargetBuffer,
               signal_change: Event,
               obl_encoder: coq2vec.CoqContextVectorizer,
               verbose: int = 0,
               ignore_after: Optional[int] = None) -> None:
    self.verbose = verbose
    self.replay_buffer = replay_buffer
    self.signal_change = signal_change
    self.obligation_encoder = obl_encoder
    self.encoding_size = unwrap(
      obl_encoder.term_encoder.hidden_size) * \
                         (obl_encoder.max_num_hypotheses + 1)
    self.max_term_length = obl_encoder.term_encoder.max_term_length
    self.target_training_buffer = target_training_buffer
    self.verification_states = {}
    self.ignore_after = ignore_after
    self.num_verification_samples_encountered = 0
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
    dist.recv(tensor=newest_prev_tactic_sample,
              src=sending_worker, tag=1)
    terms_per_state = self.obligation_encoder.max_num_hypotheses + 1
    newest_prestate_sequence: torch.LongTensor = \
      torch.zeros([terms_per_state, self.max_term_length],
                  dtype=int) #type: ignore
    dist.recv(tensor=newest_prestate_sequence, src=sending_worker, tag=2)
    newest_hashed_action_sample = torch.zeros(1, dtype=int)
    dist.recv(tensor=newest_hashed_action_sample,
              src=sending_worker, tag=3)
    newest_encoded_action_sample = torch.zeros(1, dtype=int)
    dist.recv(tensor=newest_encoded_action_sample, src=sending_worker, tag=4)
    number_of_poststates = torch.zeros(1, dtype=int)
    dist.recv(tensor=number_of_poststates, src=sending_worker, tag=5)
    post_state_sequences: List[torch.LongTensor] = []
    for _ in range(number_of_poststates.item()):
      newest_poststate_sequence: torch.LongTensor = \
        torch.zeros([terms_per_state, self.max_term_length],
                    dtype=int) # type: ignore
      dist.recv(tensor=newest_poststate_sequence,
                src=sending_worker, tag=6)
      post_state_sequences.append(newest_poststate_sequence.unsqueeze(0))

    with torch.no_grad():
      torch.manual_seed(0)
      newest_states_encoded = self.obligation_encoder\
                                  .seq_lists_to_vectors(
        torch.cat([newest_prestate_sequence.unsqueeze(0)]
                  + post_state_sequences,
                  dim=0)).view(number_of_poststates + 1, -1)
    post_states = [EObligation(state_encoded.to(device),
                               newest_encoded_action_sample.item(),
                               tokens)
                   for state_encoded, tokens in
                   zip(newest_states_encoded[1:], post_state_sequences)]
    newest_prestate_sample = newest_states_encoded[0]

    if self.ignore_after is not None and self.replay_buffer.buffer_steps >= self.ignore_after:
        eprint("Ignoring a sample, but training anyway", guard=self.verbose >= 1)
    else:
        from_obl = EObligation(newest_prestate_sample.to(device),
                               newest_prev_tactic_sample.item(),
                               newest_prestate_sequence.unsqueeze(0))
        sequence_hash = int.from_bytes(hashlib.md5(
          json.dumps(newest_prestate_sequence.view(-1).tolist(),
                     sort_keys=True).encode('utf-8')).digest())
        eprint(f"EObligation hash is {from_obl.context_hash()} "
               f"with previous tactic "
               f"{newest_prev_tactic_sample.item()}, "
               f"from sequence_hash {sequence_hash}.",
               guard=self.verbose >= 1)
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
    dist.recv(tensor=newest_prev_tactic_sample,
              src=sending_worker, tag=10)
    terms_per_state = self.obligation_encoder.max_num_hypotheses + 1
    state_sequence_buffer: torch.LongTensor = \
      torch.zeros([terms_per_state, self.max_term_length],
                  dtype=int)  #type: ignore
    dist.recv(tensor=state_sequence_buffer, src=sending_worker, tag=11)
    target_steps: torch.LongTensor = torch.zeros(1, dtype=int) #type: ignore
    dist.recv(tensor=target_steps, src=sending_worker, tag=12)
    with torch.no_grad():
      state_sample_vec = self.obligation_encoder\
                             .seq_lists_to_vectors(
        state_sequence_buffer.unsqueeze(0)).view(1,-1).squeeze(0)
    state_sample = EObligation(state_sample_vec.to(device),
                               newest_prev_tactic_sample.item(),
                               state_sequence_buffer.unsqueeze(0))
    sequence_hash = int.from_bytes(hashlib.md5(
      json.dumps(state_sequence_buffer.view(-1).tolist(),
                 sort_keys=True).encode("utf-8")).digest())
    eprint(f"Receiving targeted sample {state_sample.context_hash()} "
           f"with target {target_steps.item()}, "
           f"from sequence hash {sequence_hash}.", guard=self.verbose >= 1)
    if state_sample in self.verification_states:
      if target_steps.item() > self.verification_states[state_sample]:
        eprint("WARNING: Trying to add validation sample "
               "but got a larger length than previous sample! "
               "Skipping...", self.verbose >= 1)
        return
      eprint("Updating existing verification sample", guard=self.verbose >= 1)
      vsample_changed = True
      self.verification_states[state_sample] = target_steps.item()
    elif state_sample in self.target_training_buffer._contents :
      eprint("Updating existing fixed training sample", guard=self.verbose >= 1)
      self.target_training_buffer.add_target(state_sample, target_steps.item())
    elif self.num_verification_samples_encountered % 3 == 0  :
      eprint("Adding new verification sample", guard=self.verbose >= 1)
      self.verification_states[state_sample] = target_steps.item()
    else :
      eprint("Adding new original target sample", guard=self.verbose >= 1)
      if self.replay_buffer.exists(state_sample) :
        self.replay_buffer.remove(state_sample)
      self.target_training_buffer.add_target( state_sample, target_steps.item() )
    self.num_verification_samples_encountered += 1

  def receive_negative_sample(self, sending_worker: int) -> None:
    device = "cuda"
    newest_prev_tactic_sample = torch.zeros(1, dtype=int)
    dist.recv(tensor=newest_prev_tactic_sample, src=sending_worker, tag=20)
    terms_per_state = self.obligation_encoder.max_num_hypotheses + 1
    state_sequence_buffer: torch.LongTensor = \
      torch.zeros([terms_per_state, self.max_term_length],
                  dtype=int) # type: ignore
    dist.recv(tensor=state_sequence_buffer, src=sending_worker, tag=21)
    sequence_hash = int.from_bytes(hashlib.md5(
      json.dumps(state_sequence_buffer.view(-1).tolist(),
                 sort_keys=True).encode("utf-8")).digest())
    with torch.no_grad():
      state_sample_vec = self.obligation_encoder\
                             .seq_lists_to_vectors(
        state_sequence_buffer.unsqueeze(0)).view(1, -1).squeeze(0)
    state_sample = EObligation(state_sample_vec.to(device),
                               newest_prev_tactic_sample.item(),
                               state_sequence_buffer.unsqueeze(0))
    self.replay_buffer.add_negative_sample(state_sample)
    eprint(f"Receiving negative sample {state_sample.context_hash()} "
           f"from sequence hash {sequence_hash}.",
           guard=self.verbose >= 1)
    self.replay_buffer.buffer_steps += 1
    self.signal_change.set()

ETransition = Tuple[int, Sequence[EObligation]]
EFullTransition = Tuple[EObligation, int, List[EObligation]]

class TrueTargetBuffer:
  _contents: Dict[EObligation, int]
  lock: Lock
  def __init__(self, allow_partial_batches) -> None:
    self.lock = Lock()
    self._contents = {}
    self.allow_partial_batches = allow_partial_batches
    return None

  def sample(self, batch_size:int) -> \
      Optional[List[Tuple[EObligation, int]]] :
    sampled = None
    with self.lock:
      if len(self._contents) >= batch_size :
        sampled = random.sample(list(self._contents.items()), batch_size)
      if self.allow_partial_batches and len(self._contents) > 0:
        sampled = list(self._contents.items())
    return sampled

  
  def add_target(self, state : EObligation, target : int) -> None :
    with self.lock :
      global vsample_changed
      if state in self._contents :
        if target > self._contents[state] :
          eprint("WARNING: Got sent a target less than previous target, ignoring")
          return
      self._contents[state] = target
      vsample_changed = True

class EncodedReplayBuffer:
  buffer_steps: int
  lock: Lock
  _contents: Dict[EObligation, Tuple[int, Set[ETransition]]]
  window_size: int
  window_end_position: int
  allow_partial_batches: bool
  verbose: int
  def __init__(self, window_size: int,
               allow_partial_batches: bool,
               verbose: int = 0) -> None:
    self.verbose = verbose
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
      eprint(f"ReplayBuffer has {len(sample_pool)} valid items",
             guard=self.verbose >= 1)
      if len(sample_pool) >= batch_size:
        return random.sample(sample_pool, batch_size)
      if self.allow_partial_batches and len(sample_pool) > 0:
        return sample_pool
      return None

  def remove(self, from_obl:EObligation) -> None :
    del self._contents[from_obl]
  
  def exists(self, from_obl:EObligation) -> bool:
    return from_obl in self._contents
    
  def add_transition(self, transition: EFullTransition) -> None:
    with self.lock:
      from_obl, action, _ = transition
      to_obls = tuple(transition[2])
      existing_entry = self._contents.get(from_obl, (0, set()))
      if from_obl in self._contents and len(existing_entry[1]) == 0:
        eprint("WARNING: Trying to add transition from "
               "{from_obl.context_hash()};{from_obl.previous_tactics}, "
               "but it's already marked as a negative example! Skipping...")
        return
      # assert from_obl not in self._contents or len(existing_entry[1]) > 0
      for existing_action, existing_to_obls in existing_entry[1]:
        if action == existing_action:
          if to_obls != existing_to_obls:
            eprint(f"WARNING: Transition from state "
                   f"{from_obl.context_hash()};"
                   f"{from_obl.previous_tactic} "
                   "clashed with previous entry! Skipping")
          return
        # assert action != existing_action or to_obls == existing_to_obls,\
        #   f"From state {hash(from_obl)}, taking action has {action}, " \
        #   f"resulted in obls {[hash(obl) for obl in to_obls]}, " \
        #   "but in the past it resulted in obls " \
        #   f"{[hash(obl) for obl in existing_to_obls]}."
      eprint(f"Adding positive transition from "
             f"{from_obl.context_hash()};{from_obl.previous_tactic}",
             guard=self.verbose >= 1)

      self._contents[from_obl] = \
        (self.window_end_position,
         {(action, to_obls)} | existing_entry[1])
      self.window_end_position += 1

  def add_negative_sample(self, state: EObligation) -> None:
    with self.lock:
      if state in self._contents :
        if len(self._contents[state][1]) > 0:
          eprint(f"WARNING: State {state.context_hash()};"
                 f"{state.previous_tactic} already had sample "
                 f"{self._contents[state]}, but we're trying to mark it as negative. "
                 "Skipping...")
        return
      eprint(f"Adding negative transition from "
             f"{state.context_hash()};{state.previous_tactic} ",
             guard=self.verbose >= 1)
      # assert state not in self._contents or len(self._contents[state][1]) == 0, \
      #   f"State {hash(state)} already had sample {self._contents[state]}, but we're marking it as negative now"
      self._contents[state] = (self.window_end_position, set())
      self.window_end_position += 1
  def get_negative_samples(self) -> List[EObligation]:
    return [obl for obl, (pos, transitions)  in self._contents.items()
            if len(transitions) == 0]
  def get_buffer_contents(self) -> List[Tuple[EObligation, int,
                                              List[EObligation]]]:
    results: List[Tuple[EObligation, int,
                        Optional[List[Eobligation]]]]  = []
    for eobligation, (position, transitions) in self._contents.items():
      if len(transitions) == 0:
        results.append((eobligation, 0, None))
      for action, obligations in transitions:
        results.append((eobligation, action, obligations))
    return results

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
  eprint(f"Average step error across {len(items)} initial states: {avg_error:.6f}")

if __name__ == "__main__":
  main()
