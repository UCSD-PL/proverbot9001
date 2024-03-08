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
import sys
import functools
import signal
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
from util import eprint, unwrap, print_time, sighandler_context
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
  parser.add_argument("--optimizer", choices=optimizers.keys(), default=list(optimizers.keys())[0])
  parser.add_argument("--verifyv-every", type=int, default=None)
  parser.add_argument("--start-from", type=Path, default=None)
  parser.add_argument("--dump-negative-examples", type=Path, default=None)
  parser.add_argument("--dump-replay-buffer", type=Path, default=None)
  parser.add_argument("--start-after", type=int, default=None)
  parser.add_argument("--ignore-after", type=int, default=None)
  parser.add_argument("--loss-smoothing", type=int, default=1)
  parser.add_argument("--learning-rate-step", type=int, default=None)
  parser.add_argument("--learning-rate-decay", type=float, default=0.8)
  parser.add_argument("--reset-on-updated-sample", action='store_true')
  parser.add_argument("--no-reset-on-sync", action='store_false', dest='reset_on_sync')
  parser.add_argument("--decrease-lr-on-reset", action='store_true', dest='decrease_lr_on_reset')
  parser.add_argument("--verbose", "-v", help="verbose output", action="count", default=0)
  parser.add_argument("--loss", choices=["simple", "log"],
                      default="simple")
  parser.add_argument("--no-catch-interrupts", action='store_false', dest="catch_interrupts")
  args = parser.parse_args()
  torch.manual_seed(0)
  random.seed(0)
  os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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
  term_encoder.load_state(torch.load(args.coq2vec_weights, map_location=device))
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
      _args = network_state
    v_network.load_state_dict(inner_network_state)
  target_network = VModel(encoding_size, args.tactic_vocab_size,
                             args.tactic_embedding_size,
                             args.hidden_size,
                             args.num_layers).to(device)
  target_network.load_state_dict(v_network.state_dict())
  optimizer: optim.Optimizer = optimizers[args.optimizer](
    [{"params": v_network.tactic_embedding.parameters(),
      "lr": args.learning_rate * 20},
     {"params": v_network.prediction_network.parameters()}],
    lr=args.learning_rate)
  adjuster = scheduler.StepLR(optimizer, args.learning_rate_step,
                              args.learning_rate_decay)
  learning_rate_restart = args.learning_rate
  replay_buffer = EncodedReplayBuffer(args.window_size,
                                      args.allow_partial_batches,
                                      args.verbose)
  true_target_buffer = TrueTargetBuffer(args.allow_partial_batches)
  signal_change = Event()
  signal_end = Event()
  buffer_thread = BufferPopulatingThread(
    replay_buffer, true_target_buffer,
    v_network, signal_change, signal_end, obligation_encoder,
    args)
  buffer_thread.start()

  steps_last_trained = 0
  steps_last_synced_target = 0
  steps_last_synced_workers = 0
  common_network_version = 0
  iters_trained = 0
  last_iter_verified = 0
  loss_buffer: List[torch.FloatTensor] = []

  time_started_waiting = time.time()
  signal_change.wait()
  signal_change.clear()
  if args.start_after:
     for i in range(args.start_after - 1):
       eprint(f"Waited {i} samples before starting "
              "(because of --start-after)")
       signal_change.wait()
       signal_change.clear()
  with sighandler_context(signal.SIGINT,
                          functools.partial(interrupt_early, args,
                                            v_network, signal_end),
                          guard=args.catch_interrupts):
    while True:
      if signal_end.is_set():
        eprint("Finished learning, exiting learning server", guard=args.verbose >= 1)
        sys.exit(0)
    # while signal_change.wait():
      # signal_change.clear()
      # eprint(f"Waited {time.time() - time_started_waiting:.4f}s for signal")
      # if replay_buffer.buffer_steps - steps_last_trained >= args.train_every:
      if len(replay_buffer) < args.batch_size and not args.allow_partial_batches:
        continue
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
          eprint(f"Loss: {smoothed_loss} (learning rate {lr/20:.3e})")
        if vsample_changed and args.reset_on_updated_sample:
          vsample_changed = False
          optimizer = optimizers[args.optimizer](
            [{"params": v_network.tactic_embedding.parameters(),
              "lr": args.learning_rate * 20},
             {"params": v_network.prediction_network.parameters()}],
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
      if iters_trained - steps_last_synced_target >= args.sync_target_every:
        eprint(f"Syncing target network at step {replay_buffer.buffer_steps} "
               f"({iters_trained - steps_last_synced_target} "
               "steps since last synced)", guard=args.verbose >= 1)
        steps_last_synced_target = iters_trained
        if args.ignore_after is not None and iters_trained > args.ignore_after:
          eprint("Skipping sync because we're ignoring samples now", guard=args.verbose >= 1)
        else:
          target_network.load_state_dict(v_network.state_dict())
        if args.reset_on_sync:
          if args.decrease_lr_on_reset:
            learning_rate_restart = learning_rate_restart * args.learning_rate_decay
          optimizer = optimizers[args.optimizer](
            [{"params": v_network.tactic_embedding.parameters(),
              "lr": learning_rate_restart * 20},
             {"params": v_network.prediction_network.parameters()}],
            lr=learning_rate_restart)
          adjuster = scheduler.StepLR(optimizer, args.learning_rate_step,
                                      args.learning_rate_decay)
          eprint("Resetting the optimizer and adjuster for sync", guard=args.verbose >= 1)
      if args.verifyv_every is not None and \
         iters_trained - last_iter_verified >= args.verifyv_every and \
         len(buffer_thread.verification_states) > 0:
        with print_time("Verifying"):
          error = print_vvalue_errors(args.gamma, v_network,
                                      buffer_thread.verification_states)
          last_iter_verified = iters_trained
        with (args.state_dir / "latest_error.txt").open('w') as f:
          print(error, file=f)
      time_started_waiting = time.time()

def fairly_sample_buffers(args: argparse.Namespace,
                          replay_buffer: EncodedReplayBuffer,
                          original_target_buffer: TrueTargetBuffer) \
   -> Optional[Tuple[List[ReplayBufferSample], List[TrueBufferSample]]]:
  target_buffer_size = args.batch_size
  tbuf_size = len(original_target_buffer)
  rbuf_size = len(replay_buffer)
  smaller_half = target_buffer_size // 2
  larger_half = target_buffer_size - smaller_half
  if tbuf_size == 0 and rbuf_size == 0:
    return None
  if tbuf_size + rbuf_size < target_buffer_size \
      and not args.allow_partial_batches:
    return None
  if rbuf_size >= larger_half:
    if tbuf_size >= smaller_half:
      return (unwrap(replay_buffer.sample(larger_half)),
              unwrap(original_target_buffer.sample(smaller_half)))
    return (unwrap(replay_buffer.sample(target_buffer_size - tbuf_size)),
            unwrap(original_target_buffer.sample(tbuf_size)))
  else:
    obuffer_samples = original_target_buffer.sample(target_buffer_size
                                                    - rbuf_size)
    if obuffer_samples is None:
      obuffer_samples = []
    return (unwrap(replay_buffer.sample(rbuf_size)), obuffer_samples)

def obl_tensors(obls: List[EObligation]) -> \
  Tuple[torch.FloatTensor, torch.LongTensor]:
  assert len(obls) > 0
  contexts = torch.cat([obl.local_context.view(1, -1)
                        for obl in obls])
  prev_tactics = torch.LongTensor(
    [obl.previous_tactic for obl in obls]).to("cuda")
  return (contexts, prev_tactics)

def tbuf_samples_to_tensors(args: argparse.Namespace,
                            samples: List[TrueBufferSample])\
  -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
  if len(samples) == 0:
    return (torch.FloatTensor([]).to("cuda"),
            torch.LongTensor([]).to("cuda"),
            torch.FloatTensor([]).to("cuda"))
  contexts, prev_tactics = obl_tensors([obl for obl, _ in samples])
  outputs = torch.FloatTensor(
    [args.gamma ** target for obl, target in samples]).to("cuda")
  if args.verbose >= 1:
    for idx, (obl, target) in enumerate(samples):
      eprint(f"[{idx}] For obl {obl.context_hash()}, "
             f"{obl.previous_tactic}, using score "
             f"{args.gamma} ** {target} = {args.gamma ** target}")
  return (contexts, prev_tactics, outputs)

def rbuf_samples_to_tensors(args: argparse.Namespace,
                            target_model: VModel,
                            samples: List[ReplayBufferSample],
                            starting_idx: int = 0)\
    -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
  if len(samples) == 0:
    return (torch.FloatTensor([]).to("cuda"),
            torch.LongTensor([]).to("cuda"),
            torch.FloatTensor([]).to("cuda"))
  contexts, prev_tactics = obl_tensors(
    [start_obl for start_obl, _ in samples])
  num_resulting_obls = [[len(resulting_obls)
                         for _action, resulting_obls in action_records]
                         for _start_obl, action_records in samples]
  all_resulting_obls = [obl for _start_obl, action_records in samples
                        for _action, resulting_obls in action_records
                        for obl in resulting_obls]

  if len(all_resulting_obls) > 0:
    with torch.no_grad():
      resulting_contexts, resulting_prev_tacs = \
        obl_tensors(all_resulting_obls)
      resulting_obl_scores = target_model(resulting_contexts,
                                          resulting_prev_tacs)
  else:
      resulting_obl_scores = torch.FloatTensor([])

  outputs = []
  cur_row = 0

  for idx, ((starting_obl, _), resulting_obl_lens) in \
      enumerate(zip(samples, num_resulting_obls),
                start=starting_idx):
    if len(resulting_obl_lens) == 0:
      eprint(f"[{idx}] for obl {starting_obl.context_hash()}, "
             f"{starting_obl.previous_tactic}, "
             "training as negative sample", args.verbose >= 1)
      outputs.append(sys.float_info.min)
      continue
    action_outputs = []
    for num_obls in resulting_obl_lens:
      selected_obl_scores = [
        obl_score.item() for obl_score in
        resulting_obl_scores[cur_row:cur_row+num_obls]]
      action_outputs.append(args.gamma * math.prod(selected_obl_scores))
      if args.verbose >= 1:
        eprint(f"[{idx}] For obl {starting_obl.context_hash()}, "
               f"{starting_obl.previous_tactic}, "
               "multiplying scores of obls:")
        selected_obls = all_resulting_obls[cur_row:cur_row+num_obls]
        for obl, obl_score in zip(selected_obls, selected_obl_scores):
          eprint(f"{obl.context_hash()}, {obl.previous_tactic}: "
                 f"{obl_score}")
        eprint(f"And gamma ({args.gamma}) for new score "
               f"{action_outputs[-1]}")
      cur_row += num_obls
    outputs.append(max(action_outputs))
  return contexts, prev_tactics, torch.FloatTensor(outputs).to("cuda")

def train(args: argparse.Namespace, v_model: VModel,
          target_model: VModel,
          optimizer: optim.Optimizer,
          replay_buffer: EncodedReplayBuffer,
          originaltargetbuffer: TrueTargetBuffer) \
    -> Optional[torch.FloatTensor]:
  samples = fairly_sample_buffers(
    args, replay_buffer, originaltargetbuffer)
  assert samples is not None, \
    "Shouldn't try to train before there are samples"
  rbuf_samples, tbuf_samples = samples
  start_obl_hashes_seen: Set[Tuple[int, int]] = set()
  eprint(f"Got {len(rbuf_samples)} samples from replay buffer, "
         f"{len(tbuf_samples)} samples from true target buffer",
         guard=args.verbose >= 1)

  tbuf_contexts, tbuf_prev_tacs, tbuf_outputs = \
    tbuf_samples_to_tensors(args, tbuf_samples)
  tbuf_obls = [obl for obl, _ in tbuf_samples]

  rbuf_contexts, rbuf_prev_tacs, rbuf_outputs = \
    rbuf_samples_to_tensors(args, target_model, rbuf_samples,
                            starting_idx=len(tbuf_samples))
  rbuf_obls = [obl for obl, _ in rbuf_samples]

  all_contexts = torch.cat((tbuf_contexts, rbuf_contexts))
  all_prev_tacs = torch.cat((tbuf_prev_tacs, rbuf_prev_tacs))
  all_obls = tbuf_obls + rbuf_obls
  target_values = torch.cat((tbuf_outputs, rbuf_outputs))
  actual_values = v_model(all_contexts, all_prev_tacs)\
      .view(len(tbuf_samples) + len(rbuf_samples))

  device = "cuda"

  loss: torch.FloatTensor
  if args.loss == "simple":
    loss = F.mse_loss(actual_values, target_values)
  else:
    assert args.loss == "log"
    loss = F.mse_loss(torch.log(actual_values), torch.log(target_values))
  if args.verbose >= 1:
    eprint("Training obligations to values:")
    for idx, (context, prev_tactic, output, actual_value) \
        in enumerate(zip(tbuf_obls + rbuf_obls,
                     all_prev_tacs, target_values, actual_values)):
      eprint(f"[{idx}] {context.context_hash()}, {prev_tactic.item()}: "
             f"{actual_value.item():.6f} -> {output.item():.6f} ")
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss

class BufferPopulatingThread(Thread):
  replay_buffer: EncodedReplayBuffer
  verification_states: Dict[EObligation, int]
  target_training_buffer: TrueTargetBuffer
  signal_change: Event
  signal_end: Event
  ignore_after: Optional[int]
  obligation_encoder: coq2vec.CoqContextVectorizer
  v_model: VModel
  actors_finished: List[int]
  encoding_size: int
  max_term_length: int
  verbose: int
  def __init__(self, replay_buffer: EncodedReplayBuffer,
               target_training_buffer: TrueTargetBuffer,
               v_model: VModel,
               signal_change: Event,
               signal_end: Event,
               obl_encoder: coq2vec.CoqContextVectorizer,
               args: argparse.Namespace) -> None:
    self.args = args
    self.verbose = args.verbose
    self.replay_buffer = replay_buffer
    self.signal_change = signal_change
    self.signal_end = signal_end
    self.obligation_encoder = obl_encoder
    self.v_model = v_model
    self.encoding_size = unwrap(
      obl_encoder.term_encoder.hidden_size) * \
                         (obl_encoder.max_num_hypotheses + 1)
    self.max_term_length = unwrap(obl_encoder.term_encoder
                                  .max_term_length)
    self.target_training_buffer = target_training_buffer
    self.verification_states = {}
    self.ignore_after = args.ignore_after
    self.num_verification_samples_encountered = 0
    self.actors_finished: List[int] = []
    super().__init__()
    pass
  def run(self) -> None:
    while True:
      if self.signal_end.is_set():
        eprint("Ended early, ending receiver thread", guard=self.verbose >= 1)
        return
      send_type = torch.zeros(1, dtype=int)
      sending_worker = dist.recv(tensor=send_type, tag=0)
      if send_type.item() == 0:
        self.receive_experience_sample(sending_worker)
      elif send_type.item() == 1:
        self.receive_verification_sample(sending_worker)
      elif send_type.item() == 2:
        self.receive_negative_sample(sending_worker)
      elif send_type.item() == 3:
        self.send_latest_weights(sending_worker)
      elif send_type.item() == 4:
        self.actors_finished.append(sending_worker)
        num_actors = int(os.environ["SLURM_NTASKS"]) - 1
        if len(self.actors_finished) == num_actors:
          save_new_weights(self.args, self.v_model, 1)
          self.signal_end.set()
          sys.exit(0)
      else:
        assert False, "Bad mesage tag"

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
        if from_obl in self.target_training_buffer._contents:
          eprint(f"Skipping {from_obl.context_hash()}, "
                 f"{from_obl.previous_tactic} "
                 "because it's already in the original target buffer",
                 guard=self.verbose >= 1)
          return

        sequence_hash = int.from_bytes(hashlib.md5(
          json.dumps(newest_prestate_sequence.view(-1).tolist(),
                     sort_keys=True).encode('utf-8')).digest(), byteorder='big')
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
                 sort_keys=True).encode("utf-8")).digest(), byteorder='big')
    eprint(f"Receiving targeted sample {state_sample.context_hash()}"
           f";{newest_prev_tactic_sample.item()} "
           f"with target {target_steps.item()}, "
           f"from sequence hash {sequence_hash}.",
           guard=self.verbose >= 1)
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
                 sort_keys=True).encode("utf-8")).digest(), byteorder='big')
    with torch.no_grad():
      state_sample_vec = self.obligation_encoder\
                             .seq_lists_to_vectors(
        state_sequence_buffer.unsqueeze(0)).view(1, -1).squeeze(0)
    state_sample = EObligation(state_sample_vec.to(device),
                               newest_prev_tactic_sample.item(),
                               state_sequence_buffer.unsqueeze(0))
    eprint(f"Receiving negative sample {state_sample.context_hash()} "
           f"from sequence hash {sequence_hash}.",
           guard=self.verbose >= 1)
    if state_sample in self.target_training_buffer._contents:
      eprint(f"Receiving negative sample that used to be in target buffer, "
             f"removing from target buffer.", guard=self.verbose >= 1)
      self.target_training_buffer.remove_target(state_sample)

    self.replay_buffer.add_negative_sample(state_sample)
    self.replay_buffer.buffer_steps += 1
    self.signal_change.set()
  def send_latest_weights(self, target_worker: int) -> None:
    for param in self.v_model.parameters():
      dist.send(tensor=param.data.to("cpu"), tag=30, dst=target_worker)

ETransition = Tuple[int, Sequence[EObligation]]
EFullTransition = Tuple[EObligation, int, List[EObligation]]

TrueBufferSample = Tuple[EObligation, int]

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
    with self.lock:
      if len(self._contents) >= batch_size :
        return random.sample(list(self._contents.items()), batch_size)
      elif self.allow_partial_batches and len(self._contents) > 0:
        return list(self._contents.items())
      else:
        return None
  
  def add_target(self, state : EObligation, target : int) -> None :
    with self.lock :
      global vsample_changed
      if state in self._contents :
        if target > self._contents[state] :
          eprint("WARNING: Got sent a target less than previous target, ignoring")
          return
      self._contents[state] = target
      vsample_changed = True

  def remove_target(self, state: EObligation) -> None:
    with self.lock:
      assert state in self._contents, \
        "Tried to remove an obligation that wasn't "\
        "in the target buffer!"
      del self._contents[state]
  def __len__(self) -> int:
    return len(self._contents)

ReplayBufferSample = Tuple[EObligation, Set[ETransition]]

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
        Optional[List[ReplayBufferSample]]:
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
               "{from_obl.context_hash()}, {from_obl.previous_tactics}, "
               "but it's already marked as a negative example! Skipping...")
        return
      # assert from_obl not in self._contents or len(existing_entry[1]) > 0
      for existing_action, existing_to_obls in existing_entry[1]:
        if action == existing_action:
          if to_obls != existing_to_obls:
            eprint(f"WARNING: Transition from state "
                   f"{from_obl.context_hash()}, "
                   f"{from_obl.previous_tactic} "
                   "clashed with previous entry! Skipping")
          return
        # assert action != existing_action or to_obls == existing_to_obls,\
        #   f"From state {hash(from_obl)}, taking action has {action}, " \
        #   f"resulted in obls {[hash(obl) for obl in to_obls]}, " \
        #   "but in the past it resulted in obls " \
        #   f"{[hash(obl) for obl in existing_to_obls]}."
      eprint(f"Adding positive transition from "
             f"{from_obl.context_hash()}, {from_obl.previous_tactic}",
             guard=self.verbose >= 1)

      self._contents[from_obl] = \
        (self.window_end_position,
         {(action, to_obls)} | existing_entry[1])
      self.window_end_position += 1

  def add_negative_sample(self, state: EObligation) -> None:
    with self.lock:
      if state in self._contents :
        if len(self._contents[state][1]) > 0:
          eprint(f"WARNING: State {state.context_hash()}, "
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
                                              Optional[List[EObligation]]]]:
    results: List[Tuple[EObligation, int,
                        Optional[List[EObligation]]]]  = []
    for eobligation, (position, transitions) in self._contents.items():
      if len(transitions) == 0:
        results.append((eobligation, 0, None))
      for action, obligations in transitions:
        results.append((eobligation, action, list(obligations)))
    return results
  def __len__(self) -> int:
    return len(self._contents)

def save_new_weights(args: argparse.Namespace, v_network: nn.Module, version: int) -> None:
  save_path = str(args.state_dir / "weights" / f"common-v-network-{version}.dat")
  torch.save(v_network.state_dict(), save_path + ".tmp")
  os.rename(save_path + ".tmp", save_path)

def print_vvalue_errors(gamma: float, vnetwork: nn.Module,
                        verification_samples: Dict[EObligation, int]):
  device = "cuda"
  items = list(verification_samples.items())
  predicted_v_values = vnetwork(
    torch.cat([obl.local_context.view(1, -1) for obl, _ in items], dim=0),
    torch.LongTensor([obl.previous_tactic for obl, _ in items]).to(device)).view(-1)
  predicted_steps = torch.log(predicted_v_values) / math.log(gamma)
  num_predicted_zeros = torch.count_nonzero(predicted_steps == float("inf"))
  target_steps: FloatTensor = torch.tensor([steps for _, steps in items]).to(device) #type: ignore
  step_errors = torch.abs(predicted_steps - target_steps)

  total_error = torch.sum(torch.where(predicted_steps == float("inf"),
                                      torch.zeros_like(step_errors),
                                      step_errors)).item()
  avg_error = total_error / (len(items) - num_predicted_zeros)
  eprint(f"Average step error across {len(items) - num_predicted_zeros} "
         f"initial states with finite predictions: {avg_error:.6f}")
  eprint(f"{num_predicted_zeros} predicted as infinite steps (impossible)")
  return float(avg_error.item())

def interrupt_early(args: argparse.Namespace, v_model: VModel,
                    signal_end: Event,
                    *rest_args) -> None:
  save_new_weights(args, v_model, 0)
  signal_end.set()
  sys.exit(1)

if __name__ == "__main__":
  main()
