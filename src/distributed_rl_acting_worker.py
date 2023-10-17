#!/usr/bin/env python3

import argparse
import os
import json
import random
import re
import sys
from glob import glob

from pathlib import Path
from typing import Tuple, List, Dict, Optional, Set, Counter, Any, OrderedDict
from dataclasses import dataclass

import coq2vec
import coq_serapy
from coq_serapy.contexts import (Obligation, ProofContext,
                                 truncate_tactic_context, FullContext)
import torch
import torch.cuda
import torch.distributed as dist

sys.path.append(str(Path(os.getcwd()) / "src"))

#pylint: disable=wrong-import-position
import distributed_rl as drl
import rl
from gen_rl_tasks import RLTask
from models.tactic_predictor import TacticPredictor
from search_worker import get_predictor
from util import FileLock, eprint, safe_abbrev, print_time, unwrap
#pylint: enable=wrong-import-position

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--progress", "-P", help="show progress of files",
                      action='store_true')
  parser.add_argument("filenames", nargs="+", type=Path)
  parser.add_argument("--prelude", type=Path, default=".")
  parser.add_argument("--tasks-file", type=Path)
  parser.add_argument("--include-proof-relevant", action="store_true")
  parser.add_argument("--state-dir", type=Path, default="drl_state")
  parser.add_argument("--curriculum", action='store_true')
  parser.add_argument("--no-interleave", action='store_false', dest='interleave')
  parser.add_argument("--starting-epsilon", default=0, type=float)
  parser.add_argument("--ending-epsilon", default=1.0, type=float)
  parser.add_argument("--smoothing-factor", default=4.0, type=float)
  parser.add_argument("--no-set-switch", dest="set_switch", action='store_false')
  parser.add_argument("-s", "--steps-per-episode", type=int)
  parser.add_argument("-n", "--num-episodes", type=int)
  parser.add_argument("-p", "--num_predictions", type=int)
  parser.add_argument("--optimizer", choices=rl.optimizers.keys(),
                      default=list(rl.optimizers.keys())[0])
  parser.add_argument("--blacklist-tactic", action="append",
                      dest="blacklisted_tactics")
  parser.add_argument("--backend", choices=["serapi", "lsp", "auto"],
                      default='auto')
  parser.add_argument("--max-sertop-workers", type=int, default=16)
  parser.add_argument("--coq2vec-weights", type=Path)
  parser.add_argument("--supervised-weights", type=Path, dest="weightsfile")
  parser.add_argument("-v", "--verbose", action='count', default=0)
  parser.add_argument("-t", "--print-timings", action='store_true')
  parser.add_argument("-w", "--workerid", required=True)
  args = parser.parse_args()

  workerid = args.workerid
  # assert 'SLURM_ARRAY_TASK_ID' in environ
  # workerid = int(environ['SLURM_ARRAY_TASK_ID'])

  with (args.state_dir / "actors_scheduled.txt").open('a') as f, FileLock(f):
    print(workerid, file=f, flush=True)

  if args.filenames[0].suffix == '.json':
    args.splits_file = args.filenames[0]
    args.filenames = None
  else:
    args.splits_file = None
  args.proofs_file = None
  args.proof = None

  reinforcement_act(args, workerid)
  eprint("Done, exiting")
  sys.exit(0)

def reinforcement_act(args: argparse.Namespace, workerid: int) -> None:
  learning_connection = LearningServerConnection(
    args.coq2vec_weights)
  task_eps = drl.get_all_task_episodes(args)
  actor = initialize_actor(args)
  task_state = TaskState(task_eps)
  all_files = task_state.all_files()
  while True:
    next_task_ep = allocate_next_task(args, workerid, task_state)
    if next_task_ep is None:
      eprint(f"Finished worker {workerid}")
      break
    if args.verbose > 0:
      eprint(f"Starting task ep {next_task_ep}")
    next_task, next_ep = next_task_ep
    cur_epsilon = compute_cur_epsilon(args, all_files, len(task_eps))
    successful, samples = actor.run_task_reinforce(
      next_task, cur_epsilon)
    if next_ep == 0:
      learning_connection.encode_and_send_target_length(samples[0][0],
                                                        next_task.target_length)
    if successful and len(samples) < next_task.target_length:
      update_shorter_proofs_dict(args, all_files, next_task, len(samples),
                                 samples[0][0],
                                 learning_connection)
    for prestate, action, poststates in samples:
      learning_connection.encode_and_send_sample(prestate, action, poststates)
    mark_task_done(args, workerid, next_task_ep)
    load_latest_q_network(args, actor.v_network)

class RLActor:
  args: argparse.Namespace
  predictor: TacticPredictor
  v_network: rl.VNetwork
  file_workers: OrderedDict[str, rl.FileReinforcementWorker]

  def __init__(self, args: argparse.Namespace,
               predictor: TacticPredictor,
               v_network: rl.VNetwork) -> None:
    self.args = args
    self.predictor = predictor
    self.v_network = v_network
    self.file_workers: OrderedDict[str, rl.FileReinforcementWorker] \
      = OrderedDict()

  def _get_worker(self, filename: str) -> rl.FileReinforcementWorker:
    if filename not in self.file_workers:
      args_copy = argparse.Namespace(**vars(self.args))
      args_copy.verbose = self.args.verbose - 2
      worker = rl.FileReinforcementWorker(args_copy, None)
      worker.enter_instance()
      self.file_workers[filename] = worker
      self.file_workers.move_to_end(filename)
    if len(self.file_workers) > self.args.max_sertop_workers:
      filename, evicted_worker = self.file_workers.popitem(last=False)
      evicted_worker.coq.kill()
    return self.file_workers[filename]
  def run_task_reinforce(self, task: RLTask, epsilon: float,
                         restart: bool = True) -> Tuple[bool, List[Tuple[Obligation, str, List[Obligation]]]]:
    if not rl.tactic_prefix_is_usable(task.tactic_prefix):
      if self.args.verbose >= 2:
        eprint(f"Skipping job {task.to_job()} with prefix {task.tactic_prefix} "
               "because it can't purely focused")
      else:
        eprint("Skipping a job because it can't be purely focused")
      return False, []
    with print_time("Getting worker", guard=self.args.print_timings):
      file_worker = self._get_worker(task.src_file)
    assert file_worker.coq is not None
    try:
      with print_time("Running into task", guard=self.args.print_timings):
        file_worker.run_into_task(task.to_job(), task.tactic_prefix)
      with print_time("Experienceing", guard=self.args.print_timings):
        return experience_proof(self.args, file_worker.coq,
                                self.predictor, self.v_network, epsilon)
    except coq_serapy.CoqAnomaly:
      eprint("Encountered Coq anomaly.")
      file_worker.restart_coq()
      file_worker.reset_file_state()
      file_worker.enter_file(task.src_file)
      if restart:
        return self.run_task_reinforce(task, epsilon, restart=False)
      eprint("Encountered anomaly without restart, closing current job")
    return False, []

TaskEpisode = Tuple[RLTask, int]
IndexedTaskEpisode = Tuple[int, TaskEpisode]
def task_eps_as_dict(task_episodes: List[TaskEpisode]) \
      -> Dict[Path, List[IndexedTaskEpisode]]:
  file_all_tes_dict: Dict[Path, List[IndexedTaskEpisode]] = {}
  for task_ep_idx, (task, episode) in enumerate(task_episodes):
    if Path(task.src_file) in file_all_tes_dict:
      file_all_tes_dict[Path(task.src_file)].append((task_ep_idx, (task, episode)))
    else:
      file_all_tes_dict[Path(task.src_file)] = [(task_ep_idx, (task, episode))]
  return file_all_tes_dict

class LearningServerConnection:
  obligation_encoder: coq2vec.CoqContextVectorizer
  def __init__(self, coq2vec_weights: Path, backend='mpi') -> None:
    term_encoder = coq2vec.CoqTermRNNVectorizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    term_encoder.load_state(torch.load(str(coq2vec_weights), map_location=device))
    num_hyps = 5
    self.obligation_encoder = rl.CachedObligationEncoder(term_encoder, num_hyps)
    eprint("Establishing connection")
    dist.init_process_group(backend)
    eprint("Connection Initialized")
    pass
  def send_sample(self, pre_state_encoded: torch.FloatTensor,
                  action_encoded: int,
                  post_state_encodeds: List[torch.FloatTensor]) -> None:
    dist.send(tensor=torch.tensor(0, dtype=int), tag=4, dst=0)
    dist.send(tensor=pre_state_encoded, tag=0, dst=0)
    dist.send(tensor=torch.LongTensor([action_encoded]), tag=1, dst=0)
    dist.send(tensor=torch.LongTensor([len(post_state_encodeds)]), tag=2, dst=0)
    for state in post_state_encodeds:
      dist.send(tensor=state, tag=3, dst=0)
  def encode_and_send_sample(self, pre_state: Obligation,
                             action: str,
                             post_states: List[Obligation]) -> None:
    states_encoded = self.obligation_encoder.obligations_to_vectors_cached(
      [pre_state] + post_states)
    self.send_sample(states_encoded[0], hash(action), states_encoded[1:])

  def send_target_v(self, state_encoded: torch.FloatTensor, target_length: int) -> None:
    dist.send(tensor=torch.tensor(1, dtype=int), tag=4, dst=0)
    dist.send(tensor=state_encoded, tag=5, dst=0)
    dist.send(tensor=torch.tensor(target_length, dtype=int), tag=6, dst=0)

  def encode_and_send_target_length(self, state: Obligation, target_length: int) -> None:
    state_encoded = self.obligation_encoder.obligations_to_vectors_cached([state])[0]
    self.send_target_v(state_encoded, target_length)

@dataclass
class FileTaskState:
  all_files: List[Path]
  filename: Path
  file_task_episodes: List[IndexedTaskEpisode]
  our_taken_task_episodes: Set[int]
  cur_episode: int
  skip_taken_proofs: bool

@dataclass
class TaskState:
  file_all_tes_dict: Dict[Path, List[IndexedTaskEpisode]]
  file_our_taken_dict: Dict[Path, Set[int]]
  files_finished_this_ep: Set[Path]
  cur_episode: int
  skip_taken_proofs: bool

  def __init__(self, task_eps: List[TaskEpisode]) -> None:
    self.file_all_tes_dict = task_eps_as_dict(task_eps)
    self.file_our_taken_dict: Dict[Path, List[IndexedTaskEpisode]] = {}
    self.files_finished_this_ep = set()
    self.cur_episode = 0
    self.skip_taken_proofs = True

  def all_files(self) -> List[Path]:
    return list(self.file_all_tes_dict.keys())

  def file_task_state(self, filename: Path) -> FileTaskState:
    return FileTaskState(self.all_files(), filename,
                         self.file_all_tes_dict[filename],
                         self.file_our_taken_dict.get(filename, set()),
                         self.cur_episode, self.skip_taken_proofs)

def allocate_next_task(args: argparse.Namespace, workerid: int,
                       task_state: TaskState) \
      -> Optional[TaskEpisode]:
  all_files = task_state.all_files()
  while True:
    # We put this in a while loop because we might go through all the work of
    # allocating a file, only to find out there are no eligible tasks in that
    # file. In that case, we want to mark that file as finished this episode, and
    # start the file allocating process over again.
    while True:
      # First thing we do to allocate a file is open and take a lock on the
      # "taken-files.txt" file. This tells us what files have been taken by other
      # workers.
      with (args.state_dir / "taken" / "taken-files.txt"
            ).open("r+") as taken_files_handle, FileLock(taken_files_handle):
        # Since multiple workers can be working on a file, we'll count here how
        # many workers have taken each file. At first we only care about whether
        # the file is taken by a non-zero number of workers, but once we run out
        # of unclaimed and our own files, we'll care about which file was taken
        # the least number of times.
        taken_files = Counter(Path(p.strip()) for p in taken_files_handle)
        # If we've looked at every file this episode, either we're on our last
        # episode and we're done, or there was only one file. In the latter case
        # we might still have episodes to work on, so we'll break to our
        # episode/phase incrementing code at the end of the function.
        if len(all_files) == len(task_state.files_finished_this_ep):
          assert task_state.cur_episode == args.num_episodes - 1 or len(all_files) == 1
          break
        # Get the number of workers taking the least-taken file.
        least_taken_count = min(taken_files[filename] for filename in all_files
                                if filename not in task_state.files_finished_this_ep)
        cur_file: Optional[Path] = None
        for src_file in all_files:
          # Skip files that we know we already finished with for now.
          if src_file in task_state.files_finished_this_ep:
            continue
          # There are three cases where we'll accept a file as eligible:
          # 1. We've already started working on it (but haven't finished it yet)
          # 2. It hasn't been taken by any other worker
          # 3. We've already finished all files that we initially claimed
          #    or were unclaimed, and it's the least-taken file.
          #
          # The condition in 3 is a bit tricky, because the condition doesn't
          # explicitly say that we've finished all files that would satisfy
          # condition (2). But if there is a file that satisfies condition (2),
          # then the least_taken_count will be 0, making condition (3) just a
          # stricter version of (2).
          if (src_file in task_state.file_our_taken_dict.keys() or
              taken_files[src_file] == 0 or
              (all(filename in task_state.files_finished_this_ep
                   for filename in task_state.file_our_taken_dict.keys()) and
               task_state.cur_episode == args.num_episodes - 1 and
               taken_files[src_file] == least_taken_count)):
             cur_file = src_file
             break
        # If a new file to claim was found, and we hadn't worked on it before,
        # add a claim to it for this worker.
        if (cur_file is not None and
            cur_file not in task_state.file_our_taken_dict.keys()):
          print(cur_file, file=taken_files_handle, flush=True)
      # If we came out of here without a file, we either need to increment our
      # episode/skip_taken, or we're done
      if cur_file is None:
        break
      # Allocate from the file in question.
      next_te_and_idx = allocate_next_task_from_file_with_retry(
        args, task_state.file_task_state(cur_file))
      # If there was a task in that file, add this file and task to our taken
      # dict, write it to our taken file, and return it.
      if next_te_and_idx is not None:
        task_index, (task, episode) = next_te_and_idx
        src_path = Path(task.src_file)
        if src_path in task_state.file_our_taken_dict:
          task_state.file_our_taken_dict[src_path].add(task_index)
        else:
          task_state.file_our_taken_dict[src_path] = {task_index}
        with (args.state_dir / "taken" / f"taken-{workerid}.txt").open('a') as f:
          print(json.dumps((task.as_dict(), episode)), file=f, flush=True)
        return (task, episode)
      # Otherwise if we don't find a task in that file, then we need to mark it
      # as finished for this episode/phase and try the next one.
      eprint(f"Couldn't find an available task for file {cur_file}, "
             f"trying next file...",
             guard=args.verbose >= 2)
      task_state.files_finished_this_ep.add(cur_file)
    # If we leave the loop without returning, then it's time to increment our
    # phase (cur_episode + skip_taken_proofs)
    if task_state.cur_episode == args.num_episodes - 1:
      # If we're on the last episode, and skip_taken_proofs is already false,
      # then we're on the last phase, and there's nothing to return.
      if not task_state.skip_taken_proofs:
        return None
      # If we're on the last episode but we're still skipping tasks within taken
      # proofs, we can now start sharing proofs with others.
      task_state.skip_taken_proofs = False
    else:
      # If we're not on the last episode yet, increment the episode
      task_state.cur_episode += 1
    # If we incremented the phase at all, then we have a new chance to make
    # progress on every file, so clear the set of files finished this episode.
    task_state.files_finished_this_ep = set()

def allocate_next_task_from_file_with_retry(
      args: argparse.Namespace,
      file_task_state: FileTaskState,
      num_retries: int = 3) -> Optional[IndexedTaskEpisode]:
  for i in range(num_retries):
    try:
      return allocate_next_task_from_file(args, file_task_state)
    except json.decoder.JSONDecodeError:
      if i == num_retries - 1:
        raise
      eprint("Failed to decode, retrying", guard=args.verbose >= 1)
  assert False
  return None

ProofSpec = Tuple[str, str, str]
ProofEpisodeSpec = Tuple[ProofSpec, int]

def allocate_next_task_from_file(args: argparse.Namespace,
                                 file_task_state: FileTaskState) \
      -> Optional[IndexedTaskEpisode]:
  filepath = args.state_dir / "taken" / \
    ("file-" + safe_abbrev(file_task_state.filename,
                           file_task_state.all_files) + ".txt")
  with filepath.open("r+") as f, FileLock(f):
    # We need to keep track of two seperate sets of taken tasks. The ones that
    # were already taken at all by anyone ever, which we don't want to take.
    # And the ones that were taken by another worker, because we don't want to
    # take other tasks in the same proofs as those unless skip_taken is False.
    taken_task_episodes: Set[int] = set()
    taken_by_others_this_iter: Set[int] = set()
    for line in f:
      te_idx, taken_this_iter = json.loads(line)
      taken_task_episodes.add(te_idx)
      # We know it's currently taken by another worker if it was taken this
      # iteration (not from before a resume) and we don't have it on record as
      # taken by us, but it's in the taken file.
      if taken_this_iter and te_idx not in file_task_state.our_taken_task_episodes:
        taken_by_others_this_iter.add(te_idx)
    # Now we need to keep track of what proofs have at least one task that
    # another worker is working on or completed this iteration.
    proof_eps_taken_by_others: Set[ProofEpisodeSpec] = set()
    # Go through every task in this file looking for one to work on
    for task_ep_idx, (task, episode) in file_task_state.file_task_episodes:
      # If interleave is true (the default), then tasks are strictly ordered
      # based on their episodes. So as soon as we see one with too high of an
      # episode, we know all later tasks will be ineligable, so we can quit and
      # return None.
      if episode > file_task_state.cur_episode:
        if args.interleave:
          return None
        else:
	  # If iterleave is false, then we might later encounter a task with a
	  # lower episode, so skip this task but keep looking.
          continue
      # If the episode was taken by someone else since the last resume, skip
      # it, but also mark the whole proof as being worked on by another worker.
      if task_ep_idx in taken_by_others_this_iter:
        proof_eps_taken_by_others.add((task.to_proof_spec(), episode))
        continue
      # If the task episode was taken by us, taken by someone before the last
      # resume, or in a proof that has another task taken by someone since the
      # last resume, then skip it.
      if (task_ep_idx in taken_task_episodes
          or ((task.to_proof_spec(), episode) in proof_eps_taken_by_others
              and file_task_state.skip_taken_proofs)):
         continue
      # Otherwise, we've found an eligible task, so mark it as claimed in the
      # file and return it.
      print(json.dumps((task_ep_idx, True)), file=f, flush=True)
      return task_ep_idx, (task, episode)
  # If we get to the end of the loop, then every task was ineligible, so return
  # None so that we can look at another file or finish the worker.
  return None

def compute_cur_epsilon(args: argparse.Namespace, all_files: List[Path],
                        num_task_eps: int) -> float:
  return args.starting_epsilon + \
    ((get_num_tasks_taken(args, all_files)
     / num_task_eps)) * (args.ending_epsilon - args.starting_epsilon)

def get_num_tasks_taken(args: argparse.Namespace, all_files: List[Path]) -> int:
    tasks_taken = 0
    for filename in all_files:
        with (args.state_dir / "taken" /
              ("file-" + safe_abbrev(filename, all_files) + ".txt")).open("r") as f:
            tasks_taken += sum(1 for _ in f)
    return tasks_taken

def mark_task_done(args: argparse.Namespace, workerid: int,
                   done_task_ep: TaskEpisode) -> None:
  with (args.state_dir / f"done-{workerid}.txt").open('a') as f, FileLock(f):
    task, ep = done_task_ep
    print(json.dumps((task.as_dict(), ep)), file=f, flush=True)

def update_shorter_proofs_dict(args: argparse.Namespace,
                               all_files: List[Path],
                               task: RLTask,
                               solution_length: int,
                               starting_state: Obligation,
                               server_connection: LearningServerConnection) -> None:
  with (args.state_dir / "shorter_proofs" /
        (safe_abbrev(Path(task.src_file), all_files) + ".json")
        ).open("r+") as shorter_proofs_handle, FileLock(shorter_proofs_handle):
    shorter_proofs_dict = {RLTask(**task_dict): shorter_length
                           for l in shorter_proofs_handle
                           for task_dict, shorter_length in (json.loads(l),)}
    if task in shorter_proofs_dict and \
      shorter_proofs_dict[task] <= solution_length:
      return
    shorter_proofs_dict[task] = solution_length
    shorter_proofs_handle.truncate()
    for task, shorter_length in shorter_proofs_dict.items():
      print(json.dumps((task.as_dict(), shorter_length)), file=shorter_proofs_handle)

  server_connection.encode_and_send_target_length(starting_state, solution_length)

def initialize_actor(args: argparse.Namespace) \
    -> RLActor:
  predictor = rl.MemoizingPredictor(get_predictor(args))
  network = rl.VNetwork(args.coq2vec_weights, 0.0,
                        1, 1, args.optimizer)
  load_latest_q_network(args, network)
  return RLActor(args, predictor, network)

def load_latest_q_network(args: argparse.Namespace, v_network: rl.VNetwork) -> None:
  root_dir = str(args.state_dir / "weights")
  current_working_directory = os.getcwd()
  os.chdir(root_dir)
  q_networks = glob("common-v-network-*.dat")
  os.chdir(current_working_directory)

  if len(q_networks) == 0:
    return
  q_network_save_nums = [
      int(unwrap(re.match(r"common-v-network-(\d+).dat", path)).group(1))
      for path in q_networks]
  newest_index = max(q_network_save_nums)
  latest_q_network_path = str(
    args.state_dir / "weights" /
    f"common-v-network-{newest_index}.dat")
  eprint(f"Loading latest q network from {latest_q_network_path}")
  q_network_state = torch.load(latest_q_network_path, map_location="cpu")
  v_network.network.load_state_dict(q_network_state)

def experience_proof(args: argparse.Namespace,
                     coq: coq_serapy.CoqAgent,
                     predictor: TacticPredictor,
                     v_network: rl.VNetwork,
                     epsilon: float) \
      -> Tuple[bool, List[Tuple[Obligation, str, List[Obligation]]]]:
  path: List[ProofContext] = [coq.proof_context]
  initial_open_obligations = len(coq.proof_context.all_goals)
  samples: List[Tuple[Obligation, str, List[Obligation]]] = []

  for step in range(args.steps_per_episode):
    before_obl = unwrap(coq.proof_context).fg_goals[0]
    if args.verbose >= 3:
      coq_serapy.summarizeContext(coq.proof_context)
    actions = predictor.predictKTactics(
      truncate_tactic_context(FullContext(coq.local_lemmas[:-1],
                                          coq.prev_tactics,
                                          unwrap(coq.proof_context)).as_tcontext(),
                              30),
      args.num_predictions,
      blacklist = args.blacklisted_tactics)
    eprint(f"Using predictions {[action.prediction for action in actions]}",
           guard=args.verbose >= 3)
    if random.random() < epsilon:
      eprint("Using best-scoring action", guard=args.verbose >= 3)
      action_scores = rl.evaluate_actions(
        coq, v_network, path,
        [action.prediction for action in actions])
      chosen_action, chosen_score = max(zip(actions, action_scores),
                                        key=lambda p: p[1])
      if chosen_score == -float("Inf"):
        break
    else:
      eprint("Using random action", guard=args.verbose >=3)
      chosen_action = None
      for action in random.sample(actions, k=len(actions)):
        try:
          coq.run_stmt(action.prediction)
          resulting_context = coq.proof_context
          coq.cancel_last_noupdate()
          if any(coq_serapy.contextSurjective(resulting_context, path_context)
                 for path_context in path):
            continue
          chosen_action = action
          break
        except (coq_serapy.CoqTimeoutError, coq_serapy.ParseError,
                coq_serapy.CoqExn, coq_serapy.CoqOverflowError,
                RecursionError,
                coq_serapy.UnrecognizedError) as e:
          eprint(f"Action produced error {e}", guard=args.verbose >= 3)
      if chosen_action is None:
        break
    resuting_obls = rl.execute_action(coq, chosen_action.prediction)
    eprint(f"Taking action {chosen_action}",
           guard=args.verbose >= 2)
    if args.verbose >= 3:
      coq_serapy.summarizeContext(coq.proof_context)
    path.append(coq.proof_context)
    samples.append((before_obl, chosen_action.prediction, resuting_obls))
    if len(coq.proof_context.all_goals) < initial_open_obligations:
      eprint(f"Completed task with trace {[sample[1] for sample in samples]}")
      return True, samples
    assert len(coq.proof_context.all_goals) > 0
    assert len(coq.proof_context.fg_goals) > 0
  return False, samples
if __name__ == "__main__":
  main()
