#!/usr/bin/env python

import argparse
import json
import random
import time
import contextlib
import math
from pathlib import Path
from typing import (List, Optional, Dict, Tuple, Union, Any, Set,
                    Sequence, TypeVar, Callable)

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import torch.optim.lr_scheduler as scheduler

from tqdm import tqdm

import coq_serapy
from coq_serapy.contexts import (FullContext, truncate_tactic_context,
                                 Obligation, TacticContext, ProofContext)
import coq2vec

from search_file import get_all_jobs
from search_worker import ReportJob, Worker, get_predictor
from search_strategies import completed_proof

from models.tactic_predictor import (TacticPredictor, Prediction)

from util import unwrap, eprint, print_time, nostderr

def main():
    parser = argparse.ArgumentParser(
        description="Train a state estimator using reinforcement learning"
        "to complete proofs using Proverbot9001.")
    parser.add_argument("--prelude", default=".", type=Path)
    parser.add_argument("--output", "-o", dest="output_file",
                        help="output data folder name",
                        default="data/rl_weights.dat",
                        type=Path)
    parser.add_argument("--verbose", "-v", help="verbose output",
                        action="count", default=0)
    parser.add_argument("--progress", "-P", help="show progress of files",
                        action='store_true')
    parser.add_argument("--no-set-switch", dest="set_switch", action='store_false')
    parser.add_argument("--include-proof-relevant", action="store_true")
    parser.add_argument("--backend", choices=['serapi', 'lsp', 'auto'], default='auto')
    parser.add_argument('filenames', help="proof file name (*.v)",
                        nargs='+', type=Path)
    proofsGroup = parser.add_mutually_exclusive_group()
    proofsGroup.add_argument("--proof", default=None)
    proofsGroup.add_argument("--proofs-file", default=None)

    parser.add_argument("--no-interleave", dest="interleave", action="store_false")
    parser.add_argument('--supervised-weights', type=Path, dest="weightsfile")
    parser.add_argument("--coq2vec-weights", type=Path)
    parser.add_argument("-l", "--learning-rate", default=2.5e-4, type=float)
    parser.add_argument("-g", "--gamma", default=0.9, type=float)
    parser.add_argument("--starting-epsilon", default=0, type=float)
    parser.add_argument("--ending-epsilon", default=1.0, type=float)
    parser.add_argument("-s", "--steps-per-episode", default=16, type=int)
    parser.add_argument("-n", "--num-episodes", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-w", "--window-size", default=2560)
    parser.add_argument("-p", "--num-predictions", default=5, type=int)
    parser.add_argument("--batch-step", default=25, type=int)
    parser.add_argument("--lr-step", default=0.8, type=float)
    parser.add_argument("--batches-per-proof", default=1, type=int)
    parser.add_argument("--print-loss-every", default=None, type=int)
    parser.add_argument("--sync-target-every",
                        help="Sync target network to v network every <n> episodes",
                        default=10, type=int)
    parser.add_argument("--allow-partial-batches", action='store_true')
    parser.add_argument("--blacklist-tactic", action="append", dest="blacklisted_tactics")
    parser.add_argument("--resume", choices=["no", "yes", "ask"], default="ask")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    if args.filenames[0].suffix == ".json":
        args.splits_file = args.filenames[0]
        args.filenames = None
    else:
        args.splits_file = None

    reinforce_jobs(args)

class FileReinforcementWorker(Worker):
    def finish_proof(self) -> None:
        while not coq_serapy.ending_proof(self.remaining_commands[0]):
            self.remaining_commands.pop(0)
        ending_cmd = self.remaining_commands.pop(0)
        coq_serapy.admit_proof(self.coq, self.coq.prev_tactics[0], ending_cmd)

class ReinforcementWorker(Worker):
    v_network: 'VNetwork'
    target_v_network: 'VNetwork'
    replay_buffer: 'ReplayBuffer'
    verbosity: int
    file_workers: Dict[str, FileReinforcementWorker]
    def __init__(self, args: argparse.Namespace,
                 predictor: TacticPredictor,
                 v_network: 'VNetwork',
                 target_network: 'VNetwork',
                 switch_dict: Optional[Dict[str, str]] = None,
                 initial_replay_buffer: Optional['ReplayBuffer'] = None) -> None:
        self.original_args = args
        args_copy = argparse.Namespace(**vars(args))
        args_copy.verbose = args.verbose - 2
        super().__init__(args_copy, 0, predictor, switch_dict)
        self.v_network = v_network
        self.target_v_network = target_network
        if initial_replay_buffer:
            self.replay_buffer = initial_replay_buffer
        else:
            self.replay_buffer = ReplayBuffer(args.window_size,
                                              args.allow_partial_batches)
        self.verbosity = args.verbose
        self.file_workers = {}
    def run_job_reinforce(self, job: ReportJob, epsilon: float, restart: bool = True) -> None:
        if job.filename not in self.file_workers:
            self.file_workers[job.filename] = FileReinforcementWorker(self.args, 0,
                                                                      None, None)
            self.file_workers[job.filename].enter_instance()
        job_name = job.module_prefix + coq_serapy.lemma_name_from_statement(
            job.lemma_statement)
        # with print_time(f"Entering job {job_name}", guard=self.verbosity >= 1):
        self.file_workers[job.filename].run_into_job(job, restart, False)
        # with print_time("Experiencing proof", guard=self.verbosity >= 1):
        try:
            experience_proof(self.original_args,
                             self.file_workers[job.filename].coq,
                             self.predictor, self.v_network,
                             self.replay_buffer, epsilon)
            self.file_workers[job.filename].finish_proof()
        except coq_serapy.CoqAnomaly:
            self.file_workers[job.filename].restart_coq()
            self.file_workers[job.filename].enter_file(job.filename)
        train_v_network(self.original_args, self.v_network, self.target_v_network,
                        self.replay_buffer)
    def sync_networks(self) -> None:
        self.target_v_network.network.load_state_dict(self.v_network.network.state_dict())


def reinforce_jobs(args: argparse.Namespace) -> None:
    if args.splits_file:
        with args.splits_file.open('r') as f:
            project_dicts = json.loads(f.read())
        if any("switch" in item for item in project_dicts):
            switch_dict = {item["project_name"]: item["switch"]
                           for item in project_dicts}
        else:
            switch_dict = None
    else:
        switch_dict = None
    # predictor = get_predictor(args)
    predictor = MemoizingPredictor(get_predictor(args))
    # predictor = DummyPredictor()
    if args.resume == "ask" and args.output_file.exists():
        print(f"Found existing weights at {args.output_file}. Resume?")
        response = input("[Y/n] ")
        if response.lower() not in ["no", "n"]:
            args.resume = "yes"
        else:
            args.resume = "no"
    elif not args.output_file.exists():
        args.resume = "no"

    if args.resume == "yes":
        replay_buffer, steps_already_done, network_state, tnetwork_state, random_state = \
            torch.load(str(args.output_file))
        random.setstate(random_state)
        print(f"Resuming from existing weights of {steps_already_done} steps")
        v_network = VNetwork(None, args.learning_rate,
                             args.batch_step, args.lr_step)
        target_network = VNetwork(None, args.learning_rate,
                                  args.batch_step, args.lr_step)
        v_network.load_state(network_state)
        target_network.load_state(tnetwork_state)

    else:
        assert args.resume == "no"
        steps_already_done = 0
        replay_buffer = None
        v_network = VNetwork(args.coq2vec_weights, args.learning_rate,
                             args.batch_step, args.lr_step)
        target_network = VNetwork(args.coq2vec_weights, args.learning_rate,
                             args.batch_step, args.lr_step)

    jobs = get_all_jobs(args)

    with ReinforcementWorker(args, predictor, v_network, target_network, switch_dict,
                             initial_replay_buffer = replay_buffer) as worker:
        if args.interleave:
            tasks = jobs * args.num_episodes
        else:
            tasks = [task for job in jobs for task in [job] * args.num_episodes]

        for step in range(steps_already_done):
            with nostderr():
                worker.v_network.adjuster.step()

        for step, task in enumerate(tqdm(tasks[steps_already_done:]),
                                    start=steps_already_done+1):
            cur_epsilon = args.starting_epsilon + ((step / len(tasks)) * (args.ending_epsilon - args.starting_epsilon))
            worker.run_job_reinforce(task, cur_epsilon)
            if (step + 1) % args.save_every == 0:
                save_state(args, worker, step + 1)
            if (step + 1) % args.sync_target_every == 0:
                worker.sync_networks()
        if steps_already_done < len(tasks):
            save_state(args, worker, step)
        if args.evaluate:
            evaluate_results(args, worker, jobs)

class CachedObligationEncoder(coq2vec.CoqContextVectorizer):
    obl_cache: Dict[Obligation, torch.FloatTensor]
    def __init__(self, term_encoder: 'CoqTermRNNVectorizer',
                 max_num_hypotheses: int) -> None:
        super().__init__(term_encoder, max_num_hypotheses)
        self.obl_cache = {}
    def obligations_to_vectors_cached(self, obls: List[Obligation]) -> torch.FloatTensor:
        encoded_obl_size = self.term_encoder.hidden_size * (self.max_num_hypotheses + 1)
        encoded = run_network_with_cache(
            lambda x: self.obligations_to_vectors(x).view(len(x), encoded_obl_size),
            [coq2vec.Obligation(list(obl.hypotheses), obl.goal) for obl in obls],
            [self.obl_cache.get(obl, None) for obl in obls])
        # for row, obl in zip(encoded, obls):
        #     assert obl not in self.obl_cache or (self.obl_cache[obl] == row).all(), \
        #         (self.obl_cache[obl] == row)
        for row, obl in zip(encoded, obls):
            self.obl_cache[obl] = row
        return encoded

class VNetwork:
    obligation_encoder: Optional[coq2vec.CoqContextVectorizer]
    network: nn.Module
    steps_trained: int
    optimizer: optim.Optimizer
    batch_step: int
    lr_step: int
    learning_rate: float
    def get_state(self) -> Any:
        return (self.network.state_dict(),
                self.obligation_encoder.term_encoder.get_state())

    def _load_encoder_state(self, encoder_state: Any) -> None:
        term_encoder = coq2vec.CoqTermRNNVectorizer()
        term_encoder.load_state(encoder_state)
        num_hyps = 5
        assert self.obligation_encoder is None, "Can't load weights twice!"
        self.obligation_encoder = CachedObligationEncoder(term_encoder, num_hyps)
        insize = term_encoder.hidden_size * (num_hyps + 1)
        self.network = nn.Sequential(
            nn.Linear(insize, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1),
        )
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate)
        self.adjuster = scheduler.StepLR(self.optimizer, self.batch_step,
                                         self.lr_step)


    def load_state(self, state: Any) -> None:
        network_state, encoder_state = state
        self._load_encoder_state(encoder_state)
        self.network.load_state_dict(network_state)


    def __init__(self, coq2vec_weights: Optional[Path], learning_rate: float,
                 batch_step: int, lr_step: float) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_step = batch_step
        self.lr_step = lr_step
        self.learning_rate = learning_rate
        self.obligation_encoder = None
        self.network = None
        # Steps trained only counts from the last resume! Don't use
        # for anything more essential than printing.
        self.steps_trained = 0
        self.total_loss = torch.tensor(0.0)
        if coq2vec_weights is not None:
            self._load_encoder_state(torch.load(coq2vec_weights, map_location=device))

    def __call__(self, obls: Union[Obligation, List[Obligation]]) -> torch.FloatTensor:
        assert self.obligation_encoder, \
            "No loaded encoder! Pass encoder weights to __init__ or call set_state()"
        if isinstance(obls, Obligation):
            obls = [obls]
        else:
            assert isinstance(obls, list)
            if len(obls) == 0:
                return torch.tensor([])

        encoded = self.obligation_encoder.\
            obligations_to_vectors_cached(obls)
        scores = self.network(encoded).view(len(obls))
        return scores

    def train(self, inputs: List[Obligation],
              target_outputs: List[float],
              verbosity: int = 0) -> float:
        assert self.obligation_encoder, \
            "No loaded encoder! Pass encoder weights to __init__ or call set_state()"
        # with print_time("Training", guard=verbosity >= 1):
        actual = self(inputs)
        target = torch.FloatTensor(target_outputs)
        loss = F.mse_loss(actual, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.adjuster.step()
        # eprint(f"Actual: {actual}; Target: {target}", guard=verbosity >= 2)
        # eprint(f"Loss: {loss}", guard=verbosity >= 1)
        self.steps_trained += 1
        self.total_loss += loss
        return loss
def experience_proof(args: argparse.Namespace,
                     coq: coq_serapy.CoqAgent,
                     predictor: TacticPredictor,
                     v_network: VNetwork,
                     replay_buffer: 'ReplayBuffer',
                     epsilon: float) -> None:
    path: List[ProofContext] = [coq.proof_context]
    for _step in range(args.steps_per_episode):
        before_obl = unwrap(coq.proof_context).fg_goals[0]
        if args.verbose >= 3:
            coq_serapy.summarizeContext(coq.proof_context)
        actions = predictor.predictKTactics(
            truncate_tactic_context(FullContext(coq.local_lemmas,
                                                coq.prev_tactics,
                                                unwrap(coq.proof_context)).as_tcontext(),
                                    30),
            args.num_predictions,
            blacklist=args.blacklisted_tactics)
        eprint(f"Using predictions {[action.prediction for action in actions]}",
               guard=args.verbose >= 3)
        if random.random() < epsilon:
            eprint("Using best-scoring action", guard=args.verbose >= 3)
            action_scores = evaluate_actions(
                coq, v_network, path,
                [action.prediction for action in actions])
            chosen_action, chosen_score = max(zip(actions, action_scores),
                                              key=lambda p: p[1])
            if chosen_score == -float("Inf"):
                break
        else:
            eprint("Using random action", guard=args.verbose >= 3)
            chosen_action = None
            for action in random.sample(actions, k=len(actions)):
                try:
                    coq.run_stmt(action.prediction)
                    if any(coq_serapy.contextSurjective(coq.proof_context, path_context)
                           for path_context in path):
                        coq.cancel_last()
                        continue
                    coq.cancel_last()
                    chosen_action = action
                    break
                except (coq_serapy.CoqTimeoutError, coq_serapy.ParseError,
                        coq_serapy.CoqExn, coq_serapy.CoqOverflowError,
                        coq_serapy.ParseError,
                        RecursionError,
                        coq_serapy.UnrecognizedError):
                    pass
            if chosen_action is None:
                break
        resulting_obls = execute_action(coq, chosen_action.prediction)

        eprint(f"Taking action {chosen_action}",
               guard=args.verbose >= 2)

        if args.verbose >= 3:
            coq_serapy.summarizeContext(coq.proof_context)
        replay_buffer.add_transition((before_obl, chosen_action.prediction, resulting_obls))
        path.append(coq.proof_context)
        if completed_proof(coq):
            break
def evaluate_actions(coq: coq_serapy.CoqAgent,
                     v_network: VNetwork, path: List[ProofContext],
                     actions: List[str]) -> List[float]:
    resulting_contexts: List[Optional[ProofContext]] = \
        [action_result(coq, path, action) for action in actions]
    num_output_obls: List[int] = [len(context.fg_goals) if context else 0
                                   for context in resulting_contexts]
    all_obls = [obl for context in resulting_contexts
                for obl in (context.fg_goals if context else [])]
    all_obl_scores = v_network(all_obls)
    resulting_action_scores = []
    cur_obl_idx = 0
    for num_obls in num_output_obls:
        if num_obls == 0:
            resulting_action_scores.append(float("-Inf"))
        else:
            resulting_action_scores.append(math.prod(
                all_obl_scores[cur_obl_idx:cur_obl_idx+num_obls]))
            cur_obl_idx += num_obls
    return resulting_action_scores

def action_result(coq: coq_serapy.CoqAgent, path: List[ProofContext],
                  action: str) -> Optional[ProofContext]:
    try:
        coq.run_stmt(action)
    except (coq_serapy.CoqTimeoutError, coq_serapy.ParseError,
            coq_serapy.CoqExn, coq_serapy.CoqOverflowError,
            coq_serapy.ParseError,
            RecursionError,
            coq_serapy.UnrecognizedError):
        return None
    context_after = coq.proof_context
    coq.cancel_last()
    if any(coq_serapy.contextSurjective(context_after, path_context)
           for path_context in path):
        return None
    return context_after

def evaluate_action(args: argparse.Namespace, coq: coq_serapy.CoqAgent,
                    v_network: VNetwork, path: List[ProofContext], action: str) -> float:
    try:
        coq.run_stmt(action)
    except (coq_serapy.CoqTimeoutError, coq_serapy.ParseError,
            coq_serapy.CoqExn, coq_serapy.CoqOverflowError,
            coq_serapy.ParseError,
            RecursionError,
            coq_serapy.UnrecognizedError):
        return torch.tensor(-float("Inf"))

    context_after = coq.proof_context
    coq.cancel_last()

    if any(coq_serapy.contextSurjective(context_after, path_context)
           for path_context in path):
        return torch.tensor(-float("Inf"))


    with torch.no_grad():
        product = math.prod(v_network(unwrap(context_after).fg_goals))
    return product

def execute_action(coq: coq_serapy.CoqAgent,
                   action: str) -> List[Obligation]:

    coq.run_stmt(action)
    resulting_obls = unwrap(coq.proof_context).fg_goals

    subgoals_closed = 0
    if len(unwrap(coq.proof_context).fg_goals) == 0 and \
       len(unwrap(coq.proof_context).shelved_goals) > 0: # type: ignore
        coq.run_stmt("Unshelve.")
    while len(unwrap(coq.proof_context).fg_goals) == 0 \
            and not completed_proof(coq):
        coq.run_stmt("}")
        subgoals_closed += 1
    if coq.count_fg_goals() > 1 or \
       (coq.count_fg_goals() > 0 and subgoals_closed > 0):
        coq.run_stmt("{")

    return resulting_obls

def train_v_network(args: argparse.Namespace,
                    v_network: VNetwork,
                    target_network: VNetwork,
                    replay_buffer: 'ReplayBuffer'):
    for _batch_idx in range(args.batches_per_proof):
        samples = replay_buffer.sample(args.batch_size)
        if samples is None:
            return
        inputs = [start_obl for start_obl, action_records in samples]
        num_resulting_obls = [[len(resulting_obls)
                               for action, resulting_obls in action_records]
                              for starting_obl, action_records in samples]
        all_obls = [obl
                    for starting_obl, action_records in samples
                    for action, resulting_obls in action_records
                    for obl in resulting_obls]
        with torch.no_grad():
            all_obl_scores = target_network(all_obls)
        outputs = []
        cur_row = 0
        for resulting_obl_lens in num_resulting_obls:
            action_outputs = []
            for num_obls in resulting_obl_lens:
                selected_obl_scores = all_obl_scores[cur_row:cur_row+num_obls]
                action_outputs.append(args.gamma * math.prod(
                    selected_obl_scores))
                cur_row += num_obls
            outputs.append(max(action_outputs))
        v_network.train(inputs, outputs, verbosity=args.verbose)
        if args.print_loss_every and (v_network.steps_trained + 1) % args.print_loss_every == 0:
            avg_loss = v_network.total_loss / args.print_loss_every
            v_network.total_loss = torch.tensor(0.0)
            print(f"Loss: {avg_loss}; Learning rate: {v_network.optimizer.param_groups[0]['lr']}")

class MemoizingPredictor(TacticPredictor):
    underlying_predictor: TacticPredictor
    cache: Dict[Tuple[TacticContext, int, Optional[Sequence[str]]], List[Prediction]]
    def __init__(self, underlying_predictor: TacticPredictor) -> None:
        self.underlying_predictor = underlying_predictor
        self.cache = {}
    def getOptions(self) -> List[Tuple[str, str]]:
        raise NotImplementedError()
    def predictKTactics(self, in_data : TacticContext, k : int,
                        blacklist: Optional[List[str]] = None) \
        -> List[Prediction]:
        if in_data in self.cache:
            return self.cache[(in_data, k, tuple(blacklist) if blacklist else None)]
        predictions = self.underlying_predictor.predictKTactics(in_data, k, blacklist)
        self.cache[(in_data, k, tuple(blacklist) if blacklist else None)] = predictions
        return predictions
    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) -> \
        Tuple[List[Prediction], float]:
        raise NotImplementedError()
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, correct : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        raise NotImplementedError()

class DummyPredictor(TacticPredictor):
    def __init__(self) -> None:
        pass
    def getOptions(self) -> List[Tuple[str, str]]:
        raise NotImplementedError()

    def predictKTactics(self, in_data : TacticContext, k : int,
                        blacklist: Optional[List[str]] = None) \
        -> List[Prediction]:
        del blacklist
        del in_data
        del k
        return [Prediction("intro.", 0.25), Prediction("apply conj.", 0.25),
                Prediction("reflexivity.", 0.25), Prediction("simpl.", 0.25)]
    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) -> \
        Tuple[List[Prediction], float]:
        raise NotImplementedError()
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, correct : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        raise NotImplementedError()

def save_state(args: argparse.Namespace, worker: ReinforcementWorker,
               step: int) -> None:
    with args.output_file.open('wb') as f:
        torch.save((worker.replay_buffer, step,
                    worker.v_network.get_state(),
                    worker.target_v_network.get_state(),
                    random.getstate()), f)

def evaluate_results(args: argparse.Namespace,
                     worker: ReinforcementWorker,
                     jobs: List[ReportJob]) -> None:
    proofs_completed = 0
    for job in jobs:
        worker.run_into_job(job, True, False)
        path: List[ProofContext] = [worker.coq.proof_context]
        proof_succeeded = False
        for _step in range(args.steps_per_episode):
            actions = worker.predictor.predictKTactics(
                truncate_tactic_context(FullContext(
                    worker.coq.local_lemmas,
                    worker.coq.prev_tactics,
                    unwrap(worker.coq.proof_context)).as_tcontext(),
                                        30),
                args.num_predictions,
                blacklist=args.blacklisted_tactics)
            if args.verbose >= 1:
                coq_serapy.summarizeContext(worker.coq.proof_context)
            eprint(f"Trying predictions {[action.prediction for action in actions]}",
                   guard=args.verbose >= 2)
            action_scores = [evaluate_action(args, worker.coq, worker.v_network,
                                             path, action.prediction)
                             for action in actions]
            best_action, best_score = max(zip(actions, action_scores), key=lambda p: p[1])
            if best_score == -float("Inf"):
                break
            eprint(f"Taking action {best_action} with estimated value {best_score}",
                   guard=args.verbose >= 1)
            execute_action(worker.coq, best_action.prediction)
            path.append(worker.coq.proof_context)
            if completed_proof(worker.coq):
                proof_succeeded = True
                break
        proof_name = coq_serapy.lemma_name_from_statement(job.lemma_statement)
        while not coq_serapy.ending_proof(worker.remaining_commands[0]):
            worker.remaining_commands.pop(0)
        ending_command = worker.remaining_commands.pop(0)
        if proof_succeeded:
            eprint(f"Solved proof {proof_name}!")
            proofs_completed += 1
            worker.coq.run_stmt(ending_command)
        else:
            eprint(f"Failed to solve proof {proof_name}")
            coq_serapy.admit_proof(worker.coq, job.lemma_statement, ending_command)
    print(f"{proofs_completed} out of {len(jobs)} "
          f"theorems/lemmas successfully proven "
          f"({stringified_percent(proofs_completed, len(jobs))}%)")

def stringified_percent(total : float, outof : float) -> str:
    if outof == 0:
        return "NaN"
    return f"{(total * 100 / outof):10.2f}"

Transition = Tuple[str, Sequence[Obligation]]
FullTransition = Tuple[Obligation, str, List[Obligation]]

class ReplayBuffer:
    _contents: Dict[Obligation, Tuple[int, Set[Transition]]]
    window_size: int
    window_end_position: int
    allow_partial_batches: int
    def __init__(self, window_size: int,
                 allow_partial_batches: bool) -> None:
        self.window_size = window_size
        self.window_end_position = 0
        self.allow_partial_batches = allow_partial_batches
        self._contents = {}

    def sample(self, batch_size) -> Optional[List[Tuple[Obligation, Set[Transition]]]]:
        sample_pool: List[Tuple[Obligation, List[Transition]]] = []
        for obl, (last_updated, transitions) in self._contents.copy().items():
            if last_updated <= self.window_end_position - self.window_size:
                del self._contents[obl]
            else:
                sample_pool.append((obl, transitions))
        if len(sample_pool) >= batch_size:
            return random.sample(sample_pool, batch_size)
        if self.allow_partial_batches:
            return sample_pool
        return None

    def add_transition(self, transition: FullTransition) -> None:
        from_obl = transition[0]
        action = transition[1]
        to_obls = tuple(transition[2])
        self._contents[from_obl] = \
            (self.window_end_position,
             {(action, to_obls)} |
             self._contents.get(from_obl, (0, set()))[1])
        self.window_end_position += 1

# NOT THREAD SAFE
@contextlib.contextmanager
def log_time(msg: str) -> None:
    start = time.time()
    try:
        yield
    finally:
        time_taken = time.time() - start
        try:
            with open("timings.json", 'r') as f:
                timings = json.load(f)
        except FileNotFoundError:
            timings = {}
        timings[msg] = time_taken + timings.get(msg, 0.0)
        with open("timings.json", 'w') as f:
            json.dump(timings, f)

# This function takes a function which can run on multiple inputs in
# batch, a list of input values, and a list containing outputs for
# some of the values. It then calls the function on all the values
# that don't have an output, and puts combines their outputs with the
# given output list to create a non-None output value for every input,
# efficiently.
T = TypeVar('T')
def run_network_with_cache(f: Callable[List[T], torch.FloatTensor],
                           values: List[T],
                           cached_outputs: List[Optional[torch.FloatTensor]]) \
                           -> torch.FloatTensor:
    assert len(values) == len(cached_outputs)
    output_list: List[Optional[torch.FloatTensor]] = list(cached_outputs)
    uncached_values: List[T] = []
    uncached_value_indices: List[int] = []
    for i, value in enumerate(values):
        if output_list[i] is None:
            uncached_values.append(value)
            uncached_value_indices.append(i)
    if len(uncached_values) > 0:
        new_results = f(uncached_values)
        for idx, result in zip(uncached_value_indices, new_results):
            output_list[idx] = result
    return torch.cat([t.unsqueeze(0) for t in output_list], dim=0)



if __name__ == "__main__":
    main()
