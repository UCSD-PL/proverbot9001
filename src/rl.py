#!/usr/bin/env python

import argparse
import json
import random
import math
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union, Any

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from tqdm import tqdm, trange

import coq_serapy
from coq_serapy.contexts import (FullContext, truncate_tactic_context,
                                 Obligation, TacticContext, ProofContext)
import coq2vec

from search_file import get_all_jobs
from search_worker import ReportJob, Worker, get_predictor
from search_strategies import completed_proof

from models.tactic_predictor import (TacticPredictor, Prediction)

from util import unwrap, eprint, print_time

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
    parser.add_argument('--supervised-weights', default=None, type=Path, dest="weightsfile")
    parser.add_argument("--coq2vec-weights", type=Path)
    parser.add_argument("-l", "--learning-rate", default=2.5e-4, type=float)
    parser.add_argument("-g", "--gamma", default=0.9, type=float)
    parser.add_argument("-s", "--steps-per-episode", default=16, type=int)
    parser.add_argument("-n", "--num-episodes", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("--batches-per-proof", default=1, type=int)
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

    jobs = get_all_jobs(args)

    reinforce_jobs(args, jobs)

class ReinforcementWorker(Worker):
    v_network: 'VNetwork'
    replay_buffer: List['Transition']
    verbosity: int
    def __init__(self, args: argparse.Namespace,
                 predictor: TacticPredictor,
                 v_network: 'VNetwork',
                 switch_dict: Optional[Dict[str, str]] = None,
                 initial_replay_buffer: List['Transition'] = None) -> None:
        super().__init__(args, 0, predictor, switch_dict)
        self.v_network = v_network
        if not initial_replay_buffer:
            self.replay_buffer = []
        else:
            self.replay_buffer = initial_replay_buffer
        self.verbosity = args.verbose
    def run_job_reinforce(self, job: ReportJob, restart: bool = True) -> None:
        assert self.coq
        self.run_into_job(job, restart, False)
        with print_time("Experiencing proof", guard=self.verbosity >= 1):
            experience_proof(self.args, self.coq, self.predictor, self.v_network,
                             self.replay_buffer)
        train_v_network(self.args, self.v_network, self.replay_buffer)
        # Pop the rest of the proof from the remaining commands
        while not coq_serapy.ending_proof(self.remaining_commands[0]):
            self.remaining_commands.pop(0)
        # Pop the actual Qed/Defined/Save
        ending_cmd = self.remaining_commands.pop(0)
        coq_serapy.admit_proof(self.coq, self.coq.prev_tactics[0], ending_cmd)


def reinforce_jobs(args: argparse.Namespace, jobs: List[ReportJob]) -> None:
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
    predictor = DummyPredictor()
    episodes_already_done = 0
    replay_buffer = None
    if args.output_file.exists() and args.resume != "no":
        if args.resume == "yes":
            print("Resuming from existing weights")
            replay_buffer, episodes_already_done, network_state = \
                torch.load(str(args.output_file))
            v_network = VNetwork(None, args.learning_rate)
            v_network.load_state(network_state)
        else:
            assert args.resume == "ask"
            print(f"Found existing weights at {args.output_file}. Resume?")
            response = input("[Y/n]")
            if response.lower() not in ["no", "n"]:
                print("Resuming from existing weights")
                replay_buffer, episodes_already_done, network_state = \
                    torch.load(str(args.output_file))
                v_network = VNetwork(None, args.learning_rate)
                v_network.load_state(network_state)
            else:
                v_network = VNetwork(args.coq2vec_weights, args.learning_rate)
    else:
        v_network = VNetwork(args.coq2vec_weights, args.learning_rate)

    with ReinforcementWorker(args, predictor, v_network, switch_dict,
                             initial_replay_buffer = replay_buffer) as worker:
        step = 0
        if args.interleave:
            for episode in trange(episodes_already_done, args.num_episodes,
                                  disable=args.verbose >= 1):
                for job in jobs:
                    worker.run_job_reinforce(job)
                    if (step + 1) % args.save_every == 0:
                        save_state(args, worker, episode + 1)
                    step += 1
        else:
            for job in tqdm(jobs, disable=args.verbosity >= 1):
                for episode in range(episodes_already_done, args.num_episodes):
                    worker.run_job_reinforce(job)
                    if step % args.save_every == 0:
                        save_state(args, worker, episode)
                    step += 1
        if args.evaluate:
            evaluate_results(args, worker, jobs)

Transition = Tuple[Obligation, str, List[Obligation]]

class VNetwork:
    obligation_encoder: Optional[coq2vec.CoqContextVectorizer]
    network: nn.Module
    def get_state(self) -> Any:
        return (self.network.state_dict(),
                self.obligation_encoder.term_encoder.get_state())

    def _load_encoder_state(self, encoder_state: Any) -> None:
        term_encoder = coq2vec.CoqTermRNNVectorizer()
        term_encoder.load_state(encoder_state)
        num_hyps = 5
        assert self.obligation_encoder is None, "Can't load weights twice!"
        self.obligation_encoder = coq2vec.CoqContextVectorizer(term_encoder, num_hyps)
        insize = term_encoder.hidden_size * (num_hyps + 1)
        self.network = nn.Sequential(
            nn.Linear(insize, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1),
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)


    def load_state(self, state: Any) -> None:
        network_state, encoder_state = state
        self._load_encoder_state(encoder_state)
        self.network.load_state_dict(network_state)


    def __init__(self, coq2vec_weights: Optional[Path], learning_rate: float) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.obligation_encoder = None
        self.network = None
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

        encoded_obl_size = (self.obligation_encoder.term_encoder.hidden_size *
                            (self.obligation_encoder.max_num_hypotheses + 1))

        encoded = self.obligation_encoder.obligations_to_vectors(
            [coq2vec.Obligation(obl.hypotheses, obl.goal) for obl in obls])\
                                         .view(len(obls), encoded_obl_size)
        scores = self.network(encoded).view(len(obls))
        return scores

    def train(self, inputs: List[Obligation],
              target_outputs: List[float],
              verbosity: int = 0) -> None:
        assert self.obligation_encoder, \
            "No loaded encoder! Pass encoder weights to __init__ or call set_state()"
        with print_time("Training", guard=verbosity >= 1):
            actual = self(inputs)
            target = torch.FloatTensor(target_outputs)
            loss = F.mse_loss(actual, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        eprint(f"Actual: {actual}; Target: {target}", guard=verbosity >= 2)
        eprint(f"Loss: {loss}", guard=verbosity >= 1)
def experience_proof(args: argparse.Namespace,
                     coq: coq_serapy.CoqAgent,
                     predictor: TacticPredictor,
                     v_network: VNetwork,
                     replay_buffer: List[Transition]) -> None:
    path: List[ProofContext] = [coq.proof_context]
    for _step in range(args.steps_per_episode):
        before_obl = unwrap(coq.proof_context).fg_goals[0]
        actions = predictor.predictKTactics(
            truncate_tactic_context(FullContext(coq.local_lemmas,
                                                coq.prev_tactics,
                                                unwrap(coq.proof_context)).as_tcontext(),
                                    30),
            5,
            blacklist=args.blacklisted_tactics)
        eprint(f"Trying predictions {[action.prediction for action in actions]}",
               guard=args.verbose >= 3)
        action_scores = [evaluate_action(args, coq, v_network, path, action.prediction)
                         for action in actions]
        best_action = max(zip(actions, action_scores), key=lambda p: p[1])[0]

        eprint(f"Taking action {best_action}",
               guard=args.verbose >= 2)

        resulting_obls = execute_action(coq, best_action.prediction)
        eprint(f"New context is {coq.proof_context}",
               guard=args.verbose >= 3)
        replay_buffer.append((before_obl, best_action.prediction, resulting_obls))
        path.append(coq.proof_context)
        if completed_proof(coq):
            break

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

    product = math.prod(v_network(goal) for goal in
                        unwrap(context_after).fg_goals)
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
                    replay_buffer: List[Transition]):
    for _batch_idx in range(args.batches_per_proof):
        if len(replay_buffer) >= args.batch_size:
            samples = random.sample(replay_buffer, args.batch_size)
        elif args.allow_partial_batches:
            samples = replay_buffer
        else:
            break
        with print_time("Computing outputs", guard=args.verbose >= 1):
            resulting_obls_lens = [len(obls) for state, action, obls in samples]
            all_obl_scores = v_network([obl for state, action, obls in samples for obl in obls])
            outputs = []
            cur_row = 0
            for num_obls in resulting_obls_lens:
                outputs.append(args.gamma * math.prod(all_obl_scores[cur_row:cur_row+num_obls]))
                cur_row += num_obls
        v_network.train([start_obl for start_obl, action, resulting_obls in samples],
                        outputs, verbosity=args.verbose)

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
               episode: int) -> None:
    with args.output_file.open('wb') as f:
        torch.save((worker.replay_buffer, episode,
                    worker.v_network.get_state()), f)

def evaluate_results(args: argparse.Namespace,
                     worker: ReinforcementWorker,
                     jobs: List[ReportJob]) -> None:
    proofs_completed = 0
    for job in jobs:
        worker.run_into_job(job, True, False)
        path: List[ProofContext] = [worker.coq.proof_context]
        for _step in range(args.steps_per_episode):
            actions = worker.predictor.predictKTactics(
                truncate_tactic_context(FullContext(
                    worker.coq.local_lemmas,
                    worker.coq.prev_tactics,
                    unwrap(worker.coq.proof_context)).as_tcontext(),
                                        30),
                5,
                blacklist=args.blacklisted_tactics)
            if args.verbose >= 1:
                coq_serapy.summarizeContext(worker.coq.proof_context)
            eprint(f"Trying predictions {[action.prediction for action in actions]}",
                   guard=args.verbose >= 2)
            action_scores = [evaluate_action(args, worker.coq, worker.v_network,
                                             path, action.prediction)
                             for action in actions]
            best_action, best_score = max(zip(actions, action_scores), key=lambda p: p[1])
            eprint(f"Taking action {best_action} with estimated value {best_score}",
                   guard=args.verbose >= 1)
            execute_action(worker.coq, best_action.prediction)
            path.append(worker.coq.proof_context)
            if completed_proof(worker.coq):
                proofs_completed += 1
                break
    print(f"{proofs_completed} out of {len(jobs)} "
          f"theorems/lemmas successfully proven "
          f"({stringified_percent(proofs_completed, len(jobs))}%)")

def stringified_percent(total : float, outof : float) -> str:
    if outof == 0:
        return "NaN"
    return f"{(total * 100 / outof):10.2f}"

if __name__ == "__main__":
    main()
