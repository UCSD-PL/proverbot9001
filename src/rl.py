#!/usr/bin/env python

import argparse
import json
import random
import math
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

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
    parser.add_argument("--output", "-o", dest="output_dir",
                        help="output data folder name",
                        default="search-report",
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
    def __init__(self, args: argparse.Namespace,
                 predictor: TacticPredictor,
                 v_network: 'VNetwork',
                 switch_dict: Optional[Dict[str, str]] = None) -> None:
        super().__init__(args, 0, predictor, switch_dict)
        self.v_network = v_network
        self.replay_buffer = []
    def run_job_reinforce(self, job: ReportJob, restart: bool = True) -> None:
        assert self.coq
        self.run_into_job(job, restart, False)
        with print_time("Experiencing proof"):
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
    v_network = VNetwork(args.coq2vec_weights, args.learning_rate)
    with ReinforcementWorker(args, predictor, v_network, switch_dict) as worker:
        if args.interleave:
            for _episode in range(args.num_episodes):
                for job in jobs:
                    worker.run_job_reinforce(job)
        else:
            for job in jobs:
                for _episode in range(args.num_episodes):
                    worker.run_job_reinforce(job)

Transition = Tuple[Obligation, str, List[Obligation]]

class VNetwork:
    obligation_encoder: coq2vec.CoqContextVectorizer
    network: nn.Module
    def __init__(self, coq2vec_weights: Path, learning_rate: float) -> None:
        term_encoder = coq2vec.CoqTermRNNVectorizer()
        term_encoder.load_weights(coq2vec_weights)
        num_hyps = 5
        self.obligation_encoder = coq2vec.CoqContextVectorizer(term_encoder, num_hyps)
        insize = term_encoder.hidden_size * (num_hyps + 1)
        self.network = nn.Sequential(
            nn.Linear(term_encoder.hidden_size * (num_hyps + 1), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1),
        )

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def __call__(self, obls: Union[Obligation, List[Obligation]]) -> torch.FloatTensor:
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
        with print_time("Training"):
            actual = self(inputs)
            target = torch.FloatTensor(target_outputs)
            loss = F.mse_loss(actual, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if verbosity >= 1:
            eprint(f"Actual: {actual}; Target: {target}")
            eprint(f"Loss: {loss}")

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
    score = torch.tensor(args.gamma) * product

    return args.gamma * score

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
        with print_time("Computing outputs"):
            outputs = [args.gamma * math.prod(v_network(obls))
                       for state, action, obls in samples]
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

if __name__ == "__main__":
    main()
