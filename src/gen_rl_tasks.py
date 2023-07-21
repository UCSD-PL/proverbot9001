#!/usr/bin/env python

import argparse
import json
import re
import os

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from tqdm import tqdm

import coq_serapy
from coq_serapy.contexts import (FullContext, ProofContext,
                                 ScrapedTactic, truncate_tactic_context)

from dataloader import scraped_from_file

from search_file import get_all_jobs
from search_worker import get_predictor, ReportJob
from models.tactic_predictor import TacticPredictor

from util import unwrap, eprint


def main():
    parser = argparse.ArgumentParser(
        description="Generate demonstrating-tasks up to a given length "
        "for training an rl agent refining a given predictor")
    add_args_to_parser(parser)
    args = parser.parse_args()

    gen_rl_tasks(args)

def add_args_to_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", "-v", help="verbose output",
                        action="count", default=0, dest='verbosity')
    parser.add_argument("--prelude", default=".", type=Path)
    parser.add_argument("--output", "-o", dest="output_file", type=Path,
                        default="data/rl_jobs.json")
    parser.add_argument('--supervised-weights', type=Path, dest="weightsfile")
    parser.add_argument("--include-proof-relevant", action="store_true")
    parser.add_argument("--blacklist-tactic", action="append", dest="blacklisted_tactics")
    parser.add_argument("--no-resume", action='store_false', dest='resume')
    parser.add_argument("--ignore-lin-hash", action='store_true')
    parser.add_argument("--just-print-jobs", action='store_true', help="Just print the jobs you *would* do, then exit")
    parser.add_argument("--data-partition", choices=["test", "train"], default="train")
    proofsGroup = parser.add_mutually_exclusive_group()
    proofsGroup.add_argument("--proof", default=None)
    proofsGroup.add_argument("--proofs-file", default=None)
    parser.add_argument("-l", "--max-target-length", type=int, default=None)
    parser.add_argument("-p", "--num-predictions", default=16, type=int)
    parser.add_argument("json_project_file", type=Path)

@dataclass
class RLTask:
    src_file: Path
    module_prefix: str
    proof_statement: str
    tactic_prefix: List[str]
    orig_solution: List[str]
    target_length: int
    largest_prediction_idx: int
    def to_job(self) -> ReportJob:
        return ReportJob(".", self.src_file, self.module_prefix, self.proof_statement)
    @classmethod
    def from_job(cls, job: ReportJob) -> 'RLTask':
        return RLTask(job.filename, job.module_prefix, job.lemma_statment, [], [], -1, -1)

def get_job_interactions(args: argparse.Namespace, job: ReportJob) -> List[ScrapedTactic]:
    full_path = args.prelude / job.project_dir / job.filename
    file_interactions = scraped_from_file(str(full_path.with_suffix(".v.scrape")))

    sm_stack = coq_serapy.initial_sm_stack(job.filename)
    in_proof = False
    job_interactions = []
    for interaction in file_interactions:
        if isinstance(interaction, str):
            eprint(f"Processing {interaction}", guard=args.verbosity > 1)
            sm_stack = coq_serapy.update_sm_stack(sm_stack, interaction)
            sm_prefix = coq_serapy.sm_prefix_from_stack(sm_stack)
            if coq_serapy.kill_comments(job.lemma_statement).strip() == \
               coq_serapy.kill_comments(interaction).strip() and \
               sm_prefix == job.module_prefix:
                in_proof = True
        elif in_proof:
            eprint(f"Processing {interaction.tactic}", guard=args.verbosity > 1)
            if re.match(r"[\{\}\+\-\*]+", coq_serapy.kill_comments(interaction.tactic).strip()) :
                continue
            job_interactions.append(ScrapedTactic.from_structeq(interaction))
            if coq_serapy.ending_proof(interaction.tactic):
                return job_interactions
    assert False, "Couldn't find job or proof ending"


def gen_rl_tasks(args: argparse.Namespace) -> None:
    predictor = get_predictor(args, allow_static_predictor=False)

    args.splits_file = args.json_project_file
    all_jobs = get_all_jobs(args, partition=args.data_partition + "_files")

    partial_output = Path(str(args.output_file) + ".partial")
    jobs_done_output = Path(str(args.output_file) + ".donejobs")

    if jobs_done_output.exists() and args.resume:
        with jobs_done_output.open('r') as f:
            jobs_already_done = [ReportJob(*json.loads(line)) for line in f]
    else:
        with jobs_done_output.open('w') as f:
            pass
        with partial_output.open('w') as f:
            pass
        jobs_already_done = []

    for job in tqdm(all_jobs,desc="Creating tasks"):
        if job in jobs_already_done and args.resume:
            continue
        if "Program" in coq_serapy.kill_comments(job.lemma_statement) :
            continue
        if args.verbosity > 0:
            eprint(f"Running job {job}")
        tasks = get_job_tasks(args, predictor, job)
        with partial_output.open('a') as f:
            for task in tasks:
                print(json.dumps(vars(task)), file=f)
        with jobs_done_output.open('a') as f:
            print(json.dumps(job), file=f)

    os.rename(partial_output, args.output_file)
    os.remove(jobs_done_output)

def get_job_tasks(args: argparse.Namespace, predictor: TacticPredictor,
                  job: ReportJob) -> List[RLTask]:
    commands = get_job_interactions(args, job)
    normalized = normalize_proof_interactions(commands, args.verbosity)
    tasks = gen_rl_obl_tasks_job(args, predictor, normalized, job)
    return tasks

@dataclass
class JobObligation:
    tactic_prefix: List[str]
    # For each tactic in the obligation, the tactic and whether it is
    # in the predictions
    tactic_contents: List[Tuple[str, bool]]

# Only works properly for goal-selector-normalized solutions that are
# annotated with whether they are in the predictions.
def obls_from_solution(cmds: List[Tuple[str, Optional[int]]]) -> List[JobObligation]:
    def get_cur_obl_solution(remaining_cmds: List[Tuple[str, Optional[int]]]) \
            -> List[Tuple[str, Optional[int]]]:
        sol = []
        bracket_depth = 0
        for cmd in remaining_cmds:
            if cmd[0] == "}" and bracket_depth == 0:
                return sol
            if cmd[0] == "{":
                bracket_depth += 1
            elif cmd[0] == "}":
                bracket_depth -= 1
            sol.append(cmd)
        return sol
    obligations = [JobObligation([], cmds[:-1])]
    for cmd_idx, (cmd, _) in enumerate(cmds):
        if cmd == "{":
            obligations.append(JobObligation([cmd[0] for cmd in cmds[:cmd_idx+1]],
                                             get_cur_obl_solution(cmds[cmd_idx+1:])))
    return obligations

def normalize_proof_interactions(interactions: List[ScrapedTactic],
                                 verbosity:int = 0) -> List[ScrapedTactic]:
    output_interactions: List[ScrapedTactic] = []
    num_subgoals_stack: List[int] = [1]
    previous_num_subgoals: int = 1
    for interaction in interactions:
        if verbosity > 1:
            coq_serapy.summarizeContext(interaction.context)
            eprint(interaction.tactic)
        num_subgoals = len(interaction.context.fg_goals) + len(interaction.context.bg_goals)
        subgoals_created_by_last_tac = num_subgoals - previous_num_subgoals
        if subgoals_created_by_last_tac > 0:
            num_subgoals_stack.append(subgoals_created_by_last_tac + 1)
            output_interactions.append(
                ScrapedTactic(interaction.relevant_lemmas,
                              interaction.prev_tactics,
                              interaction.context,
                              "{"))
        if subgoals_created_by_last_tac < 0:
            assert subgoals_created_by_last_tac == -1, \
                "Shouldn't be able to close more than one subgoal at a time. " \
                f"Num subgoals before: {previous_num_subgoals}, "\
                f"num subgoals after: {len(interaction.context.all_goals)}"
            num_subgoals_stack[-1] -= 1
            output_interactions.append(
                ScrapedTactic(
                    interaction.relevant_lemmas,
                    interaction.prev_tactics,
                    interaction.context,
                "}"))
            while len(num_subgoals_stack) > 1 and num_subgoals_stack[-1] == 0:
                num_subgoals_stack.pop()
                num_subgoals_stack[-1] -= 1
                if len(num_subgoals_stack) > 1:
                    output_interactions.append(
                        ScrapedTactic(interaction.relevant_lemmas,
                                      interaction.prev_tactics,
                                      interaction.context,
                                      "}"))
                else:
                    assert num_subgoals == 0, interaction.tactic
            if len(num_subgoals_stack) > 1:
                output_interactions.append(
                    ScrapedTactic(
                        interaction.relevant_lemmas,
                        interaction.prev_tactics,
                        ProofContext(interaction.context.bg_goals[:num_subgoals_stack[-1]],
                                     interaction.context.bg_goals[num_subgoals_stack[-1]:],
                                     interaction.context.shelved_goals,
                                     interaction.context.given_up_goals),
                        "{"))
        previous_num_subgoals = num_subgoals

        if interaction.tactic.strip() not in ["{", "}"]:
            if len(interaction.context.fg_goals) == 0:
                output_interactions.append(interaction)
            else:
                output_interactions.append(
                    ScrapedTactic(
                        interaction.relevant_lemmas,
                        interaction.prev_tactics,
                        ProofContext([interaction.context.fg_goals[0]],
                                     interaction.context.fg_goals[1:] +
                                     interaction.context.bg_goals,
                                     interaction.context.shelved_goals,
                                     interaction.context.given_up_goals),
                        interaction.tactic))
    assert len(interactions) > 0
    for _ in range(len(num_subgoals_stack) - 1):
        output_interactions.append(
            ScrapedTactic(interactions[-1].relevant_lemmas,
                          interactions[-1].prev_tactics,
                          interactions[-1].context,
                          "}"))
    return output_interactions

def annotate_cmds_in_pred(args: argparse.Namespace,
                          predictor: TacticPredictor,
                          sol_contexts: List[ScrapedTactic]) \
                          -> List[Tuple[str, Optional[int]]]:

    annotated_sol: List[Tuple[str,Optional[int]]] = []
    for sol_context in tqdm(sol_contexts, desc="Checking predictions",
                            leave=False, disable=len(sol_contexts) < 200):
        sol_cmd = sol_context.tactic
        if sol_cmd in ['{', '}'] :
            annotated_sol.append((sol_cmd, 0))
            continue
        if coq_serapy.ending_proof(sol_cmd) :
            annotated_sol.append((sol_cmd,0))
            break

        predictions = predictor.predictKTactics(
            truncate_tactic_context(FullContext(sol_context.relevant_lemmas,
                                                sol_context.prev_tactics,
                                                unwrap(sol_context.context)
                                                ).as_tcontext(),
                                    30),
            args.num_predictions,
            blacklist=args.blacklisted_tactics)

        nsol_cmd = coq_serapy.kill_comments(sol_cmd).strip()
        if nsol_cmd == "auto.":
            nsol_cmd = "eauto."
        try:
            sol_rank = [p.prediction for p in predictions].index(nsol_cmd)
        except ValueError:
            sol_rank = None

        annotated_sol.append((sol_cmd, sol_rank))
    return annotated_sol



def gen_rl_obl_tasks_job(args: argparse.Namespace, predictor: TacticPredictor,
                         normalized_scrape: List[ScrapedTactic], job: ReportJob) -> List[RLTask]:

    annotated_cmds = annotate_cmds_in_pred(args, predictor, normalized_scrape)
    annotated_obls = obls_from_solution(annotated_cmds)

    tasks = []

    for aobl in annotated_obls:
        largest_prediction_rank = 0
        for cmd_idx, (cmd, prediction_rank) in \
                reversed(list(enumerate(aobl.tactic_contents))):
            if prediction_rank is None:
                break
            largest_prediction_rank = max(largest_prediction_rank, prediction_rank)
            if cmd in ["{", "}"]:
                continue
            task_prefix = aobl.tactic_prefix + [cmd[0] for cmd in aobl.tactic_contents[:cmd_idx]]
            task_solution = [cmd[0] for cmd in aobl.tactic_contents[cmd_idx:]]
            sol_tac_length = len([tac for tac in task_solution if tac not in ["{", "}"]])
            if args.max_target_length and sol_tac_length > args.max_target_length:
                break
            if len([tac for tac in task_solution if tac == "{"]) != \
               len([tac for tac in task_solution if tac == "}"]):
                continue
            tasks.append(RLTask(job.filename, job.module_prefix, job.lemma_statement,
                                task_prefix, task_solution, sol_tac_length,
                                largest_prediction_rank))
    return tasks

if __name__ == "__main__":
    main()
