#!/usr/bin/env python

import argparse
import json
import re
import os

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

from tqdm import tqdm

import coq_serapy
from coq_serapy.contexts import (FullContext, truncate_tactic_context)

from search_file import get_all_jobs
from search_worker import get_predictor, Worker, ReportJob
from models.tactic_predictor import TacticPredictor

from util import unwrap, eprint, print_time
from linearize_semicolons import get_linearized

def main():
    parser = argparse.ArgumentParser(
        description="Generate demonstrating-tasks up to a given length "
        "for training an rl agent refining a given predictor")
    parser.add_argument("--verbose", "-v", help="verbose output",
                        action="count", default=0)
    parser.add_argument("--prelude", default=".", type=Path)
    parser.add_argument("--output", "-o", dest="output_file", type=Path,
                        default="data/rl_jobs.json")
    parser.add_argument('--supervised-weights', type=Path, dest="weightsfile")
    parser.add_argument("--no-set-switch", dest="set_switch", action='store_false')
    parser.add_argument("--include-proof-relevant", action="store_true")
    parser.add_argument("--blacklist-tactic", action="append", dest="blacklisted_tactics")
    parser.add_argument("--use-linearized", action="store_true")
    parser.add_argument("--backend", choices=['serapi', 'lsp', 'auto'], default='auto')
    parser.add_argument("--careful", action='store_true')
    parser.add_argument("--no-resume", action='store_false', dest='resume')
    proofsGroup = parser.add_mutually_exclusive_group()
    proofsGroup.add_argument("--proof", default=None)
    proofsGroup.add_argument("--proofs-file", default=None)
    parser.add_argument("-l", "--max-target-length", type=int, default=3)
    parser.add_argument("-p", "--num-predictions", default=16, type=int)
    parser.add_argument("json_project_file", type=Path)
    args = parser.parse_args()

    gen_rl_tasks(args)

@dataclass
class RLTask:
    src_file: Path
    module_prefix: str
    proof_statement: str
    tactic_prefix: List[str]
    orig_solution: List[str]
    target_length: int

class TaskWorker(Worker):
    def enter_file(self, filename: str) -> None:
        assert self.coq
        self.cur_file = filename
        module_name = coq_serapy.get_module_from_filename(filename)
        self.coq.run_stmt(f"Module {module_name}.")
        if self.args.use_linearized:
            self.args.linearizer_timeout = 60 ** 2
            self.args.progress = True
            self.args.hardfail = False
            self.remaining_commands = get_linearized(self.args, ["sertop"], 0,
                                                     str(Path(self.cur_project) / filename))
        else:
            self.remaining_commands = coq_serapy.load_commands_preserve(
                self.args, 1, self.args.prelude / self.cur_project / filename)

def gen_rl_tasks(args: argparse.Namespace) -> None:
    with args.json_project_file.open('r') as f:
        project_dicts = json.loads(f.read())
    if any("switch" in item for item in project_dicts):
        switch_dict = {item["project_name"]: item["switch"]
                       for item in project_dicts}
    else:
        switch_dict = None

    predictor = get_predictor(args, allow_static_predictor=False)

    args.splits_file = args.json_project_file
    all_jobs = get_all_jobs(args) #, partition="train_files")

    partial_output = Path(str(args.output_file) + ".partial")
    jobs_done_output = Path(str(args.output_file) + ".donejobs")

    if jobs_done_output.exists():
        if args.resume:
            with jobs_done_output.open('r') as f:
                jobs_already_done = [ReportJob(*json.loads(line)) for line in f]
        else:
            with jobs_done_output.open('w') as f:
                pass
            with partial_output.open('w') as f:
                pass
    else:
        jobs_already_done = []

    with TaskWorker(args, switch_dict) as worker:
        for job in tqdm(all_jobs, desc="Processing jobs"):
            if job in jobs_already_done:
                continue
            worker.run_into_job(job, False, args.careful)
            tasks = gen_rl_obl_tasks_job(args, predictor, worker, job)
            with partial_output.open('a') as f:
                for task in tasks:
                    print(json.dumps(vars(task)), file=f)
            with jobs_done_output.open('a') as f:
                print(json.dumps(job), file=f)
    os.rename(partial_output, args.output_file)
    os.remove(jobs_done_output)

def get_cur_job_solution(worker: Worker) -> List[str]:
    job_solution = []
    remaining_commands = list(worker.remaining_commands)
    while not coq_serapy.ending_proof(remaining_commands[0]):
        cmd = remaining_commands.pop(0)
        if re.match(r"[\{\}\+\-\*]+", coq_serapy.kill_comments(cmd).strip()):
            continue
        if cmd.strip() == "Proof.":
            continue
        job_solution.append(cmd)
    return job_solution

@dataclass
class JobObligation:
    tactic_prefix: List[str]
    # For each tactic in the obligation, the tactic and whether it is
    # in the predictions
    tactic_contents: List[Tuple[str, bool]]

# Only works properly for goal-selector-normalized solutions that are
# annotated with whether they are in the predictions.
def obls_from_solution(cmds: List[Tuple[str, bool]]) -> List[JobObligation]:
    def get_cur_obl_solution(remaining_cmds: List[Tuple[str, bool]]) -> List[str]:
        sol = []
        bracket_depth = 0
        for cmd in remaining_cmds:
            if coq_serapy.ending_proof(cmd[0]) or cmd[0] == "}" and bracket_depth == 0:
                return sol
            if cmd[0] == "{":
                bracket_depth += 1
            elif cmd[0] == "}":
                bracket_depth -= 1
            sol.append(cmd)
        return sol
    obligations = [JobObligation([], get_cur_obl_solution(cmds))]
    for cmd_idx, (cmd, _) in enumerate(cmds):
        if cmd == "{":
            obligations.append(JobObligation([cmd[0] for cmd in cmds[:cmd_idx+1]],
                                             get_cur_obl_solution(cmds[cmd_idx+1:])))
    return obligations

def normalize_and_check_predictions(args: argparse.Namespace,
                                    coq: coq_serapy.CoqAgent,
                                    predictor: TacticPredictor,
                                    input_cmds: List[str]) -> List[Tuple[str, bool]]:
    normalized_checked_cmds = []
    for cmd in input_cmds:
        # Ignore original goal selectors, we'll add them ourselves.
        if re.match(r"[\{\}]|([\+\-\*]+)", coq_serapy.kill_comments(cmd).strip()):
            continue

        if coq_serapy.ending_proof(cmd):
            normalized_checked_cmds.append((cmd, True))
            break
        # Next call the predictor, to determine if the command is in
        # the predictions.
        predictions = predictor.predictKTactics(
            truncate_tactic_context(FullContext(coq.local_lemmas,
                                                coq.prev_tactics,
                                                unwrap(coq.proof_context)
                                                ).as_tcontext(),
                                    30),
            args.num_predictions,
            blacklist=args.blacklisted_tactics)
        in_predictions = (coq_serapy.kill_comments(cmd).strip()
                          in [p.prediction for p in predictions])
        # Special-case the `auto` tactic, since the proverbot model
        # preprocesses it into "eauto." If the solution tactic is
        # "auto", but we predict "eauto", that would work too so count
        # it as in-domain.
        if coq_serapy.kill_comments(cmd).strip() == "auto.":
            in_predictions |= "eauto." in [p.prediction for p in predictions]

        # All non-goal-selectors get added to the normalized solution.
        normalized_checked_cmds.append((cmd, in_predictions))

        coq.run_stmt(cmd)

        # All goal manipulation commands will be given a True for
        # in_predictions, since the prediction running engine is able
        # to run them automatically.

        # If we've completed all our subgoals, close goals until we
        # hit the Qed or some goals come into the foreground.
        just_closed = False
        while coq.count_fg_goals() == 0 and \
              (len(coq.proof_context.all_goals) > 0 or \
               coq.tactic_history.curDepth() > 0):
            just_closed = True
            coq.run_stmt("}")
            normalized_checked_cmds.append(("}", True))
        # If we just created multiple obligations, or if we just
        # closed an obligation, and we're starting the next one, add
        # the open bracket.
        if coq.count_fg_goals() > 1 or (just_closed and coq.count_fg_goals() > 0):
            coq.run_stmt("{")
            normalized_checked_cmds.append(("{", True))
    # Clean up our state by cancelling to right after the theorem
    # statement.
    while len(coq.prev_tactics) > 1:
        coq.cancel_last()

    return normalized_checked_cmds


def gen_rl_tasks_job(args: argparse.Namespace, predictor: TacticPredictor,
                     worker: Worker, job: ReportJob) -> List[RLTask]:

    job_existing_solution = get_cur_job_solution(worker)
    norm_sol, in_preds = zip(*normalize_and_check_predictions(
        args, worker.coq, predictor, job_existing_solution))

    tasks: List[RLTask] = []

    for cur_task_length in range(1, len(norm_sol)):
        if not in_preds[-cur_task_length]:
            break
        task_prefix = job_existing_solution[:-cur_task_length]
        task_solution = job_existing_solution[-cur_task_length:]
        if len([tac for tac in task_solution if tac not in ["{", "}", "Unshelve."]]) > \
           args.max_target_length:
            break
        if len([tac for tac in task_solution if tac == "{"]) != \
           len([tac for tac in task_solution if tac == "}"]):
            continue
        tasks.append(RLTask(job.filename, job.module_prefix,
                            job.lemma_statement,
                            task_prefix, task_solution,
                            cur_task_length))
    return tasks


def gen_rl_obl_tasks_job(args: argparse.Namespace, predictor: TacticPredictor,
                         worker: Worker, job: ReportJob) -> List[RLTask]:
    annotated_cmds = normalize_and_check_predictions(args, worker.coq, predictor,
                                                     get_cur_job_solution(worker))
    annotated_obls = obls_from_solution(annotated_cmds)

    tasks = []

    for aobl in annotated_obls:
        for cmd_idx, (cmd, in_pred) in reversed(list(enumerate(aobl.tactic_contents))):
            if not in_pred:
                break
            if cmd in ["{", "}"]:
                continue
            task_prefix = aobl.tactic_prefix + [cmd[0] for cmd in aobl.tactic_contents[:cmd_idx]]
            task_solution = [cmd[0] for cmd in aobl.tactic_contents[cmd_idx:]]
            sol_tac_length = len([tac for tac in task_solution if tac not in ["{", "}"]])
            if sol_tac_length > args.max_target_length:
                break
            if len([tac for tac in task_solution if tac == "{"]) != \
               len([tac for tac in task_solution if tac == "}"]):
                continue
            tasks.append(RLTask(job.filename, job.module_prefix, job.lemma_statement,
                                task_prefix, task_solution, sol_tac_length))
    worker.skip_proof(args.careful)
    return tasks

if __name__ == "__main__":
    main()
