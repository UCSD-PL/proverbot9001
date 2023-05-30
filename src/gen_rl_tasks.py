#!/usr/bin/env python

import argparse
import json
import re

from pathlib import Path
from dataclasses import dataclass
from typing import List

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

    with args.output_file.open('w'):
        pass

    with TaskWorker(args, switch_dict) as worker:
        for job in tqdm(all_jobs, desc="Processing jobs"):
            worker.run_into_job(job, False, args.careful)
            tasks = gen_rl_tasks_obligation_job(args, predictor, worker, job)
            with args.output_file.open('a') as f:
                for task in tasks:
                    print(json.dumps(vars(task)), file=f)

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



def get_curr_obligation_job_solution(worker:Worker) -> List[List[str]] :
    all_job_solutions = []
    job_solution = []
    remaining_commands = list(worker.remaining_commands)
    command_index = 0
    while not coq_serapy.ending_proof( remaining_commands[command_index]):
        cmd = remaining_commands[command_index].strip()
        command_index += 1
        if re.match(r"\}", coq_serapy.kill_comments(cmd).strip()) :
            all_job_solutions.append(list(job_solution))
            job_solution.append(cmd)
            continue
        if re.match(r"[\+\-\*]+", coq_serapy.kill_comments(cmd).strip()):
            eprint("For the command set", remaining_commands)
            raise ValueError("Use Linearized version of the file. Found non Linearized command : " + cmd )
        if cmd.strip() == "Proof.":
            continue
        job_solution.append(cmd)

    assert coq_serapy.ending_proof( remaining_commands[command_index])
    if len(all_job_solutions) == 0 or len(job_solution) > len(all_job_solutions[-1]) : #Add the last closing Obligation, if an Obligation has not been immidiately closed
        all_job_solutions.append(list(job_solution))
    while not coq_serapy.ending_proof(remaining_commands[0]) :
        remaining_commands.pop(0)

    return all_job_solutions


def sol_cmds_in_predictions(args: argparse.Namespace,
                            worker: Worker, predictor: TacticPredictor,
                            sol_commands: List[str]) -> List[bool]:
    sol_command_in_predictions: List[bool] = []
    for sol_cmd in sol_commands:
        predictions = predictor.predictKTactics(
            truncate_tactic_context(FullContext(worker.coq.local_lemmas,
                                                worker.coq.prev_tactics,
                                                unwrap(worker.coq.proof_context)
                                                ).as_tcontext(),
                                    30),
            args.num_predictions,
            blacklist=args.blacklisted_tactics)
        in_predictions = (coq_serapy.kill_comments(sol_cmd).strip()
                          in [p.prediction for p in predictions])
        sol_command_in_predictions.append(in_predictions)
        worker.coq.run_stmt(sol_cmd)
    worker.skip_proof(args.careful)
    return sol_command_in_predictions



def gen_rl_tasks_job(args: argparse.Namespace, predictor: TacticPredictor,
                     worker: Worker, job: ReportJob) -> List[RLTask]:
    _, filename, module_prefix, lemma_statement = job

    job_existing_solution = get_cur_job_solution(worker)

    sol_command_in_prediction: List[bool] = \
        sol_cmds_in_predictions(args, worker, predictor, job_existing_solution)
    tasks: List[RLTask] = []

    cur_task_length = 1
    while cur_task_length <= args.max_target_length \
        and cur_task_length < len(job_existing_solution) \
        and sol_command_in_prediction[-cur_task_length]:
        tasks.append(RLTask(filename, module_prefix, lemma_statement,
                            job_existing_solution[:-cur_task_length],
                            job_existing_solution[-cur_task_length:],
                            cur_task_length))
        cur_task_length += 1
    return tasks


def remove_brackets(sol) :
    bracketless_solution = []
    for command in sol :
        if re.match(r"[\{\}]+", coq_serapy.kill_comments(command).strip()) :
            continue
        else :
            bracketless_solution.append(command)
    return bracketless_solution


def gen_rl_tasks_obligation_job(args: argparse.Namespace, predictor: TacticPredictor,
                     worker: Worker, job: ReportJob) -> List[RLTask]:
    _, filename, module_prefix, lemma_statement, tactic_prefix = job

    job_existing_solutions = get_curr_obligation_job_solution(worker)
    bracketless_solutions = [remove_brackets(sol) for sol in job_existing_solutions]
    sol_command_in_predictions: List[bool] = \
        sol_cmds_in_predictions(args, worker, predictor, bracketless_solutions[-1])
    tasks: List[RLTask] = []

    for job_existing_bracketless_solution, job_existing_solution in \
            zip(bracketless_solutions, job_existing_solutions):
        curr_sol_command_in_prediction = sol_command_in_predictions[:len(job_existing_bracketless_solution)]

        cur_task_length = 1
        closed_brace_count = 1
        cur_checked_length = 0


        while cur_task_length <= args.max_target_length \
            and cur_checked_length < len(job_existing_solution) \
            and curr_sol_command_in_prediction[-cur_task_length]  \
            and closed_brace_count > 0:

            cur_checked_length += 1
            if re.match(r"[\{*]+", coq_serapy.kill_comments(job_existing_solution[-cur_checked_length]).strip()) :
                closed_brace_count -= 1
                continue
            if re.match(r"[\}*]+", coq_serapy.kill_comments(job_existing_solution[-cur_checked_length]).strip()) :
                closed_brace_count += 1
                continue


            cur_task_length += 1
            if closed_brace_count > 1  :
                continue

            tasks.append(RLTask(filename, module_prefix, lemma_statement,
                                job_existing_bracketless_solution[:-cur_task_length],
                                job_existing_bracketless_solution[-cur_task_length:],
                                cur_task_length))

    return tasks


if __name__ == "__main__":
    main()
