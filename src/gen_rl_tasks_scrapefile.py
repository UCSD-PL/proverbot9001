#!/usr/bin/env python

import argparse
import json
import re
import os

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

from tqdm import tqdm

import coq_serapy
from coq_serapy.contexts import (FullContext, ProofContext,
                                 ScrapedTactic, truncate_tactic_context)

from dataloader import scraped_from_file

from search_file import get_all_jobs
from search_worker import get_predictor, Worker, ReportJob
from models.tactic_predictor import TacticPredictor

from util import unwrap, eprint, print_time
from linearize_semicolons import get_linearized

import data


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
    parser.add_argument("--no-resume", action='store_false', dest='resume')
    parser.add_argument("--ignore-lin-hash", action='store_true')
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

def get_job_interactions(args: argparse.Namespace, job: ReportJob) -> List[ScrapedTactic]:
    full_path = args.prelude / job.project_dir / job.filename
    file_interactions = scraped_from_file(str(full_path.with_suffix(".v.scrape")))

    sm_stack = coq_serapy.initial_sm_stack(job.filename)
    in_proof = False
    job_interactions = []
    for interaction in file_interactions:
        if isinstance(interaction, str):
            sm_stack = coq_serapy.update_sm_stack(sm_stack, interaction)
            if coq_serapy.kill_comments(job.lemma_statement).strip() == \
               coq_serapy.kill_comments(interaction).strip() and \
               coq_serapy.sm_prefix_from_stack(sm_stack) == job.module_prefix:
                in_proof = True
        elif in_proof:
            job_interactions.append(ScrapedTactic.from_structeq(interaction))
            if coq_serapy.ending_proof(interaction.tactic):
                return job_interactions
    assert False, "Couldn't find job or proof ending"

class ScrapeTaskWorker(Worker):
    def enter_file(self, filename:str) -> None :
        filename += ".scrape"
        if filename[0] =="." :
            filename = "CompCert" + filename[1:]
        interactions = data.read_all_text_data(Path(filename))
        self.all_tactics = []
        self.all_contexts = []
        for interaction in interactions :
            if isinstance(interaction, str) :
                self.all_tactics.append(interaction)
                self.all_contexts.append("")
                continue
            scraped = interaction
            self.all_tactics.append(scraped.tactic)
            proof_context = ProofContext(fg_goals = scraped.context.fg_goals, \
                                         bg_goals = scraped.context.bg_goals, \
                                            shelved_goals = scraped.context.shelved_goals, \
                                                given_up_goals = scraped.context.given_up_goals)
            self.all_contexts.append(FullContext(relevant_lemmas = scraped.relevant_lemmas, \
                                                 prev_tactics = scraped.prev_tactics, \
                                                    obligations = proof_context))
        
        self.remaining_commands = list(self.all_tactics)
        self.remaining_contexts = list(self.all_contexts)

    def run_into_job(self, job: ReportJob) -> None :
        job_project, job_file, job_module, job_lemma = job
        # If we need to change projects, we'll have to reset the coq instance
        # to load new includes, and set the opam switch
        if job_project != self.cur_project:
            if self.cur_project is not None:
                self.reset_file_state()
            self.cur_project = job_project
            if self.args.set_switch:
                self.set_switch_from_proj()
            self.enter_file(job_file)

        # If the job is in a different file load the jobs file from scratch.
        if job_file != self.cur_file:
            self.reset_file_state()
            self.enter_file(job_file)

        # Where's the lemma being collected from? Found a place in scrape file where the full scrape lemma is not exactly matching the full job lemma
        # Resorting to just finding lemma names        
        job_lemma_name = coq_serapy.kill_comments(job_lemma).split(':')[0].strip()
        for ind in range(len(self.all_tactics)) :
            cmd = coq_serapy.kill_comments(self.all_tactics[ind])
            if job_lemma_name in cmd :
                if "Program" in job_lemma_name :
                    self.remaining_commands = list(self.all_tactics[ind + 2 : ])
                    self.remaining_contexts = list(self.all_contexts[ind + 2 : ])
                else :
                    self.remaining_commands = list(self.all_tactics[ind + 1 : ])
                    self.remaining_contexts = list(self.all_contexts[ind + 1 : ])
                break
        else :
            print(coq_serapy.kill_comments(job_lemma))
            raise ValueError("Lemma " + job_lemma_name + " not found in file " + job_file)
        

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

    if jobs_done_output.exists() and args.resume:
        with jobs_done_output.open('r') as f:
            jobs_already_done = [ReportJob(*json.loads(line)) for line in f]
    else:
        with jobs_done_output.open('w') as f:
            pass
        with partial_output.open('w') as f:
            pass
        jobs_already_done = []

    with ScrapeTaskWorker(args, switch_dict) as worker:
        for job in tqdm(all_jobs, desc="Processing jobs"):
            if job in jobs_already_done:
                continue
            worker.run_into_job(job)
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
    job_contexts = []
    remaining_commands = list(worker.remaining_commands)
    remaining_contexts = list(worker.remaining_contexts)
    while not coq_serapy.ending_proof(remaining_commands[0]):
        cmd = remaining_commands.pop(0)
        context = remaining_contexts.pop(0)
        if re.match(r"[\{\}\+\-\*]+", coq_serapy.kill_comments(cmd).strip()):
            continue 
        if cmd.strip() == "Proof.":
            continue
        job_solution.append(cmd)
        job_contexts.append(context)
    return job_solution, job_contexts

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
            if cmd[0] == "}" and bracket_depth == 0:
                return sol
            if cmd[0] == "{":
                bracket_depth += 1
            elif cmd[0] == "}":
                bracket_depth -= 1
            sol.append(cmd)
        return sol
    obligations = [JobObligation([], cmds)]
    for cmd_idx, (cmd, _) in enumerate(cmds):
        if cmd == "{":
            obligations.append(JobObligation([cmd[0] for cmd in cmds[:cmd_idx+1]],
                                             get_cur_obl_solution(cmds[cmd_idx+1:])))
    return obligations

def normalize_proof_interactions(interactions: List[ScrapedTactic]) -> List[ScrapedTactic]:
    output_interactions: List[ScrapedTactic] = []
    num_subgoals_stack: List[int] = [1]
    previous_num_subgoals: int = 1
    for interaction in interactions:
        subgoals_created_by_last_tac = \
            len(interaction.context.all_goals) - previous_num_subgoals
        if subgoals_created_by_last_tac > 0:
            num_subgoals_stack.append(subgoals_created_by_last_tac + 1)
            output_interactions.append(
                ScrapedTactic(interaction.relevant_lemmas,
                              interaction.prev_tactics,
                              interaction.context,
                              "{"))
        if subgoals_created_by_last_tac < 0:
            assert subgoals_created_by_last_tac == -1, \
                "Shouldn't be able to close more than one subgoal at a time."
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
                    assert interaction.tactic.strip() in ["Qed.", "}"], interaction.tactic
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
        previous_num_subgoals = len(interaction.context.all_goals)
        if interaction.tactic.strip() not in ["{", "}"]:
            output_interactions.append(interaction)
    assert len(interactions) > 0
    for _ in range(len(num_subgoals_stack) - 1):
        output_interactions.append(
            ScrapedTactic(interactions[-1].relevant_lemmas,
                          interactions[-1].prev_tactics,
                          interactions[-1].context,
                          "}"))
    return output_interactions

def normalize_and_check_predictions(args: argparse.Namespace,
                                    predictor: TacticPredictor,
                                    input_cmds: List[str],
                                    input_contexts : List[FullContext]) -> List[Tuple[str, bool]]:
    
    #Question, Does Scrape file always produce normalized commands?
    normalized_checked_cmds = []
    
    assert len(input_cmds) == len(input_contexts), str(len(input_cmds)) + " , " + str(len(input_contexts))
    for index in range(len(input_cmds)):
        cmd = input_cmds[index]
        fullcontext = input_contexts[index]
        proofcontext = fullcontext.obligations
        # Ignore original goal selectors, we'll add them ourselves.
        if re.match(r"[\{\}]|([\+\-\*]+)", coq_serapy.kill_comments(cmd).strip()):
            continue

        if coq_serapy.ending_proof(cmd):
            normalized_checked_cmds.append((cmd, True))
            break
        # Next call the predictor, to determine if the command is in
        # the predictions.
        predictions = predictor.predictKTactics(
            truncate_tactic_context(FullContext(fullcontext.relevant_lemmas,
                                                fullcontext.prev_tactics,
                                                unwrap(fullcontext.obligations)
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

        # All goal manipulation commands will be given a True for
        # in_predictions, since the prediction running engine is able
        # to run them automatically.

        # If we've completed all our subgoals, close goals until we
        # hit the Qed or some goals come into the foreground.
        just_closed = False
        while len(proofcontext.fg_goals) == 0 and \
              (len(proofcontext.all_goals) > 0 or len(fullcontext.prev_tactics) > 0 ) : #Is this the right check for tactichistory.depth > 0?
            just_closed = True
            normalized_checked_cmds.append(("}", True))
        # If we just created multiple obligations, or if we just
        # closed an obligation, and we're starting the next one, add
        # the open bracket.
        if  len(proofcontext.fg_goals)  > 1 or (just_closed and  len(proofcontext.fg_goals) > 0):
            normalized_checked_cmds.append(("{", True))

    return normalized_checked_cmds


def gen_rl_obl_tasks_job(args: argparse.Namespace, predictor: TacticPredictor,
                         worker: Worker, job: ReportJob) -> List[RLTask]:
    annotated_cmds = normalize_and_check_predictions(args, predictor,
                                                     *get_cur_job_solution(worker))
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
    # worker.skip_proof(args.careful)
    return tasks

if __name__ == "__main__":
    main()
