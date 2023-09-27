import argparse
from collections import defaultdict
import json
import random
import uuid
import torch
import os
import subprocess
import time
import util
import shutil
import signal
import sys

from pathlib import Path
from glob import glob
from typing import List, Tuple, Optional, Dict, Set
from tqdm import tqdm


from gen_rl_tasks import RLTask
from search_file import get_all_jobs
from search_worker import project_dicts_from_args

from util import (nostderr, eprint,
                  print_time, unwrap, safe_abbrev)

def main():
    eprint("Starting main")
    parser = argparse.ArgumentParser(
        description="Evaluation worker - The rl agent"
        "to complete proofs using Proverbot9001.")
    parser.add_argument("--prelude", default=".", type=Path)
    parser.add_argument("--rl-weights", "-r",
                        help="output data folder name",
                        default="data/rl_weights.dat",
                        type=Path)
    parser.add_argument("--verbose", "-v", help="verbose output",
                        action="count", default=0)
    parser.add_argument("--progress", "-P", help="show progress of files",
                        action='store_true')
    parser.add_argument("--print-timings", "-t", action='store_true')
    parser.add_argument("--no-set-switch", dest="set_switch", action='store_false')
    parser.add_argument("--include-proof-relevant", action="store_true")
    parser.add_argument("--backend", choices=['serapi', 'lsp', 'auto'], default='auto')
    parser.add_argument('filenames', help="proof file name (*.v)",
                        nargs='+', type=Path)
    proofsGroup = parser.add_mutually_exclusive_group()
    proofsGroup.add_argument("--proof", default=None)
    proofsGroup.add_argument("--proofs-file", default=None)
    parser.add_argument("--tasks-file", default=None)
    parser.add_argument('--supervised-weights', type=Path, dest="weightsfile")
    parser.add_argument("--coq2vec-weights", type=Path)
    parser.add_argument("--max-sertop-workers", default=16, type=int)
    parser.add_argument("-p", "--num-predictions", default=5, type=int)
    parser.add_argument("--blacklist-tactic", action="append",
                        dest="blacklisted_tactics")
    parser.add_argument("-s", "--steps-per-episode", default=16, type=int)
    parser.add_argument("--resume", choices=["no", "yes", "ask"], default="ask")
    parser.add_argument("--eval-resume", action="store_true")
    evalGroup = parser.add_mutually_exclusive_group(required=True)
    evalGroup.add_argument("--evaluate", action="store_true")
    evalGroup.add_argument("--evaluate-baseline", action="store_true")
    evalGroup.add_argument("--evaluate-random-baseline", action="store_true")
    parser.add_argument("--state-dir", default="drl_eval_state", type=Path)
    parser.add_argument("--workers-output-dir", default=Path("output"), type=Path)
    parser.add_argument("--num-eval-workers", default=1, type=int)
    parser.add_argument("--worker-alive-time", type=str, default="00:40:00")
    parser.add_argument("--partition", default="cpu")
    parser.add_argument("--mem", default="4G")
    args = parser.parse_args()
    
    if args.filenames[0].suffix == ".json":
        args.splits_file = args.filenames[0]
        args.filenames = None
    else:
        args.splits_file = None
    
    evaluate(args)



def evaluate(args: argparse.Namespace, unique_id: uuid.UUID = uuid.uuid4()) -> None:

    def signal_handler(sig, frame) -> None:
        print('\nProcess Interrupted, Cancelling all Workers')
        result = subprocess.run(f'scancel --name Rl_eval_{unique_id}', shell=True, capture_output=True, text = True)
        print(result.stdout)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    all_tasks = get_all_tasks(args)
    num_tasks_done = setup_eval_jobstate(args)
    num_workers_actually_needed = min(len(all_tasks) - num_tasks_done,
                                      args.num_eval_workers)
    print(f"Deploying {num_workers_actually_needed} workers")
    

    if num_workers_actually_needed > 0:
        if args.tasks_file :
            task_file_arg = f"--tasks-file {str(args.tasks_file)}"
        else :
            task_file_arg = ""
        
        if args.evaluate :
            evaluate_arg = "--evaluate"
        elif args.evaluate_baseline :
            print("evaluating baseline")
            evaluate_arg = "--evaluate-baseline"
        else:
            assert args.evaluate_random_baseline
            evaluate_arg = "--evaluate-random-baseline"

        if args.verbose>0 :
            verbosity_arg = "-"+"v"*args.verbose
        else :
            verbosity_arg = ""

        if args.set_switch :
            setswitchargs = ""
        else :
            setswitchargs = "--no-set-switch"

        if args.include_proof_relevant :
            proofrelevantargs = "--include-proof-relevant"
        else :
            proofrelevantargs = ""

        if args.blacklisted_tactics:
            blacklistedtacticsarg = " ".join( [ f"--blacklisted-tactic {tactic}" for tactic in args.blacklisted_tactics] )
        else :
            blacklistedtacticsarg = ""
    
    
        output_arg = f"-r {args.rl_weights}"
        prelude_arg = f"--prelude {str(args.prelude)}"
        backend_arg = f"--backend {args.backend}"
        if args.weightsfile:
            supervised_weights_arg = f"--supervised-weights {str(args.weightsfile)}"
        else:
            supervised_weights_arg = ""
        coq2vecweightsarg = f"--coq2vec-weights {str(args.coq2vec_weights)}"
        predictionarg = f"-p {args.num_predictions}"
        maxsertopworkersarg = f"--max-sertop-workers {args.max_sertop_workers}"
        statedirargs = f"--state-dir {str(args.state_dir)}"
    
    
        with open("submit/submit_multi_eval_jobs.sh","w") as f:
            submit_script = f"""#!/bin/bash
#
#SBATCH --job-name=Rl_eval_{unique_id}
#SBATCH -p {args.partition}
#SBATCH --mem={args.mem}
#SBATCH -o {args.state_dir}/worker-%a_output.txt
#SBATCH --array=1-{num_workers_actually_needed}
#SBATCH --time={args.worker_alive_time}


module add opam/2.1.2
python -u src/distributed_rl_eval_cluster_worker.py {supervised_weights_arg} {coq2vecweightsarg} \
    compcert_projs_splits.json {prelude_arg} {backend_arg} {output_arg} {task_file_arg} {evaluate_arg} \
     {verbosity_arg} {predictionarg} {setswitchargs} {proofrelevantargs} {maxsertopworkersarg} {statedirargs} \
     {blacklistedtacticsarg} -s {args.steps_per_episode} 
"""
            f.write(submit_script)
        subprocess.run(f'sbatch submit/submit_multi_eval_jobs.sh', shell=True)
        print("Submitted Jobs")
        track_progress(args, len(all_tasks))

        print("Finished Evaluation")
    check_success(args, all_tasks)

def get_jobs_done(args: argparse.Namespace) -> int:
    jobs_done = 0
    for worker_progress_file in glob(str(args.state_dir / f"progress-*.txt")):
        with open(worker_progress_file, 'r') as f:
            jobs_done += sum(1 for _ in f)
    return jobs_done

def track_progress(args: argparse.Namespace, total_num_tasks: int) -> None:
    jobs_done = get_jobs_done(args)
    with tqdm(desc="Jobs finished", total=total_num_tasks, initial=jobs_done, dynamic_ncols=True,position=0,leave=True) as bar :
        while True :
            time.sleep(0.1)
            new_jobs_done = get_jobs_done(args)
            bar.update(new_jobs_done - jobs_done)
            jobs_done = new_jobs_done

            if jobs_done == total_num_tasks:
                break

def check_success(args: argparse.Namespace, all_tasks) -> None:
    total_num_jobs_successful = 0
    total_num_jobs = len(all_tasks)
    finished_tasks = defaultdict(list)
    original_tasks = defaultdict(list)
    for finished_file in glob(str(args.state_dir / "finished-*.txt")):
        with open(finished_file, 'r') as f:
            for line in f:
                job = json.loads(line)
                finished_tasks[int(job["target_length"])].append(job)
                total_num_jobs_successful += 1
    
    for task in all_tasks :
        original_tasks[task.target_length].append(task)

    print(f"Jobs Succesfully Solved : {total_num_jobs_successful}/{total_num_jobs} = { '%.2f'% (100*total_num_jobs_successful/total_num_jobs) }%")

    task_lengths = sorted(original_tasks.keys())
    for task_length in task_lengths :
        num_finished_tasks = sum( [ 1 for _ in finished_tasks[task_length] ])
        num_total_tasks = sum( [ 1 for _ in original_tasks[task_length] ])
        print(f"Task Length {task_length} : {num_finished_tasks}/{num_total_tasks} = {'%.2f'%(100*num_finished_tasks/num_total_tasks)}%")


def cancel_workers(args: argparse.Namespace, unique_id: uuid.UUID) -> None:
    os.system(f"scancel -n RL_eval_{unique_id}")



def setup_eval_jobstate(args: argparse.Namespace) -> int:
    print("Resuming? ", args.eval_resume)
    print("State dir", args.state_dir)
    if not args.eval_resume :
        if (args.state_dir).exists() :
            shutil.rmtree(str(args.state_dir))
            
    (args.state_dir).mkdir(exist_ok=True)
    (args.state_dir / args.workers_output_dir).mkdir(exist_ok=True)
    (args.state_dir / "taken").mkdir(exist_ok=True)
    taken_path = args.state_dir / "taken" / "taken-files.txt"
    if not taken_path.exists():
        with taken_path.open('w'):
            pass

    done_tasks = []
    for workerid in range(1, args.num_eval_workers + 1):
        worker_done_tasks = []
        progress_path = args.state_dir / f"progress-{workerid}.txt"
        if not progress_path.exists():
            with progress_path.open("w"):
                pass
        else:
            with progress_path.open('r') as f:
                worker_done_tasks = [RLTask(**task_dict)
                                        for line in f
                                        for task_dict in (json.loads(line),)]
            done_tasks += worker_done_tasks
        taken_path = args.state_dir / "taken" / f"taken-{workerid}.txt"
        with taken_path.open("w") as f:
            pass
    
    file_taken_dict: Dict[Path, List[RLTask]] = {}
    for task in done_tasks:
        if Path(task.src_file) in file_taken_dict:
            file_taken_dict[Path(task.src_file)].append(task)
        else:
            file_taken_dict[Path(task.src_file)] = [task]
    all_tasks = get_all_tasks(args)
    tasks_idx_dict = {task : idx for idx, task in enumerate(all_tasks)}
    for fidx, filename in enumerate(get_all_files(args)):
        with (args.state_dir / "taken" /
              ("file-" + util.safe_abbrev(filename,
                                file_taken_dict.keys()) + ".txt")).open("w") as f:
            for tidx, task in enumerate(file_taken_dict.get(filename, [])):
                try:
                    task_idx = tasks_idx_dict[task]
                except KeyError:
                    util.eprint(f"File number {fidx}, task number {tidx}")
                    for dict_task, dict_ep in tasks_idx_dict.keys():
                        if task.to_proof_spec() == dict_task.to_proof_spec():
                            util.eprint("There is a task with a matching proof spec!")
                            break
                    raise
                print(json.dumps((task_idx, False)), file=f, flush=True)


    with (args.state_dir / "workers_scheduled.txt").open('w') as f:
        pass
    return len(done_tasks)



def get_all_tasks(args: argparse.Namespace) -> List[RLTask]:
    if args.tasks_file:
        with open(args.tasks_file, 'r') as f:
            all_tasks = [RLTask(**json.loads(line)) for line in f]
    else:
        all_tasks = [RLTask.from_job(job) for job in get_all_jobs(args)]
    return all_tasks

def get_all_files(args: argparse.Namespace) -> List[Path]:
    if args.tasks_file:
        with open(args.tasks_file, 'r') as f:
            return list({Path(json.loads(line)["src_file"]) for line in f})
    else:
        project_dicts = project_dicts_from_args(args)
        return [Path(filename) for project_dict in project_dicts
                for filename in project_dict["test_files"]]


if __name__ == '__main__' :
    main()
