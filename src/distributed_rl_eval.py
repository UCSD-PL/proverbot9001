import argparse
import json
import random
import uuid
import torch
import os
import subprocess
import time
import util
import shutil


from gen_rl_tasks import RLTask


from util import (nostderr, FileLock, eprint,
                  print_time, unwrap, safe_abbrev)
from pathlib import Path
from glob import glob
from typing import List, Tuple, Optional, Dict, Set
from tqdm import tqdm


def main() :
    eprint("Starting main")
    parser = argparse.ArgumentParser(
        description="Evaluation worker - The rl agent"
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
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--no-interleave", dest="interleave", action="store_false")
    parser.add_argument('--supervised-weights', type=Path, dest="weightsfile")
    parser.add_argument("--coq2vec-weights", type=Path)
    parser.add_argument("--max-sertop-workers", default=16, type=int)
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
    parser.add_argument("--train-every", default=1, type=int)
    parser.add_argument("--print-loss-every", default=None, type=int)
    parser.add_argument("--curriculum",action="store_true")
    parser.add_argument("--sync-target-every",
                        help="Sync target network to v network every <n> episodes",
                        default=10, type=int)
    parser.add_argument("--allow-partial-batches", action='store_true')
    parser.add_argument("--blacklist-tactic", action="append",
                        dest="blacklisted_tactics")
    parser.add_argument("--resume", choices=["no", "yes", "ask"], default="ask")
    evalGroup = parser.add_mutually_exclusive_group()
    evalGroup.add_argument("--evaluate", action="store_true")
    evalGroup.add_argument("--evaluate-baseline", action="store_true")
    parser.add_argument("--state-dir", default="drl_state", type=Path)
    parser.add_argument("--workers-output-dir", default=Path("output"), type=Path)
    parser.add_argument("--num-eval-workers", default=1, type=int)
    
    args = parser.parse_args()
    
    evaluate(args)



def evaluate(args) :
    setup_eval_jobstate(args)
    print(f"Deploying {args.num_eval_workers} workers")
    unique_id = uuid.uuid4()

    if args.tasks_file :
        task_file_arg = f"--tasks-file {str(args.tasks_file)}"
    else :
        raise ValueError("Non task file is not verified yet")
    
    if not (args.evaluate or args.evaluate_baseline) :
        raise ValueError("Evaluation function called without evaluate arguments passed. What type of Evaluation do you want? use either --evaluate or --evaluate-baseline")
    else :
        if args.evaluate :
            evaluate_arg = "--evaluate"
        elif args.evaluate_baseline :
            evaluate_arg = "--evaluate-baseline"
    
    if args.verbose>0 :
        verbosity_arg = "-"+"v"*args.verbose
    else :
        verbosity_arg = ""
    
    
    output_arg = f"-o {str(args.output_file)}"
    prelude_arg = f"--prelude {str(args.prelude)}"
    backend_arg = f"--backend {args.backend}"
    supervised_weights_arg = f"--supervised-weights {str(args.weightsfile)}"
    coq2vecweightsarg = f"--coq2vec-weights {str(args.coq2vec_weights)}"


    with open("submit/submit_multi_eval_jobs.sh","w") as f:
        submit_script = f"""#!/bin/bash
#
#SBATCH --job-name=Rl_eval_{unique_id}
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH -o {args.state_dir}/eval/slurm-%A_%a_out.txt
#SBATCH -e {args.state_dir}/eval/slurm-%A_%a_err.txt
#SBATCH --array=1-{args.num_eval_workers}


module add opam/2.1.2
python -u src/distributed_rl_eval_cluster_worker.py {supervised_weights_arg} {coq2vecweightsarg} \
    compcert_projs_splits.json {prelude_arg} {backend_arg} {output_arg} {task_file_arg} {evaluate_arg} {verbosity_arg}
"""
        f.write(submit_script)
    subprocess.run(f'sbatch submit/submit_multi_eval_jobs.sh', shell=True)
    track_progress(args)

    print("Finished Evaluation")
    check_success(args)




def track_progress(args) :
    with Path(args.tasks_file).open("r+") as f, FileLock(f):
        total_num_jobs = sum(1 for _ in f)
    
    with tqdm(desc="Jobs finished", total=total_num_jobs, initial=0, dynamic_ncols=True,position=0,leave=True) as bar :
        jobs_done = 0
        while True :
            new_jobs_done = 0
            for workeridx in range(1, args.num_eval_workers + 1) :
                with (args.state_dir / "eval" /"taken" / f"taken-{workeridx}.txt").open("r+") as f, FileLock(f):
                    new_jobs_done += sum(1 for _ in f)

            
            bar.update(new_jobs_done - jobs_done)
            
            jobs_done = new_jobs_done

            if jobs_done == total_num_jobs :
                break
        time.sleep(3)
    

def check_success(args) :
    with Path(args.tasks_file).open("r+") as f, FileLock(f):
        total_num_jobs = sum(1 for _ in f)
    total_num_jobs_successful = 0
    for workerid in range(1, args.num_eval_workers + 1) :
        with (args.state_dir / "eval" / f"finished-{workerid}.txt").open('r') as f, FileLock(f):
                total_num_jobs_successful += sum(1 for _ in f)
    print(f"Jobs Succesfully Solved : {total_num_jobs_successful}/{total_num_jobs} = { '%.2f'% (100*total_num_jobs_successful/total_num_jobs) }%")


def cancel_workers(args, unique_id) :
    os.system(f"scancel -n RL_eval_{unique_id}")



def setup_eval_jobstate(args: argparse.Namespace) -> None:

    if (args.state_dir / "eval").exists() :
        shutil.rmtree(str(args.state_dir) + "/eval")
    (args.state_dir).mkdir(exist_ok=True)
    (args.state_dir/ "eval").mkdir(exist_ok=True)
    (args.state_dir / "eval" / args.workers_output_dir).mkdir(exist_ok=True)
    (args.state_dir / "eval" / "taken").mkdir(exist_ok=True)
    taken_path = args.state_dir / "eval" / "taken" / "taken-files.txt"
    if not taken_path.exists():
        with taken_path.open('w'):
            pass

    for workerid in range(1, args.num_eval_workers + 1):
        done_path = args.state_dir / "eval" / f"done-{workerid}.txt"
        with done_path.open("w"):
            pass
        taken_path = args.state_dir / "eval" / "taken" / f"taken-{workerid}.txt"
        with taken_path.open("w") as f:
            pass
        progress_path = args.state_dir / "eval" / f"progress-{workerid}.txt"
        with progress_path.open("w") as f:
            pass

   
    for fidx, filename in enumerate(get_all_files(args)):
        with (args.state_dir / "eval" /  "taken" /
              ("file-" + util.safe_abbrev(filename,
                                []) + ".txt")).open("w") as f:
            pass
    with (args.state_dir / "eval" / "workers_scheduled.txt").open('w') as f:
        pass



def get_all_tasks(args) :    
    with open(args.tasks_file, 'r') as f:
        all_tasks = [RLTask(**json.loads(line)) for line in f]
    return all_tasks

def get_all_files(args: argparse.Namespace) -> List[Path]:
    with open(args.tasks_file, 'r') as f:
        return list({Path(json.loads(line)["src_file"]) for line in f})


if __name__ == '__main__' :
    main()
