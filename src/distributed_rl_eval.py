import argparse
import json
import random
import uuid
import torch
import os
import subprocess
import time
from tqdm import tqdm

from util import eprint, FileLock
from pathlib import Path





def main() :
    eprint("Starting main")
    parser = argparse.ArgumentParser(
        description="Evaluation worker - The rl agent"
        "to complete proofs using Proverbot9001.")
    # parser.add_argument("--prelude", default=".", type=Path)
    # parser.add_argument("--output", "-o", dest="output_file",
    #                     help="output data folder name",
    #                     default="data/rl_weights.dat",
    #                     type=Path)
    # parser.add_argument("--verbose", "-v", help="verbose output",
    #                     action="count", default=0)
    # parser.add_argument("--progress", "-P", help="show progress of files",
    #                     action='store_true')
    # parser.add_argument("--print-timings", "-t", action='store_true')
    # parser.add_argument("--no-set-switch", dest="set_switch", action='store_false')
    # parser.add_argument("--include-proof-relevant", action="store_true")
    # parser.add_argument("--backend", choices=['serapi', 'lsp', 'auto'], default='auto')
    # parser.add_argument('filenames', help="proof file name (*.v)",
    #                     nargs='+', type=Path)
    # proofsGroup = parser.add_mutually_exclusive_group()
    # proofsGroup.add_argument("--proof", default=None)
    # proofsGroup.add_argument("--proofs-file", default=None)
    parser.add_argument("--tasks-file", default=None)
    # parser.add_argument("--test-file", default=None)
    # parser.add_argument("--no-interleave", dest="interleave", action="store_false")
    # parser.add_argument('--supervised-weights', type=Path, dest="weightsfile")
    # parser.add_argument("--coq2vec-weights", type=Path)
    # parser.add_argument("--max-sertop-workers", default=16, type=int)
    # parser.add_argument("-l", "--learning-rate", default=2.5e-4, type=float)
    # parser.add_argument("-g", "--gamma", default=0.9, type=float)
    # parser.add_argument("--starting-epsilon", default=0, type=float)
    # parser.add_argument("--ending-epsilon", default=1.0, type=float)
    # parser.add_argument("-s", "--steps-per-episode", default=16, type=int)
    # parser.add_argument("-n", "--num-episodes", default=1, type=int)
    # parser.add_argument("-b", "--batch-size", default=64, type=int)
    # parser.add_argument("-w", "--window-size", default=2560)
    # parser.add_argument("-p", "--num-predictions", default=5, type=int)
    # parser.add_argument("--batch-step", default=25, type=int)
    # parser.add_argument("--lr-step", default=0.8, type=float)
    # parser.add_argument("--batches-per-proof", default=1, type=int)
    # parser.add_argument("--train-every", default=1, type=int)
    # parser.add_argument("--print-loss-every", default=None, type=int)
    # parser.add_argument("--curriculum",action="store_true")
    # parser.add_argument("--sync-target-every",
    #                     help="Sync target network to v network every <n> episodes",
    #                     default=10, type=int)
    # parser.add_argument("--allow-partial-batches", action='store_true')
    # parser.add_argument("--blacklist-tactic", action="append",
    #                     dest="blacklisted_tactics")
    # parser.add_argument("--resume", choices=["no", "yes", "ask"], default="ask")
    # parser.add_argument("--num-eval-workers", type=int, default=5)
    # evalGroup = parser.add_mutually_exclusive_group()
    # evalGroup.add_argument("--evaluate", action="store_true")
    # evalGroup.add_argument("--evaluate-baseline", action="store_true")
    parser.add_argument("--state-dir", default="drl_state/eval", type=Path)
    parser.add_argument("--num-eval-workers", default=1, type=int)
    
    args = parser.parse_args()
    evaluate(args)



def evaluate(args) :
    print(f"Deploying {args.num_eval_workers} workers")
    unique_id = uuid.uuid4()
    with open("submit/submit_multi_eval_jobs.sh","w") as f:
        submit_script = f"""#!/bin/bash
#
#SBATCH --job-name=Rl_eval_{unique_id}
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH -o drl_state/eval/slurm-%A_%a_out.txt
#SBATCH -e drl_state/eval/slurm-%A_%a_err.txt
#SBATCH --array=1-{args.num_eval_workers}


module add opam/2.1.2
python -u src/distributed_rl_eval_cluster_worker.py --supervised-weights=data/polyarg-weights-develop.dat --coq2vec-weights=data/term2vec-weights-59.dat \
    compcert_projs_splits.json --prelude=./CompCert --backend=serapi -o testy.dat --tasks-file={str(args.tasks_file)} --evaluate
"""
        f.write(submit_script)
    set_up_files(args)
    subprocess.run(f'sbatch submit/submit_multi_eval_jobs.sh', shell=True)
    track_progress(args)

    print("Finished Evaluation")
    


def set_up_files(args) :
    os.system(f"rm -f {args.state_dir}/taken*")
    os.system(f"rm -f {args.state_dir}/finished*")
    with (args.state_dir / f"taken.txt").open("w") as f:
        pass

    taken_path = args.state_dir / "taken-files.txt"
    if not taken_path.exists():
        with taken_path.open('w'):
            pass



def track_progress(args) :
    with Path(args.tasks_file).open("r+") as f, FileLock(f):
        total_num_jobs = sum(1 for _ in f)
    
    with tqdm(desc="Jobs finished", total=total_num_jobs, initial=0, dynamic_ncols=True,position=0,leave=True) as bar :
        jobs_done = 0
        while True :
            with (args.state_dir / "taken.txt").open("r+") as f, FileLock(f):
                new_jobs_done = sum(1 for _ in f)

            
            bar.update(new_jobs_done - jobs_done)
            
            jobs_done = new_jobs_done

            if jobs_done == total_num_jobs :
                break
        time.sleep(3)
    

def check_success(args) :
    with Path(args.tasks_file).open("r+") as f, FileLock(f):
        total_num_jobs = sum(1 for _ in f)
    total_num_jobs_successful = 0
    for workerid in range(args.num_eval_workers) :
        with (args.state_dir / f"finished-{workerid}.txt").open('a') as f:
                total_num_jobs_successful += sum(1 for _ in f)
    print(f"Jobs Succesfully Solved : {total_num_jobs_successful} / {total_num_jobs} = {100*total_num_jobs_successful/total_num_jobs}%")


def cancel_workers(args, unique_id) :
    os.system(f"scancel -n RL_eval_{unique_id}")

if __name__ == '__main__' :
    main()


# def dispatch_evaluation_workers(args, switch_dict, tasks) :
    
#     return_queue = Queue()
   
#     processes = []
#     for i in range(args.num_eval_workers) :
#         processes.append(Process(target = evaluation_worker_monitor, args = (args, switch_dict,
#                                  tasks[len(tasks)*i//args.num_eval_workers : len(tasks)*(i+1)//args.num_eval_workers], return_queue)))
#         processes[-1].start()
#         print("Process", i, "started")
        
#     for process in processes :
#         process.join()
    
#     results = []
#     while not return_queue.empty() :
#         results.append(return_queue.get())

#     assert len(results) == args.num_eval_workers, "Here's the queue : " + str(results)

#     proofs_completed = sum(results)
#     print(f"{proofs_completed} out of {len(tasks)} "
#           f"tasks successfully proven "
#           f"({stringified_percent(proofs_completed, len(tasks))}%)")


# def evaluation_worker_monitor( args, switch_dict, task_set, return_queue, worker_ind) :
#     task_file_path = f"output/worker/worker{worker_ind}.json"
#     os.makedirs(task_file_path, exist_ok= True)
#     with open(task_file_path,"w") as writefile:
#         for task in task_set :
#             writefile.write(json.dumps(task) + "\n")
    
    
#     taskfilestr = f"--task-file {task_file_path}" if args.task_file else ""
#     testfilestr = f"--test-file {task_file_path}" if args.task_file else ""


#     process = subprocess.run(f"module add opam/2.1.2; srun -p gpu --gpus 1 --mem 8G python\
#                              rl_evaluation_cluster_worker.py {taskfilestr}",  capture_output=True, text = True)
#     parse_output = process.stdout.split()
#     try :
#         result_ind = parse_output.index("Solution:") + 1
#         results = int(parse_output[result_ind])
#     except (ValueError, IndexError) as e:
#         eprint(e)
#         eprint("Worker Crashed or didn't give any results. Following lines are worker output.")
#         eprint(process.stdout)

#     return_queue.put(results)
#     return
