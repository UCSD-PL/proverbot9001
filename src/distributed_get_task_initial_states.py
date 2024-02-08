#!/usr/bin/env python
import argparse
from collections import defaultdict
from glob import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Dict, List
from pathlib import Path
import uuid
from tqdm import tqdm
from util import FileLock
from distributed_rl import get_all_files
import util
from gen_rl_tasks import RLTask
from search_file_cluster import get_all_jobs_cluster
from search_worker import ReportJob
from util import eprint

def main() -> None:
  argparser = argparse.ArgumentParser()
  argparser.add_argument("-v", "--verbose", action='count', default=0)
  argparser.add_argument("--print-timings", action='store_true')
  argparser.add_argument("--prelude", default="./CompCert", type=Path)
  argparser.add_argument("--resume", action='store_true', dest="resume")
  argparser.add_argument("tasks_file", type=Path)
  argparser.add_argument("output_file", type=Path)
  argparser.add_argument("--num-workers", type=int, default=16)
  argparser.add_argument("--state-dir", default="drl_taskinitobl_state", type=Path)
  argparser.add_argument("--worker-alive-time", type=str, default="12:00:00")
  argparser.add_argument("--mem", default="4G")
  args = argparser.parse_args()
  args.backend = "serapi"
  args.set_switch = True

  
  
  get_states(args)


def get_states(args: argparse.Namespace, unique_id: uuid.UUID = uuid.uuid4()) -> None:

	def signal_handler(sig, frame) -> None:
		print('\nProcess Interrupted, Cancelling all Workers')
		result = subprocess.run(f'scancel --name Rl_getstate_{unique_id}', shell=True, capture_output=True, text = True)
		print(result.stdout)
		sys.exit(0)

	signal.signal(signal.SIGINT, signal_handler)

	os.makedirs(args.state_dir, exist_ok=True)

	with args.tasks_file.open("r") as f:
		all_tasks = [RLTask(**json.loads(line)) for line in f]
	
	num_tasks_done = setup_jobstate(args, all_tasks)
	num_workers_actually_needed = min(len(all_tasks) - num_tasks_done,
									  args.num_workers)
	print(f"Deploying {num_workers_actually_needed} workers")
	

	if num_workers_actually_needed > 0:
		if args.verbose>0 :
			verbosity_arg = "-"+"v"*args.verbose
		else :
			verbosity_arg = ""
	
		task_file_arg = str(args.tasks_file)
		output_arg = str(args.output_file)
		prelude_arg = f"--prelude {str(args.prelude)}"
		statedirargs = f"--state-dir {str(args.state_dir)}"
	
	
		with open(str(args.state_dir) + "/submit_taskinitobl_jobs.sh","w") as f:
			submit_script = f"""#!/bin/bash
#
#SBATCH --job-name=Rl_getstate_{unique_id}

#SBATCH --mem={args.mem}
#SBATCH -o {args.state_dir}/worker-%a_output.txt
#SBATCH --array=1-{num_workers_actually_needed}
#SBATCH --time={args.worker_alive_time}


module load opam/2.1.2 graphviz/2.49.0+py3.8.12 openmpi/4.1.3+cuda11.6.2
python -u src/distributed_get_task_initial_states_worker.py {task_file_arg} {output_arg}  {prelude_arg} \
	 {verbosity_arg} {statedirargs} \
"""
			f.write(submit_script)
		subprocess.run('sbatch '+ str(args.state_dir) + "/submit_taskinitobl_jobs.sh" , shell=True)
		print("Submitted Jobs")
		track_progress(args, len(all_tasks))

		print("Finished Getting all initial states")
	merge(args)




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

def merge(args) :
	all_lines = []
	for finished_file in glob(str(args.state_dir / "finished-*.txt")):
		with open(finished_file, 'r') as fileread:
			for line in fileread:
				all_lines.append(line)
	
	with open(args.output_file,'w') as f, FileLock(f) :
		f.writelines(all_lines)
				

def check_success(args: argparse.Namespace, all_tasks) -> None:
	if args.tasks_file :
		total_num_jobs_successful = 0
		total_num_jobs = len(all_tasks)
		finished_tasks = defaultdict(list)
		original_tasks = defaultdict(list)
		
		
		for task in all_tasks :
			original_tasks[task.target_length].append(task)

		print(f"Jobs Succesfully Solved : {total_num_jobs_successful}/{total_num_jobs} = { '%.2f'% (100*total_num_jobs_successful/total_num_jobs) }%")


		task_lengths = sorted(original_tasks.keys())
		for task_length in task_lengths :
			num_finished_tasks = sum( [ 1 for _ in finished_tasks[task_length] ])
			num_total_tasks = sum( [ 1 for _ in original_tasks[task_length] ])
			print(f"Task Length {task_length} : {num_finished_tasks}/{num_total_tasks} = {'%.2f'%(100*num_finished_tasks/num_total_tasks)}%")
	else :
		total_num_jobs_successful = 0
		total_num_jobs = len(all_tasks)
		for finished_file in glob(str(args.state_dir / "finished-*.txt")):
			with open(finished_file, 'r') as f:
				for line in f:
					total_num_jobs_successful += 1
		print(f"Jobs Succesfully Solved : {total_num_jobs_successful}/{total_num_jobs} = { '%.2f'% (100*total_num_jobs_successful/total_num_jobs) }%")

def cancel_workers(args: argparse.Namespace, unique_id: uuid.UUID) -> None:
	os.system(f"scancel -n RL_eval_{unique_id}")



def setup_jobstate(args: argparse.Namespace, all_tasks) -> int:
	if not args.resume :
		if (args.state_dir).exists() :
			shutil.rmtree(str(args.state_dir))
			
	(args.state_dir).mkdir(exist_ok=True)
	(args.state_dir / "taken").mkdir(exist_ok=True)
	taken_path = args.state_dir / "taken" / "taken-files.txt"
	if not taken_path.exists():
		with taken_path.open('w'):
			pass

	done_tasks = []
	for workerid in range(1, args.num_workers + 1):
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




if __name__ == "__main__":
  main()
