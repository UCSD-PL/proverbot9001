#!/usr/bin/env python3

import argparse
import subprocess
import os
import sys
import math
import threading
import queue
import re

from shutil import *
from format import format_command_record

import serapi_instance
import linearize_semicolons

from serapi_instance import count_fg_goals
from helper import lift_and_linearize

num_jobs = 0
jobs = queue.Queue()
workers = []
num_tactics = 0
num_correct = 0
output_lock = threading.Lock()

class Worker(threading.Thread):
    def __init__(self, workerid, coqargs, includes):
        threading.Thread.__init__(self, daemon=True)
        self.coqargs = coqargs
        self.includes = includes
        pass

    def process_file(self, filename):
        num_tactics_in_file = 0
        num_correct_in_file = 0
        with open(filename, 'r') as fin:
            contents = serapi_instance.kill_comments(fin.read())
        commands_orig = serapi_instance.split_commands(contents)
        commands_preprocessed = [newcmd for cmd in commands_orig
                                 for newcmd in serapi_instance.preprocess_command(cmd)]
        commands = lift_and_linearize(commands_preprocessed,
                                      coqargs, includes, filename)
        with serapi_instance.SerapiContext(self.coqargs, self.includes) as coq:
            for command in commands:
                if re.match(";", command) and options["no-semis"]:
                    return
                in_proof = coq.proof_context
                if in_proof:
                    prev_goals = coq.get_goals()
                    prev_hyps = coq.get_hypothesis()
                    prev_tactics = coq.prev_tactics
                    num_tactics_in_file += 1
                    response, errors = subprocess.Popen(darknet_command,
                                                        stdin=subprocess.PIPE,
                                                        stdout=subprocess.PIPE,
                                                        stderr=subprocess.PIPE
                    ).communicate(input=query.encode('utf-8'))
                    result = response.decode('utf-8').strip()
                    if command == result:
                        num_correct_in_file += 1

                    coq.run_stmt(command)
                    still_in_proof = count_fg_goals(coq) != 0
                    if still_in_proof:
                        post_goal = coq.get_goals()
                        post_hyps = coq.get_hypothesis()
                        query += format_command_record(prev_tactics, prev_hyps, prev_goal,
                                                       command, post_hyps, post_goal)
                else:
                    coq.run_stmt(command)
        output_lock.acquire()
        num_tactics += num_tactics_in_file
        num_correct += num_correct_in_file
    def run(self):
        try:
            while(True):
                job = jobs.get_nowait()
                print("Processing file {} ({} of {})".format(job,
                                                             num_jobs - jobs.qsize(),
                                                             num_jobs))
                self.process_file(job)
        except queue.Empty:
            pass


parser = argparse.ArgumentParser(description=
                                 "try to match the file by predicting a tacti")
parser.add_argument('-j', '--threads', default=1, type=int)
parser.add_argument('--prelude', default=".")
parser.add_argument('filenames', nargs="+", help="proof file name (*.v)")
args = parser.parse_args()
filenames = args.filenames

base = os.path.dirname(os.path.abspath(__file__))
darknet_command = ["{}/try-auto.py".format(base)]
coqargs = ["{}/coq-serapi/sertop.native".format(base),
           "--prelude={}/coq".format(base)]
includes = subprocess.Popen(['make', '-C', args.prelude, 'print-includes'],
                            stdout=subprocess.PIPE).communicate()[0].decode('utf-8')

num_jobs = len(args.filenames)
for infname in args.filenames:
    jobs.put(infname)

for idx in range(args.threads):
    worker = Worker(idx, coqargs, includes)
    worker.start()
    workers.append(worker)

for worker in workers:
    worker.join()

print("Accuracy: %{} ({}/{})".format(math.floor(num_correct / num_tactics * 100),
                                     num_correct, num_tactics))
