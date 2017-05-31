#!/usr/bin/env python3

import argparse
import subprocess
import os
import sys
import math
import threading
import queue
import re

from format import format_context

import serapi_instance
import linearize_semicolons

from serapi_instance import count_fg_goals
from helper import *

num_jobs = 0
jobs = queue.Queue()
workers = []
num_tactics = 0
num_correct = 0
output_lock = threading.Lock()
finished_queue = queue.Queue()

class Worker(threading.Thread):
    def __init__(self, workerid, coqargs, includes):
        threading.Thread.__init__(self, daemon=True)
        self.coqargs = coqargs
        self.includes = includes
        self.workerid = workerid
        pass

    def process_file(self, filename):
        global num_tactics
        global num_correct
        num_tactics_in_file = 0
        num_correct_in_file = 0

        commands = lift_and_linearize(load_commands(filename),
                                      coqargs, includes, filename)
        with serapi_instance.SerapiContext(self.coqargs, self.includes) as coq:
            for command in commands:
                if re.match(";", command) and options["no-semis"]:
                    return
                in_proof = coq.proof_context
                if in_proof:
                    prev_goal = coq.get_goals()
                    prev_hyps = coq.get_hypothesis()
                    prev_tactics = coq.prev_tactics
                    num_tactics_in_file += 1
                    query = format_context(prev_tactics, prev_hyps, prev_goal)
                    response, errors = subprocess.Popen(darknet_command,
                                                        stdin=subprocess.PIPE,
                                                        stdout=subprocess.PIPE,
                                                        stderr=subprocess.PIPE
                    ).communicate(input=query.encode('utf-8'))
                    result = response.decode('utf-8').strip()
                    if command == result:
                        num_correct_in_file += 1

                coq.run_stmt(command)
        output_lock.acquire()
        num_tactics += num_tactics_in_file
        num_correct += num_correct_in_file
        output_lock.release()
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
        finally:
            finished_queue.put(self.workerid)


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

for idx in range(args.threads):
    finished_id = finished_queue.get()
    workers[finished_id].join()

print("Accuracy: %{} ({}/{})".format(math.floor(num_correct / num_tactics * 100),
                                     num_correct, num_tactics))
