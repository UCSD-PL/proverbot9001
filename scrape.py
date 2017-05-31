#!/usr/bin/env python3

import subprocess
import threading
import re
import queue
import os
import os.path
import argparse
import sys
# This dependency is in pip, the python package manager
from sexpdata import *
from traceback import *
from format import format_context, format_tactic
from helper import *

import linearize_semicolons
import serapi_instance

num_jobs = 0
jobs = queue.Queue()
workers = []
output_lock = threading.Lock()
finished_queue = queue.Queue()

options = {}

class Worker(threading.Thread):
    def __init__(self, output, workerid, prelude="."):
        threading.Thread.__init__(self, daemon=True)
        self.coqargs = [os.path.dirname(os.path.abspath(__file__)) + "/coq-serapi/sertop.native",
                   "--prelude={}/coq".format(os.path.dirname(os.path.abspath(__file__)))]
        includes=subprocess.Popen(['make', '-C', prelude, 'print-includes'], stdout=subprocess.PIPE).communicate()[0]
        self.tmpfile_name = "/tmp/proverbot_worker{}".format(workerid)
        self.workerid = workerid
        with open(self.tmpfile_name, 'w') as tmp_file:
            tmp_file.write('')
        self.outbuf = ""
        self.includes=includes.strip().decode('utf-8')
        if output == None:
            self.fout = sys.stdout
        else:
            self.fout = open(args.output, 'w')
        pass

    def process_statement(self, coq, command):
        if re.match(";", command) and options["no-semis"]:
            return
        in_proof = coq.proof_context
        if in_proof:
            prev_tactics = coq.prev_tactics
            prev_hyps = coq.get_hypothesis()
            prev_goal = coq.get_goals()
            with open(self.tmpfile_name, 'a') as tmp_file:
                tmp_file.write(format_context(coq.prev_tactics, coq.get_hypothesis(),
                                              coq.get_goals()))
            coq.run_stmt(command)
            with open(self.tmpfile_name, 'a') as tmp_file:
                tmp_file.write(format_tactic(command))
        else:
            coq.run_stmt(command)

    def process_file(self, filename):
        try:
            commands = lift_and_linearize(load_commands(filename),
                                          self.coqargs, self.includes,
                                          filename)
            with serapi_instance.SerapiContext(self.coqargs, self.includes) as coq:
                for command in commands:
                    self.process_statement(coq, command)
        except Exception as e:
            print("In file {}:".format(filename))
            raise e

        output_lock.acquire()
        with open(self.tmpfile_name, 'r') as tmp_file:
            for line in tmp_file:
                self.fout.write(line)
        with open(self.tmpfile_name, 'w') as tmp_file:
            tmp_file.write('')
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

parser = argparse.ArgumentParser(description="scrape a proof")
parser.add_argument('-o', '--output', help="output data file name", default=None)
parser.add_argument('-j', '--threads', default=1, type=int)
parser.add_argument('--prelude', default=".")
parser.add_argument('--no-semis', default=False, const=True, action='store_const',
                    dest='no_semis')
parser.add_argument('inputs', nargs="+", help="proof file name(s) (*.v)")
args = parser.parse_args()
options["no-semis"] = args.no_semis

num_jobs = len(args.inputs)

for infname in args.inputs:
    jobs.put(infname)

for idx in range(args.threads):
    worker = Worker(args.output, idx, args.prelude)
    worker.start()
    workers.append(worker)

for idx in range(args.threads):
    finished_id = finished_queue.get()
    workers[finished_id].join()
