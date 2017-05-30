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
from format import format_command_record

import linearize_semicolons
import serapi_instance

num_jobs = 0
jobs = queue.Queue()
workers = []
output_lock = threading.Lock()

options = {}

class Worker(threading.Thread):
    def __init__(self, output, workerid, prelude="."):
        threading.Thread.__init__(self, daemon=True)
        self.coqargs = [os.path.dirname(os.path.abspath(__file__)) + "/coq-serapi/sertop.native",
                   "--prelude={}/coq".format(os.path.dirname(os.path.abspath(__file__)))]
        includes=subprocess.Popen(['make', '-C', prelude, 'print-includes'], stdout=subprocess.PIPE).communicate()[0]
        self.tmpfile_name = "/tmp/proverbot_worker{}".format(workerid)
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
        self.outbuf = ""
        if coq.proof_context:
            prev_tactics = coq.prev_tactics
            prev_hyps = coq.get_hypothesis()
            prev_goal = coq.get_goals()
            self.outbuf += "*****\n"
            self.outbuf += prev_goal + "\n"
            self.outbuf += "+++++\n"
            self.outbuf += command + "\n"
        else:
            prev_goal = None
        coq.run_stmt(command)


        if coq.proof_context and prev_goal:
            post_hyps = coq.get_hypothesis()
            post_goal = coq.get_goals()
            self.outbuf += "-----\n"
            self.outbuf += post_goal + "\n"
        with open(self.tmpfile_name, 'a') as tmp_file:
            tmp_file.write(self.outbuf)

    def process_file(self, filename):
        try:
            with open(filename, 'r') as fin:
                contents = serapi_instance.kill_comments(fin.read())
            commands_orig = serapi_instance.split_commands(contents)
            commands_preprocessed = [newcmd for cmd in commands_orig
                                     for newcmd in serapi_instance.preprocess_command(cmd)]
            commands = lift_and_linearize(commands_preprocessed,
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

def lifted_vernac(command):
    return re.match("Ltac\s", command)

def lift_and_linearize(commands, coqargs, includes, filename):
    with serapi_instance.SerapiContext(coqargs, includes) as coq:
        result = list(linearize_semicolons.linearize_commands(generate_lifted(commands, coq),
                                                              coq, filename))
        return result

def generate_lifted(commands, coq):
    lemma_stack = []
    try:
        for command in commands:
            if serapi_instance.possibly_starting_proof(command):
                coq.run_stmt(command)
                if coq.proof_context != None:
                    lemma_stack.append([])
                coq.cancel_last()
            if len(lemma_stack) > 0 and not lifted_vernac(command):
                lemma_stack[-1].append(command)
            else:
                yield command
            if serapi_instance.ending_proof(command):
                yield from lemma_stack.pop()
        assert(len(lemma_stack) == 0)
    except Exception as e:
        coq.kill()
        raise e

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

try:
    for worker in workers:
        worker.join()
except:
    for worker in workers:
        worker.kill()
