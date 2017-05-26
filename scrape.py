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
        self.coq= None
        pass

    def kill(self):
        if (self.coq != None):
            self.coq.kill()
        threading.Thread.join(self)

    def process_statement(self, command):
        if re.match(";", command) and options["no-semis"]:
            return
        self.outbuf = ""
        if self.coq.proof_context:
            prev_tactics = self.coq.prev_tactics
            prev_hyps = self.coq.get_hypothesis()
            prev_goal = self.coq.get_goals()
            self.outbuf += "*****\n"
            self.outbuf += prev_goal + "\n"
            self.outbuf += "+++++\n"
            self.outbuf += command + "\n"
        else:
            prev_goal = None
        self.coq.run_stmt(command)


        if self.coq.proof_context and prev_goal:
            post_hyps = self.coq.get_hypothesis()
            post_goal = self.coq.get_goals()
            self.outbuf += "-----\n"
            self.outbuf += post_goal + "\n"
        with open(self.tmpfile_name, 'a') as tmp_file:
            tmp_file.write(self.outbuf)

    def process_file(self, filename):
        try:
            with open(filename, 'r') as fin:
                contents = kill_comments(fin.read())
            commands_orig = split_commands(contents)
            commands_preprocessed = [newcmd for cmd in commands_orig for newcmd in preprocess_command(cmd)]
            commands = lift_and_linearize(commands_preprocessed,
                                          self.coqargs, self.includes,
                                          filename)
            self.coq = serapi_instance.SerapiInstance(self.coqargs, self.includes)
            for command in commands:
                self.process_statement(command)
            self.coq.kill()
        except Exception as e:
            print("In file {}:".format(filename))
            if (self.coq != None):
                self.coq.kill()
                self.coq = None
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

def kill_comments(string):
    result = ""
    depth = 0
    in_quote = False
    for i in range(len(string)):
        if in_quote:
            if depth == 0:
                result += string[i]
            if string[i] == '"' and string[i-1] != '\\':
                in_quote = False
        else:
            if string[i:i+2] == '(*':
                depth += 1
            if depth == 0:
                result += string[i]
            if string[i-1:i+1] == '*)':
                depth -= 1
            if string[i] == '"' and string[i-1] != '\\':
                in_quote = True
    return result

def split_commands(string):
    result = []
    next_command = ""
    in_quote = False
    for i in range(len(string)):
        if in_quote:
            if string[i] == '"' and string[i-1] != '\\':
                in_quote = False
        else:
            if string[i] == '"' and string[i-1] != '\\':
                in_quote = True
            if (re.match("[\{\}]", string[i]) and
                re.fullmatch("\s*", next_command)):
                result.append(string[i])
                next_command = ""
                continue
            if (re.match("[\+\-\*]", string[i]) and
                string[i] != string[i+1] and
                re.fullmatch("\s*[\+\-\*]*", next_command)):
                next_command += string[i]
                result.append(next_command.strip())
                next_command = ""
                continue
            if (re.match("\.($|\s)", string[i:i+2]) and
                (not string[i-1] == "." or string[i-2] == ".")):
                result.append(next_command.strip() + ".")
                next_command = ""
                continue
        next_command += string[i]
    return result

def lifted_vernac(command):
    return re.match("Ltac\s", command)

def lift_and_linearize(commands, coqargs, includes, filename):
    coq = serapi_instance.SerapiInstance(coqargs, includes)
    try:
        result = list(linearize_semicolons.linearize_commands(generate_lifted(commands, coq),
                                                              coq, filename))
        coq.kill()
        return result
    except Exception as e:
        coq.kill()
        raise e

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

def preprocess_command(cmd):
    needPrefix = ["String", "Classical", "ClassicalFacts",
                  "ClassicalDescription", "ClassicalEpsilon",
                  "Equivalence", "Init.Wf", "Program.Basics",
                  "Max", "Wf_nat", "EquivDec", "Znumtheory",
                  "Bool", "Zquot", "FSets", "FSetAVL",
                  "Wellfounded", "FSetInterface",
                  "OrderedType", "Program", "Recdef", "Eqdep_dec",
                  "FunctionalExtensionality", "Zwf", "Permutation",
                  "Orders", "Mergesort", "List", "ZArith", "Int31",
                  "Syntax", "Streams", "Equality",
                  "ProofIrrelevance", "Setoid", "EqNat",
                  "Arith", "Cyclic31", "Omega", "Relations",
                  "RelationClasses", "OrderedTypeAlt", "FMapAVL",
                  "BinPos", "BinNat", "DecidableClass", "Reals",
                  "Psatz", "ExtrOcamlBasic", "ExtrOcamlString",
                  "Ascii"]
    for lib in needPrefix:
        match = re.fullmatch("\s*Require(\s+(?:(?:Import)|(?:Export)))?((?:\s+\S+)*)\s+({})\s*((?:\s+\S*)*)\.".format(lib), cmd)
        if match:
            if match.group(1):
                impG= match.group(1)
            else:
                impG=""
            if match.group(4):
                after = match.group(4)
            else:
                after=""
            if (re.fullmatch("\s*", match.group(2)) and re.fullmatch("\s*", after)):
                return ["From Coq Require" + impG + " " + match.group(3) + "."]
            else:
                return ["From Coq Require" + impG + " " + match.group(3) + "."] + preprocess_command("Require " + impG.strip() + " " + match.group(2).strip() + " " + after + ".")
    return [cmd]

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
