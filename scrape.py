#!/usr/bin/env python3

import subprocess
import threading
import re
import queue
import os
import argparse
import sys
# This dependency is in pip, the python package manager
from sexpdata import *

class AckError(Exception):
    def __init__(self, msg):
        self.msg = msg
    pass
class CompletedError(Exception):
    def __init__(self, msg):
        self.msg = msg
    pass

class CoqExn(Exception):
    def __init__(self, msg):
        self.msg = msg
    pass
class BadResponse(Exception):
    def __init__(self, msg):
        self.msg = msg
    pass

def split_semis_carefully(string):
    stack = 0
    item = []
    for char in string:
        if char == ';' and stack == 0:
            yield ''.join(item)
            item = []
            continue
        item.append(char)
        # if we are entering a block
        pos = '(['.find(char)
        if pos >= 0:
            stack += 1
            continue
        pos = ')]'.find(char)
        if pos >= 0:
            stack -= 1
    if stack != 0:
        raise ValueError("unclosed parentheses: %s" % ''.join(item))
    if item:
        yield ''.join(item)

# split_semis_brackets : String -> [[String]]
# "a ; [b | c | d]; e[.]"
# becomes
# [ a , [ b , c , d ], e ]
# Note that b, c, d may still be of the shape "foo ; [ bar | baz ]"
def split_semis_brackets(s):
    s = s.rstrip(' .')
    #print('SPLITTING: ' + str(s))
    semiUnits = list(split_semis_carefully(s.rstrip(' .')))
    #print("semiUnits :" + str(semiUnits))
    def unbracket(s):
        s = s.lstrip(' ')
        if s.startswith('['): return s.lstrip(' [').rstrip(' ]').split('|')
        else:                 return [s]
    return list(map(unbracket, semiUnits))

def count_open_proofs(coq):
    return len(coq.query_goals()[2][1])

def count_fg_goals(coq):
    if count_open_proofs(coq) == 0:
        return 0
    return len(coq.query_goals()[2][1][0][1][0][1])

# semiands   : [[String]]
# ksemiands  : [[[String]]]
# periodands : [String]
# done       : Int

# Note: each tactic has a copy of ksemiands they can mutate
# Note: all tactics share the same periodand
def linearize(coq, fout, semiands, ksemiands, periodands, done):

    nb_goals_before = count_fg_goals(coq)
    #print("CURRENT GOAL COUNT: " + str(nb_goals_before) + ", done: " + str(done))

    if nb_goals_before == done:
        #print("DONE WITH THIS PATH")
        return

    #print("LINEARIZING " + str(semiands) + ", done: " + str(done))

    # if done with the current semicolon periscope
    if len(semiands) == 0:
        if len(ksemiands) != 0:
            #print("POPPING NEXT KSEMIAND")
            nextTactic = ksemiands.pop(0)
            return linearize(coq, fout, split_semis_brackets(nextTactic), ksemiands, periodands, done)
        else:
            if len(periodands) != 0:
                nextTactic = periodands.pop(0)
                while nextTactic in ['+', '-', '*', '{', '}']:
                    if len(periodands) == 0:
                        print("ERROR: ran out of tactics w/o finishing the proof")
                    else:
                        #print("Skipping bullet: " + nextTactic)
                        nextTactic = periodands.pop(0)
                #print("POPPING NEXT PERIODAND: " + nextTactic)
                return linearize(coq, fout, split_semis_brackets(nextTactic), [], periodands, done)
            else:
                print("ERROR: ran out of tactic w/o finishing the proof")
                return

    #print(str(semiands))
    semiand = semiands.pop(0)

    # if the current semiand is a dispatch
    # [ a | b | c ] or [ a | .. b .. | c ]
    if len(semiand) > 1:
        #print("DETECTED DISPATCH, length " + str(len(semiand)))
        ksemiandsCopy = ksemiands[:] # pass a copy to each subgoal
        ksemiandsCopy.append(semiands[:])
        for i in range(len(semiand)):
            # Say the tactic was `[ a | b | c] ; d`
            # Subgoal 0 is done when `done`
            # Subgoal 1 is done when `done-1`
            # Subgoal 2 is done when `done-2`
            # Subgoal i is done when `done - i`
            new_semiands = list(split_semis_brackets(semiand[i]))
            linearize(coq, fout, new_semiands, ksemiandsCopy, periodands, done - i)
        return

    # if the current tactic is not a dispatch, run it?
    tactic = semiand[0]
    coq.run_stmt(tactic + '.')
    #print('>>> ' + tactic + '.')
    fout.write(tactic + '.\n')

    nb_goals_after = count_fg_goals(coq)
    nb_subgoals = 1 + nb_goals_after - nb_goals_before
    #print("Goal difference: " + str(nb_goals_difference))

    semiandsCopy = semiands[:] # pass a copy to each subgoal
    for i in range(nb_subgoals):
        #print("LOOP ITERATION " + str(i))
        # Say the tactic was `a; b` and `a` generates 3 subgoals
        # Subgoal 0 is done when `done+2`
        # Subgoal 1 is done when `done+1`
        # Subgoal 2 is done when `done`
        # Subgoal i is done when `done + nb_subgoals - (i + 1)`
        linearize(coq, fout, semiandsCopy[:], [], periodands, done + nb_subgoals - (i + 1))

class SerapiInstance(threading.Thread):
    def __init__(self, coq_command, includes):
        threading.Thread.__init__(self, daemon=True)
        self._proc = subprocess.Popen(coq_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._fout = self._proc.stdout
        self._fin = self._proc.stdin
        self.messages = queue.Queue()
        self.start()
        self.discard_initial_feedback()
        self.exec_includes(includes)

    def run_stmt(self, stmt):
        stmt = stmt.replace("\\", "\\\\")
        stmt = stmt.replace("\"", "\\\"")
        try:
            self._fin.write("(Control (StmAdd () \"{}\"))\n".format(stmt).encode('utf-8'))
            self._fin.flush()
            next_state = self.get_next_state()
            self._fin.write("(Control (StmObserve {}))\n".format(next_state).encode('utf-8'))
            self._fin.flush()
            feedbacks = self.get_feedbacks()
        except Exception as e:
            print("Problem running statement: {}".format(stmt))
            raise e

    def get_ack(self):
        ack = self.messages.get()
        if (ack[0] != Symbol("Answer") or
            ack[2] != Symbol("Ack")):
            raise AckError(ack)

    def get_completed(self):
        completed = self.messages.get()
        if (completed[0] != Symbol("Answer") or
            completed[2] != Symbol("Completed")):
            raise CompletedError(completed)

    def add_lib(self, origpath, logicalpath):
        addStm = ("(Control (StmAdd () \"Add Rec LoadPath \\\"{}\\\" as {}.\"))\n"
                  .format(origpath, logicalpath))
        self._fin.write(addStm.format(origpath, logicalpath).encode('utf-8'))
        self._fin.flush()
        self.get_next_state()

    def exec_includes(self, includes_string):
        for match in re.finditer("-R\s*(\S*)\s*(\S*)\s*", includes_string):
            self.add_lib(match.group(1), match.group(2))

    def get_next_state(self):
        self.get_ack()

        msg = self.messages.get()
        while msg[0] == Symbol("Feedback"):
            msg = self.messages.get()
        if (msg[0] != Symbol("Answer")):
            raise BadResponse(msg)
        submsg = msg[2]
        if (submsg[0] == Symbol("CoqExn")):
            raise CoqExn(submsg)
        elif submsg[0] != Symbol("StmAdded"):
            raise BadResponse(submsg)
        else:
            state_num = submsg[1]
            self.get_completed()
            return state_num

    def discard_initial_feedback(self):
        feedback1 = self.messages.get()
        feedback2 = self.messages.get()
        if (feedback1[0] != Symbol("Feedback") or
            feedback2[0] != Symbol("Feedback")):
            raise BadResponse("Not feedback")

    def get_feedbacks(self):
        self.get_ack()

        feedbacks = []
        next_message = self.messages.get()
        while(next_message[0] == Symbol("Feedback")):
            feedbacks.append(next_message)
            next_message = self.messages.get()
        fin = next_message
        if (fin[0] != Symbol("Answer")):
            raise BadResponse(fin)
        if (isinstance(fin[2], list)):
            raise BadResponse(fin)
        elif(fin[2] != Symbol("Completed")):
            raise BadResponse(fin)

        return feedbacks

    def query_goals(self):
        query = "(Query () Goals)\n"
        self._fin.write(query.encode('utf-8'))
        self._fin.flush()
        self.get_ack()
        answer = self.messages.get()
        return answer

    def run(self):
        while(True):
            try:
                line = self._fout.readline()
                response = loads(line.decode('utf-8'))
                # print("Got message {}".format(response))
                self.messages.put(response)
            except:
                break
    def kill(self):
        self._proc.terminate()
        self._proc.stdout.close()
        threading.Thread.join(self)

    pass

num_jobs = 0
jobs = queue.Queue()
workers = []

class Worker(threading.Thread):
    def __init__(self, output):
        threading.Thread.__init__(self)
        self.coqargs = [os.path.dirname(os.path.abspath(__file__)) + "/coq-serapi/sertop.native",
                   "--prelude={}/coq".format(os.path.dirname(os.path.abspath(__file__)))]
        includes=subprocess.Popen(['make', 'print-includes'], stdout=subprocess.PIPE).communicate()[0]
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

    def linearize(self, job):
        with open(job, 'r') as fin:
            with open(job + "-linear", 'w') as fout:
                contents = kill_comments(fin.read())
                self.coq = SerapiInstance(self.coqargs, self.includes)
                commands = [newcmd for cmd in split_commands(contents)
                                for newcmd in preprocess_command(cmd)]
                while commands:
                    command = commands.pop(0)
                    #print("POPPED: " + command)

                    if count_open_proofs(self.coq) == 0 or count_fg_goals(self.coq) == 0:
                        self.coq.run_stmt(command)
                        #print('>>> ' + command)
                        fout.write(command + '\n')
                        continue

                    #print("TIME TO LINEARIZE: " + command)
                    linearize(self.coq, fout, split_semis_brackets(command), [], commands, 0)

                self.coq.kill()


    def run(self):
        try:
            while(True):
                job = jobs.get_nowait()
                print("Processing file {} ({} of {})".format(job,
                                                             num_jobs - jobs.qsize(),
                                                             num_jobs))
                self.linearize(job)
                with open(job, 'r') as fin:
                    contents = kill_comments(fin.read())
                    self.coq = SerapiInstance(self.coqargs, self.includes)
                    commands = [newcmd for cmd in split_commands(contents)
                                for newcmd in preprocess_command(cmd)]
                    try:
                        for command in commands:
                            self.coq.run_stmt(command)
                    except Exception as e:
                        print("In file {}:".format(job))
                        raise e
                    self.coq.kill()
        except queue.Empty:
            pass
        except Exception as e:
            for worker in workers:
                if worker == self:
                    continue
                worker.kill()
            raise e

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
            if (re.match("\.\s", string[i:i+2]) and
                (not string[i-1] == "." or string[i-2] == ".")):
                result.append(next_command.strip() + ".")
                next_command = ""
                continue
        next_command += string[i]
    return result

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

def process_file(fin, fout, coqargs, includes):
    coq = SerapiInstance(coqargs, includes)
    contents = fin.read()
    contents = kill_comments(contents)
    commands = split_commands(contents)
    commands = [newcmd for cmd in commands for newcmd in preprocess_command(cmd)]
    for command in commands:
        coq.run_stmt(command)
    coq.kill()

parser = argparse.ArgumentParser(description="scrape a proof")
parser.add_argument('-o', '--output', help="output data file name", default=None)
parser.add_argument('-j', '--threads', default=1, type=int)
parser.add_argument('inputs', nargs="+", help="proof file name(s) (*.v)")
args = parser.parse_args()

num_jobs = len(args.inputs)

for infname in args.inputs:
    jobs.put(infname)

for idx in range(args.threads):
    worker = Worker(args.output)
    worker.start()
    workers.append(worker)

try:
    for worker in workers:
        worker.join()
except:
    for worker in workers:
        worker.kill()
