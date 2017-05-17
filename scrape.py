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
        self.proof_context = None
        self.cur_state = 0
        self.prev_tactics = []

    def run_stmt(self, stmt):
        stmt = stmt.replace("\\", "\\\\")
        stmt = stmt.replace("\"", "\\\"")
        try:
            self._fin.write("(Control (StmAdd () \"{}\"))\n".format(stmt).encode('utf-8'))
            self._fin.flush()
            self.cur_state = self.get_next_state()

            self._fin.write("(Control (StmObserve {}))\n".format(self.cur_state).encode('utf-8'))
            self._fin.flush()
            feedbacks = self.get_feedbacks()

            self._fin.write("(Query ((sid {}) (pp ((pp_format PpStr)))) Goals)".format(self.cur_state).encode('utf-8'))
            self._fin.flush()
            self.get_proof_context()

            if self.proof_context:
                self.prev_tactics.append(stmt)
            else:
                self.prev_tactics = []

        except Exception as e:
            print("Problem running statement: {}".format(stmt))
            raise e

    def cancel_last(self):
        self._fin.write("(Control (StmCancel ({})))".format(self.cur_state).encode('utf-8'))
        self._fin.flush()
        self.get_cancelled()
        self.cur_state = self.cur_state - 1

    def get_ack(self):
        ack = self.messages.get()
        if (not isinstance(ack, list) or
            ack[0] != Symbol("Answer") or
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
    def get_cancelled(self):
        self.get_ack()
        feedback = self.messages.get()
        cancelled = self.messages.get()
        self.get_completed()

    def extract_proof_context(self, raw_proof_context):
        return raw_proof_context[0][1]

    def get_goals(self):
        split = re.split("\n======+\n", self.proof_context)
        return split[1]

    def get_hypothesis(self):
        return re.split("\n======+\n", self.proof_context)[0]

    def get_proof_context(self):
        self.get_ack()

        proof_context_message = self.messages.get()
        if proof_context_message[0] != Symbol("Answer"):
            raise BadResponse(proof_context_message)
        else:
            ol_msg = proof_context_message[2]
            if (ol_msg[0] != Symbol("ObjList")):
                raise BadResponse(proof_context_message)
            if len(ol_msg[1]) != 0:
                self.proof_context = self.extract_proof_context(ol_msg[1])
            else:
                self.proof_context = None

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
output_lock = threading.Lock()

class Worker(threading.Thread):
    def __init__(self, output, prelude="."):
        threading.Thread.__init__(self, daemon=True)
        self.coqargs = [os.path.dirname(os.path.abspath(__file__)) + "/coq-serapi/sertop.native",
                   "--prelude={}/coq".format(os.path.dirname(os.path.abspath(__file__)))]
        includes=subprocess.Popen(['make', '-C', prelude, 'print-includes'], stdout=subprocess.PIPE).communicate()[0]
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

    def process_statement(self, command):
        if self.coq.proof_context:
            prev_tactics = self.coq.prev_tactics
            prev_hyps = self.coq.get_hypothesis()
            prev_goal = self.coq.get_goals()
            self.outbuf += "*****\n"
            self.outbuf += prev_goal + "\n"
            self.outbuf += "+++++\n"
            self.outbuf += command + "\n"
        self.coq.run_stmt(command)


        if self.coq.proof_context:
            post_hyps = self.coq.get_hypothesis()
            post_goal = self.coq.get_goals()
            self.outbuf += "-----\n"
            self.outbuf += post_goal + "\n"

    def process_file(self, filename):
        self.linearize(filename)
        with open(filename, 'r') as fin:
            contents = kill_comments(fin.read())
        try:
            commands = lift_inner_lemmas([newcmd for cmd
                                          in split_commands(contents)
                                          for newcmd
                                          in preprocess_command(cmd)],
                                         self.coqargs, self.includes)
            self.coq = SerapiInstance(self.coqargs, self.includes)
            for command in commands:
                self.process_statement(command)
        except Exception as e:
            print("In file {}:".format(filename))
            if (self.coq != None):
                self.coq.kill()
                self.coq = None
            raise e

        output_lock.acquire()
        self.fout.write(self.outbuf)
        output_lock.release()
        self.outbuf = ""

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
            if (re.match("\.\s", string[i:i+2]) and
                (not string[i-1] == "." or string[i-2] == ".")):
                result.append(next_command.strip() + ".")
                next_command = ""
                continue
        next_command += string[i]
    return result

def has_toplevel_colonequals(command):
    depth = 0
    for i in range(len(command)):
        if re.match("\s\{\|\s", command[i:i+4]):
            depth += 1
        if re.match("\s\|\}\s", command[i:i+4]):
            depth -= 1
        if re.match("\slet\s", command[i:i+5]):
            depth += 1
        if re.match("\sin\s", command[i:i+4]):
            depth -= 1
        if re.match(":=\s", command[i:i+4]) and depth == 0:
            return True
    return False

def possibly_starting_proof(command):
    return (re.match("Lemma\s", command) or
            re.match("Theorem\s", command) or
            re.match("Remark\s", command) or
            re.match("Proposition\s", command) or
            re.match("Definition\s", command) or
            re.match("Example\s", command) or
            re.match("Fixpoint\s", command) or
            re.match("Corollary\s", command) or
            re.match("Let\s", command) or
            ("Instance" in command and
             "Declare" not in command) or
            re.match("Function\s", command) or
            re.match("Next Obligation", command) or
            re.match("Property\s", command) or
            re.match("Add Morphism\s", command))

def ending_proof(command):
    return ("Qed" in command or
            "Defined" in command or
            (re.match("Proof\s+\S+\s*", command) and
             not re.match("Proof with", command)))

def lift_inner_lemmas(commands, args, includes):
    coq = SerapiInstance(args, includes)
    new_contents = []
    lemma_stack = []
    try:
        for command in commands:
            if possibly_starting_proof(command):
                coq.run_stmt(command)
                if coq.proof_context != None:
                    # print("Starting proof with \"{}\"".format(command))
                    lemma_stack.append([])
                coq.cancel_last()
            if len(lemma_stack) > 0:
                lemma_stack[-1].append(command)
            else:
                new_contents.append(command)
                coq.run_stmt(command)
                if ending_proof(command):
                    # print("Ending proof with \"{}\"".format(command))
                    lemma_contents = lemma_stack.pop()

                    new_contents.extend(lemma_contents)
                    for command in lemma_contents:
                        coq.run_stmt(command)
        assert (len(lemma_stack) == 0)
        coq.kill()
    except Exception as e:
        coq.kill()
        raise e
    return new_contents

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
parser.add_argument('inputs', nargs="+", help="proof file name(s) (*.v)")
args = parser.parse_args()

num_jobs = len(args.inputs)

for infname in args.inputs:
    jobs.put(infname)

for idx in range(args.threads):
    worker = Worker(args.output, args.prelude)
    worker.start()
    workers.append(worker)

try:
    for worker in workers:
        worker.join()
except:
    for worker in workers:
        worker.kill()
