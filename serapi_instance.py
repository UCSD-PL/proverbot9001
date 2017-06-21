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

class SerapiInstance(threading.Thread):
    def __init__(self, coq_command, includes, prelude):
        threading.Thread.__init__(self, daemon=True)
        self._proc = subprocess.Popen(coq_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._fout = self._proc.stdout
        self._fin = self._proc.stdin
        self.messages = queue.Queue()
        self.start()
        self.discard_initial_feedback()
        self.exec_includes(includes, prelude)
        self.proof_context = None
        self.cur_state = 0
        self.prev_tactics = []
        self.debug = False

    def run_stmt(self, stmt):
        if self.debug:
            print("Running statement: " + stmt)
        stmt = stmt.replace("\\", "\\\\")
        stmt = stmt.replace("\"", "\\\"")
        try:
            for stm in preprocess_command(kill_comments(stmt)):
                self._fin.write("(Control (StmAdd () \"{}\"))\n".format(stm).encode('utf-8'))
                self._fin.flush()
                self.cur_state = self.get_next_state()

                self._fin.write("(Control (StmObserve {}))\n".format(self.cur_state).encode('utf-8'))
                self._fin.flush()
                feedbacks = self.get_feedbacks()

                self.get_proof_context()

                if self.proof_context:
                    self.prev_tactics.append(stm)
                else:
                    self.prev_tactics = []

        except Exception as e:
            print("Problem running statement: {}".format(stmt))
            raise e

    def cancel_last(self):
        while not self.messages.empty():
            self.messages.get()
        cancel = "(Control (StmCancel ({})))".format(self.cur_state)
        self._fin.write(cancel.encode('utf-8'))
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

    def exec_includes(self, includes_string, prelude):
        for match in re.finditer("-R\s*(\S*)\s*(\S*)\s*", includes_string):
            self.add_lib(prelude + "/" + match.group(1), match.group(2))

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
        self._fin.write("(Query ((sid {}) (pp ((pp_format PpStr)))) Goals)".format(self.cur_state).encode('utf-8'))
        self._fin.flush()
        self.get_ack()

        proof_context_message = self.messages.get()
        if proof_context_message[0] != Symbol("Answer"):
            raise BadResponse(proof_context_message)
        else:
            ol_msg = proof_context_message[2]
            if (ol_msg[0] != Symbol("ObjList")):
                raise BadResponse(proof_context_message)
            if len(ol_msg[1]) != 0:
                newcontext = self.extract_proof_context(ol_msg[1])
                self.proof_context = newcontext.split("\n\n")[0]
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

class SerapiContext:
    def __init__(self, coq_commands, includes, prelude):
        self.coq_commands = coq_commands
        self.includes = includes
        self.prelude = prelude
    def __enter__(self):
        self.coq = SerapiInstance(self.coq_commands, self.includes, self.prelude)
        return self.coq
    def __exit__(self, type, value, traceback):
        self.coq.kill()

def possibly_starting_proof(command):
    stripped_command = kill_comments(command).strip()
    return (re.match("Lemma\s", stripped_command) or
            re.match("Theorem\s", stripped_command) or
            re.match("Remark\s", stripped_command) or
            re.match("Proposition\s", stripped_command) or
            re.match("Definition\s", stripped_command) or
            re.match("Example\s", stripped_command) or
            re.match("Fixpoint\s", stripped_command) or
            re.match("Corollary\s", stripped_command) or
            re.match("Let\s", stripped_command) or
            ("Instance" in stripped_command and
             "Declare" not in stripped_command) or
            re.match("Function\s", stripped_command) or
            re.match("Next Obligation", stripped_command) or
            re.match("Property\s", stripped_command) or
            re.match("Add Morphism\s", stripped_command))

def ending_proof(command):
    stripped_command = kill_comments(command).strip()
    return ("Qed" in stripped_command or
            "Defined" in stripped_command or
            (re.match("\s*Proof\s+\S+\s*", stripped_command) and
             not re.match("\s*Proof\s+with", stripped_command)))

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

def count_open_proofs(goals):
    return len(goals[2][1])

def count_fg_goals(goals):
    if count_open_proofs(goals) == 0:
        return 0
    return len(goals[2][1][0][1][0][1])
