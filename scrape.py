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
        self._fin.write("(Control (StmAdd () \"{}\"))\n".format(stmt).encode('utf-8'))
        self._fin.flush()
        next_state = self.get_next_state()
        self._fin.write("(Control (StmObserve {}))\n".format(next_state).encode('utf-8'))
        self._fin.flush()
        feedbacks = self.get_feedbacks()
        print(feedbacks)

    def add_lib(self, libname, libpath):
        self._fin.write("(LibAdd ({}) \"{}\" false)\n"
                        .format(libname, libpath.replace(".","/")).encode('utf-8'))
    def exec_includes(self, includes_string):
        for match in re.finditer("-R *(\S*) *(\S*) *", str(includes_string)):
            self.add_lib(match.group(1), match.group(2))

    def get_next_state(self):
        ack = self.messages.get()
        assert ack[0] == Symbol("Answer")
        assert ack[2] == Symbol("Ack")

        msg = self.messages.get()
        assert msg[0] == Symbol("Answer")
        submsg = msg[2]
        assert submsg[0] == Symbol("StmAdded")
        state_num = submsg[1]
        completed = self.messages.get()
        assert completed[0] == Symbol("Answer")
        assert completed[2] == Symbol("Completed")
        return state_num

    def discard_initial_feedback(self):
        feedback1 = self.messages.get()
        assert feedback1[0] == Symbol("Feedback")
        feedback2 = self.messages.get()
        assert feedback2[0] == Symbol("Feedback")

    def get_feedbacks(self):
        ack = self.messages.get()
        assert ack[0] == Symbol("Answer")
        assert ack[2] == Symbol("Ack")

        feedbacks = []
        next_message = self.messages.get()
        while(next_message[0] == Symbol("Feedback")):
            feedbacks.append(next_answer)
            next_answer = self.answers.get()
        fin = next_message
        assert fin[0] == Symbol("Answer")
        assert fin[2] == Symbol("Completed")

        return response_list

    def run(self):
        while(True):
            try:
                line = self._fout.readline()
                response = loads(line.decode('utf-8'))
                self.messages.put(response)
            except:
                break
    def kill(self):
        self._proc.terminate()
        self._proc.stdout.close()
        threading.Thread.join(self)

    pass

def kill_comments(string):
    result = ""
    depth = 0
    for i in range(len(string)):
        if string[i:i+2] == '(*':
            depth += 1
        if depth == 0:
            result += string[i]
        if string[i-1:i+1] == '*)':
            depth -= 1
    return result

def process_file(fin, fout, coqargs, includes):
    coq = SerapiInstance(coqargs, includes)
    contents = fin.read()
    contents = kill_comments(contents)
    coq.kill()

coqargs = [os.path.dirname(os.path.abspath(__file__)) + "/coq-serapi/sertop.native",
           "--prelude={}/coq".format(os.path.dirname(os.path.abspath(__file__)))]
includes=subprocess.Popen(['make', 'print-includes'], stdout=subprocess.PIPE).communicate()[0]

parser = argparse.ArgumentParser(description="scrape a proof")
parser.add_argument('-o', '--output', help="output data file name", default=None)
parser.add_argument('inputs', nargs="+", help="proof file name(s) (*.v)")
args = parser.parse_args()

if args.output == None:
    fout = sys.stdout
else:
    fout = open(args.output, 'w')

for idx, infname in enumerate(args.inputs):
    print("Processing file {} ({} of {})".format(infname, idx + 1, len(args.inputs)))
    with open(infname) as fin:
        process_file(fin, fout, coqargs, includes)
