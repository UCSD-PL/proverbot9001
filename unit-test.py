#!/usr/bin/env python3

import argparse
import subprocess
import os
import re
import math

from shutil import *
from format import format_command_record

import serapi_instance
import linearize_semicolons

from serapi_instance import count_fg_goals
from helper import lift_and_linearize

darknet_command = ['./try-auto.py']
parser = argparse.ArgumentParser(description=
                                 "try to match the file by predicting a tacti")
parser.add_argument('--prelude', default=".")
parser.add_argument('filename', nargs=1, help="proof file name (*.v)")
args = parser.parse_args()
filename = args.filename[0]

with open(filename, 'r') as fin:
    contents = serapi_instance.kill_comments(fin.read())
commands_orig = serapi_instance.split_commands(contents)
commands_preprocessed = [newcmd for cmd in commands_orig
                         for newcmd in serapi_instance.preprocess_command(cmd)]
base = os.path.dirname(os.path.abspath(__file__))
coqargs = ["{}/coq-serapi/sertop.native".format(base),
           "--prelude={}/coq".format(base)]
includes = subprocess.Popen(['make', '-C', args.prelude, 'print-includes'],
                            stdout=subprocess.PIPE).communicate()[0].decode('utf-8')
commands = lift_and_linearize(commands_preprocessed,
                              coqargs, includes, filename)

num_correct = 0
num_tactics = 0
with serapi_instance.SerapiContext(coqargs, includes) as coq:
    query = ""
    for command in commands:
        in_proof = count_fg_goals(coq) != 0
        if in_proof:
            prev_goal = coq.get_goals()
            prev_hyps = coq.get_hypothesis()
            prev_tactics = coq.prev_tactics
            # Let's predict a tactic!
            num_tactics += 1
            response, errors = subprocess.Popen(darknet_command,
                                                stdin=subprocess.PIPE,
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE
            ).communicate(input=query.encode('utf-8'))
            result = response.decode('utf-8').strip()
            if command == result:
                num_correct += 1

            coq.run_stmt(command)
            still_in_proof = count_fg_goals(coq) != 0
            if still_in_proof:
                post_goal = coq.get_goals()
                post_hyps = coq.get_hypothesis()
            query += format_command_record(prev_tactics, prev_hyps, prev_goal,
                                           command, post_hyps, post_goal)
        else:
            coq.run_stmt(command)

print("Accuracy: %{} ({}/{})".format(math.floor(num_correct / num_tactics * 100),
                                     num_correct, num_tactics))
