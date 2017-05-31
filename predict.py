#!/usr/bin/env python3

import argparse
import subprocess
import os
import re

from shutil import *
from format import format_context

import serapi_instance
from serapi_instance import count_fg_goals
import linearize_semicolons

def lifted_vernac(command):
    return re.match("Ltac\s", command)

def generate_lifted_nofail(commands, coq):
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
        if len(lemma_stack) > 0:
            rest = lemma_stack.pop()
            print(rest)
            yield from rest
    except Exception as e:
        coq.kill()
        raise e

darknet_command = ['./try-auto.py']

parser = argparse.ArgumentParser(description="predict a tactic")
parser.add_argument('-o', '--output', help="output data file name", default=None)
parser.add_argument('-i', '--inplace', help="overwrite the input file with the new tactic",
                    default=False, const=True, action='store_const')
parser.add_argument('--prelude', default=".")
parser.add_argument('filename', nargs=1, help="partially completed proof file name (*.v)")
args = parser.parse_args()

with open(args.filename[0], 'r') as fin:
    contents = serapi_instance.kill_comments(fin.read())
commands_orig = serapi_instance.split_commands(contents)
commands_preprocessed = [newcmd for cmd in commands_orig
                         for newcmd in serapi_instance.preprocess_command(cmd)]
base = os.path.dirname(os.path.abspath(__file__))
coqargs = ["{}/coq-serapi/sertop.native".format(base),
           "--prelude={}/coq".format(base)]
includes = subprocess.Popen(['make', '-C', args.prelude, 'print-includes'],
                            stdout=subprocess.PIPE).communicate()[0].decode('utf-8')
with serapi_instance.SerapiContext(coqargs, includes) as coq:
    commands = list(linearize_semicolons
                    .linearize_commands(
                        generate_lifted_nofail(commands_preprocessed, coq),
                        coq, args.filename))
with serapi_instance.SerapiContext(coqargs, includes) as coq:
    for command in commands:
        print("command: {}".format(command))
        coq.run_stmt(command)

    if count_fg_goals(coq) != 0:
        query = format_context(coq.prev_tactics, coq.get_hypothesis(), coq.get_goals())

        response, errors = subprocess.Popen(darknet_command, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).communicate(input=query.encode('utf-8'))

        result = response.decode('utf-8').strip() + "\n"
    else:
        print("Not in proof!\n")

    if args.inplace:
        args.output = args.filename[0]
    if args.output == None:
        args.output = args.filename[0] + ".pb"
    copy(args.filename[0], args.output)
    with open("args.output", "a")as fout:
        print("Response is: {}".format(result))
        fout.write(result)
