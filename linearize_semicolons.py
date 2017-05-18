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

import serapi_instance

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
num_jobs = 0
jobs = queue.Queue()
workers = []
output_lock = threading.Lock()

def linearize_commands(commands_original, coqargs, includes):
    #print("Starting a linearize_commands with: " + str(coqargs))
    commands = commands_original[:] # going to mutate it
    # linearize_commands needs its own Coq instance
    coq = serapi_instance.SerapiInstance(coqargs, includes)
    while commands:
        command = commands.pop(0)
        #print("POPPED: " + command)

        # Note: shortcircuiting `or` matters here
        if count_open_proofs(coq) == 0 or count_fg_goals(coq) == 0:
            coq.run_stmt(command)
            #print('>>> ' + command)
            yield(command)
            continue

        #print("TIME TO LINEARIZE: " + command)
        yield from linearize_proof(coq, split_semis_brackets(command), [], commands, 0)

    coq.kill()

# semiands   : [[String]]
# ksemiands  : [[[String]]]
# periodands : [String]
# done       : Int

# Note: each tactic has a copy of ksemiands they can mutate
# Note: all tactics share the same periodand
def linearize_proof(coq, semiands, ksemiands, periodands, done):

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
            yield from linearize_proof(coq, split_semis_brackets(nextTactic), ksemiands, periodands, done)
            return
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
                yield from linearize_proof(coq, split_semis_brackets(nextTactic), [], periodands, done)
                return
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
            yield from linearize_proof(coq, new_semiands, ksemiandsCopy, periodands, done - i)
        return

    # if the current tactic is not a dispatch, run it?
    tactic = semiand[0] + '.'
    #print('>>> ' + tactic)
    coq.run_stmt(tactic)
    yield tactic

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
        yield from linearize_proof(coq, semiandsCopy[:], [], periodands, done + nb_subgoals - (i + 1))


