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

# "a ; [b | c | d]; e"
# becomes
# [ a , [ b , c , d ], e ]
# Note that b, c, d may still be of the shape "foo ; [ bar | baz ]"
# If the input ends with `...`, it is replaced with `; with_tactic`
def split_semis_brackets(s, with_tactic):
    if s.endswith('...'):
        if with_tactic == '':
            raise "with_tactic was empty when replacing `...`"
        #print("Replacing ... with '; {}'".format(with_tactic))
        s = s.replace('...', "; {}".format(with_tactic))
    s = s.rstrip(' .')
    #print('SPLITTING: ' + str(s))
    semiUnits = list(split_semis_carefully(s.rstrip(' .')))
    #print("semiUnits :" + str(semiUnits))
    def unbracket(s):
        s = s.lstrip(' ')
        if s.startswith('['): return s.lstrip(' [').rstrip(' ]').split('|')
        else:                 return [s]
    res = list(map(unbracket, semiUnits))
    #print(str(res))
    return res

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

def linearize_commands(commands_original, coqargs, includes, filename):
    #print("Starting a linearize_commands with: " + str(coqargs))
    commands = commands_original[:] # going to mutate it
    # linearize_commands needs its own Coq instance
    coq = serapi_instance.SerapiInstance(coqargs, includes)
    fallback = False
    last_theorem_statement = ""
    num_theorems_started = 0
    with_tactic = ""
    while commands:
        command = commands.pop(0)
        #print("Popped command: {}".format(command))

        # Capture the tactic in `Proof with (...).`
        match = re.fullmatch("Proof with (.*)\.", command)
        if match and match.group(1):
            with_tactic = match.group(1).lstrip('(').rstrip(')')
        if re.fullmatch("Qed.", command):
            with_tactic = ""

        if count_open_proofs(coq) == 0:
            coq.run_stmt(command)
            #print('>>> ' + command)
            yield(command)
            fallback = False
            if count_open_proofs(coq) != 0:
                last_theorem_statement = command
                num_theorems_started += 1
            continue
        if fallback:
            coq.run_stmt(command)
            yield(command)
            continue

        #print("Entered a proof, time to linearize")
        # We reappend command so that we can save commands
        # in case linearization fails
        commands.insert(0, command)
        orig = commands[:]
        try:
            linearized_commands = list(
                linearize_proof(coq, with_tactic, commands)
            )
            if count_open_proofs(coq) != 0:
                qed = commands.pop(0)
                #print("qed: {}".format(qed))
                assert(qed == "Qed." or qed == "Defined.")
                coq.run_stmt(qed)
                postfix = [qed]
            else:
                postfix = []
            yield from linearized_commands
            yield from postfix
        except Exception as e:
            raise e # for debugging purposes
            print("Aborting current proof linearization!")
            print("Proof {}, in file {}"
                  .format(num_theorems_started, filename))
            print()
            coq.cancel_last()
            coq.run_stmt("Abort.")
            coq.run_stmt(last_theorem_statement)
            commands = orig
            fallback = True

    coq.kill()

# semiands   : [[String]]
# ksemiands  : [[[String]]]
# periodands : [String]
# done       : Int

# Note: each tactic has a copy of ksemiands they can mutate
# Note: all tactics share the same periodand
def linearize_proof(coq, with_tactic, commands):

    #print("linearize_proof(coq, '{}', {})".format(
    #    with_tactic, str(commands)
    #))

    def lin(semiands, ksemiands, periodands, done):

        #print("lin({}, {}, {}, {})".format(
        #str(semiands), str(ksemiands), str(periodands), str(done)
        #))

        nb_goals_before = count_fg_goals(coq)
        #print("Current goal count: {}, done when: {}"
        #.format(str(nb_goals_before), str(done)))

        if nb_goals_before == done:
            #print("Done with this path")
            return

        #print("Linearizing {}, done when: {}".format(
        #str(semiands), str(done)
        #))

        # if done with the current semicolon periscope
        if len(semiands) == 0:
            if len(ksemiands) != 0:
                #print("Turning ksemiands {} into semiands".format(str(ksemiands)))
                # The ksemiands are already in semiands shape
                yield from lin(ksemiands, [], periodands, done)
                return
            else:
                if len(periodands) != 0:
                    next_tactic = periodands.pop(0)
                    while next_tactic in ['+', '-', '*', '{', '}']:
                        if len(periodands) == 0:
                            print("ERROR: ran out of tactics w/o finishing the proof")
                        else:
                            #print("Skipping bullet: {}".format(next_tactic))
                            next_tactic = periodands.pop(0)
                    #print("Popping periodand: {}".format(next_tactic))
                    next_semiands = list(split_semis_brackets(next_tactic, with_tactic))
                    #print("With tactic: ".format(with_tactic))
                    #print("Next semiands: {}".format(str(next_semiands)))
                    yield from lin(next_semiands, [], periodands, done)
                    return
                else:
                    print("ERROR: ran out of tactic w/o finishing the proof")
                    return

        #print(str(semiands))
        semiand = semiands.pop(0)

        # if the current semiand is a dispatch
        # [ a | b | c ] or [ a | .. b .. | c ]
        if len(semiand) > 1:
            #print("Detected dispatch, length {}".format(str(len(semiand))))
            # This deserves an explanation, think about:
            # a ; [ _ | b ; [ c | d ] ; e | _ ]; f
            # When reaching [ c | d ], we have:
            # ksemiands = [f]
            # semiands  = [e]
            # So the ksemiands for c should be [e , f]
            branch_ksemiands = semiands[:] + ksemiands[:]
            for i in range(len(semiand)):
                # Say the tactic was `[ a | b | c] ; d`
                # Subgoal 0 is done when `done`
                # Subgoal 1 is done when `done-1`
                # Subgoal 2 is done when `done-2`
                # Subgoal i is done when `done - i`
                #print("Dispatching branch {}/{} with semiand {} and ksemiand {}".format(
                #str(1+i), str(len(semiand)), str(semiand[i]), str(branch_ksemiands)))
                new_semiands = list(split_semis_brackets(semiand[i], with_tactic))
                yield from lin(new_semiands, branch_ksemiands[:], periodands, done - i)
            return

        # if the current tactic is not a dispatch, run it
        tactic = semiand[0] + '.'
        coq.run_stmt(tactic)
        #print(tactic)
        yield tactic

        nb_goals_after = count_fg_goals(coq)
        nb_subgoals = 1 + nb_goals_after - nb_goals_before
        #print("{} subgoal(s)".format(str(nb_subgoals)))

        semiandsCopy = semiands[:] # pass a copy to each subgoal
        for i in range(nb_subgoals):
            #print("Loop iteration {}".format(str(i)))
            # Say the tactic was `a; b` and `a` generates 3 subgoals
            # Subgoal 0 is done when `done+2`
            # Subgoal 1 is done when `done+1`
            # Subgoal 2 is done when `done`
            # Subgoal i is done when `done + nb_subgoals - (i + 1)`
            yield from lin(semiandsCopy[:], ksemiands, periodands, done + nb_subgoals - (i + 1))

        return

    first_tactic = commands.pop(0)
    semiands = list(split_semis_brackets(first_tactic, with_tactic))
    yield from lin(semiands, [], commands, 0)
