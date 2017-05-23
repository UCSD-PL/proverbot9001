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

debug = False
show_debug = False
show_trace = False

def scope_aware_split(string, separator, opens, closes):
    stack = 0
    item = []
    for char in string:
        if char == separator and stack == 0:
            yield ''.join(item)
            item = []
            continue
        item.append(char)
        # if we are entering a block
        pos = opens.find(char)
        if pos >= 0:
            stack += 1
            continue
        pos = closes.find(char)
        if pos >= 0:
            stack -= 1
    if stack != 0:
        raise ValueError("unclosed parentheses: %s" % ''.join(item))
    if item:
        yield ''.join(item)

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

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
    semiUnits = list(scope_aware_split(s, ';', '{[(', '}])'))
    #print("semiUnits :" + str(semiUnits))
    def unbracket(s):
        s = s.strip(' ')
        if s.startswith('['):
            s = s.replace('[', '', 1)
            s = rreplace(s, ']', '', 1) # careful with `[ ... | intros [ | ] ]`
            s = s.strip(' ')
            return scope_aware_split(s, '|', '{[(', '}])')
        else:
            return [s]
    res = list(map(unbracket, semiUnits))
    res = list(map(lambda l: list(map(lambda s: s.strip(' '), l)), res))
    #print("SPLIT: {}".format(str(res)))
    return res

split1_obtained = split_semis_brackets("a; [ b | c ]...", "d; [ e | f ]")
split1_expected = [ ['a'] , ['b', 'c'] , ['d'] , ['e' , 'f'] ]

split2_obtained = split_semis_brackets("a ; [ b | c ; [ d ; e | f ] ; g | h ; i ]; j", "")
split2_expected = [ ['a'] , ['b', 'c ; [ d ; e | f ] ; g', 'h ; i'] , ['j'] ]

split_tests = [ (split1_expected, split1_obtained)
              , (split2_expected, split2_obtained)
              ]

for (e, o) in split_tests:
    if (e != o):
        print("Error in split_semi_brackets")
        print("Expected: {}".format(str(e)))
        print("Obtained: {}".format(str(o)))
        raise "FIXME"

def count_open_proofs(coq):
    return len(coq.query_goals()[2][1])

def count_fg_goals(coq):
    if count_open_proofs(coq) == 0:
        return 0
    return len(coq.query_goals()[2][1][0][1][0][1])

# a semiand is just some pipeands
def show_semiand(pipeands):
    if len(pipeands) == 1:
        return pipeands[0]
    else:
        #print(str(pipeands))
        return '[ {} ]'.format(' | '.join(pipeands))

def show_semiands(semiands):
    return ' ; '.join(map(show_semiand, semiands))

def linearize_commands(commands_sequence, coq, filename):
    num_theorems_started = 0
    command = next(commands_sequence, None)
    while command:
        # Run up to the next proof
        while count_open_proofs(coq) == 0:
            coq.run_stmt(command)
            if count_open_proofs(coq) == 0:
                yield command
                command = next(commands_sequence, None)
                if not command:
                    return
        # Cancel the proof starting command so that we're right before the proof
        coq.cancel_last()
        # Pull the entire proof from the lifter into command_batch
        command_batch = []
        while command and not serapi_instance.ending_proof(command):
            command_batch.append(command)
            command = next(commands_sequence, None)
        # Get the QED on there too.
        command_batch.append(command)

        # Now command_batch contains everything through the next
        # Qed/Defined.
        theorem_statement = command_batch.pop(0)
        coq.run_stmt(theorem_statement)
        fallback = False
        num_theorems_started += 1
        yield theorem_statement

        # This might not be super robust?
        match = re.fullmatch("Proof with (.*)\.", command_batch[0])
        if match and match.group(1):
            with_tactic = match.group(1).lstrip('(').rstrip(')')
        else:
            with_tactic = ""

        orig = command_batch[:]
        try:
            yield from list(linearize_proof(coq, with_tactic, command_batch))

            # If there are unconsumed items in the batch, they must
            # just be a single ending statement, so run them and yield
            # them.
            for command in command_batch:
                coq.run_stmt(command)
                yield command
        except Exception as e:
            if debug:
                raise e
            print("Aborting current proof linearization!")
            print("Proof {}, in file {}".format(num_theorems_started, filename))
            print()
            coq.cancel_last()
            coq.run_stmt("Abort.")
            coq.run_stmt(theorem_statement)
            for command in orig:
                coq.run_stmt(command)
                yield command

        command = next(commands_sequence, None)

# semiands   : [[String]]
# ksemiands  : [[[String]]]
# periodands : [String]
# done       : Int

def branch_done(done, nb_subgoals, branch_index):
    return done + nb_subgoals - 1 - branch_index

def linearize_proof(coq, with_tactic, commands):

    #print("linearize_proof(coq, '{}', {})".format(
    #    with_tactic, str(commands)
    #))

    def linearize_periodands(periodands, done):
        if show_debug:
            print("Linearizing next periodands, done when: {}".format(str(done)))
        if len(periodands) == 0:
            raise "Error: ran out of tactic w/o finishing the proof"
        next_tactic = periodands.pop(0)
        while next_tactic in ['+', '-', '*', '{', '}']:
            if len(periodands) == 0:
                raise "Error: ran out of tactics w/o finishing the proof"
            else:
                if show_debug:
                    print("Skipping bullet: {}".format(next_tactic))
                next_tactic = periodands.pop(0)
        if show_debug:
            print("Popping periodand: {}".format(next_tactic))
        next_semiands = list(split_semis_brackets(next_tactic, with_tactic))
        yield from lin(next_semiands, periodands, done)
        if show_debug:
            print("Done working on periodand")
        #yield from linearize_periodands(periodands, done - 1)
        return

    # IMPORTANT:
    # - all semiands should work on their own copy of semiands (not shared)
    # - all semiands should work on the same copy of periodands (shared)
    def lin(semiands, periodands, done):

        if show_debug:
            print("Linearizing {}".format(show_semiands(semiands)))
            print("Done when {} subgoals left".format(str(done)))

        nb_goals_before = count_fg_goals(coq)
        if show_debug:
            print("Goals before: {}".format(str(nb_goals_before)))

        if len(semiands) == 0:
            raise "Error: Called lin with empty semiands"
        semiand = semiands.pop(0)
        # dispatch is now preprocessed when we have the information on subgoals
        # available, so popped semiands ought to be just one command
        if len(semiand) != 1:
            raise "Error: popped a semiand that was not preprocessed"
        tactic = semiand[0] + '.'
        coq.run_stmt(tactic)
        if show_trace:
            print('    ' + tactic)
        yield tactic

        nb_goals_after = count_fg_goals(coq)
        if show_debug:
            print("Goals after: {}".format(str(nb_goals_after)))

        if nb_goals_after == done:
            if show_debug:
                print("Done with this path")
            return

        nb_subgoals = 1 + nb_goals_after - nb_goals_before
        if show_debug:
            print("{} subgoal(s) generated".format(str(nb_subgoals)))

        # here there are three cases:
        # 1. if there is no semiand next, then we want to pop the next periodand
        # 2. if the next semiand is a single tactic, we want to run it on all
        #    subgoals
        # 3. if the next semiand is a dispatch, we want to match each dispatched
        #    tactic to its subgoal

        if len(semiands) == 0: # 1.
            if show_debug:
                print("#1")
            for i in range(nb_subgoals):
                yield from linearize_periodands(periodands, branch_done(done, nb_subgoals, i))
            return

        peek_semiand = semiands[0]

        if len(peek_semiand) == 0:
            raise "Peeked an empty semiand, this should not happen"
        if len(peek_semiand) == 1: # 2.
            # each subgoal must have its own copy of semiands
            for i in range(nb_subgoals):
                if show_debug:
                    print("Linearizing subgoal {}/{} with semiands {}".format(
                    str(1+i), str(nb_subgoals), show_semiands(semiands)))
                yield from lin(semiands[:], periodands, branch_done(done, nb_subgoals, i))
            if show_debug:
                print("Done solving {} subgoal(s)".format(str(nb_subgoals)))
            # there might be more goals to be solved
            if show_debug:
                print("#2")
            #yield from linearize_periodands(periodands, done - 1)
            return
        else: # 3.
            next_semiand = semiands.pop(0) # same as peek_semiand, but need the side-effect
            if show_debug:
                print("Detected dispatch, length {}".format(str(len(peek_semiand))))
            # peek_semiand is a dispatch, 3 cases:
            # 1. [ a | b | c ] ; ...
            # 2. [ a | b | .. ] ; ...
            # 3. [ a | .. b .. | c ] ; ...
            if len(next_semiand) == 0:
                raise "Error: empty next semiand"
            if next_semiand[-1] == '..': # 2.
                if show_debug:
                    print('Found .. in: {}'.format(show_semiand(next_semiand)))
                next_semiand = next_semiand[:-1]
                delta = nb_subgoals - len(next_semiand)
                # I don't want to do 'idtac' here but as a first approximation
                next_semiand = next_semiand + ([ 'idtac' ] * delta)
                if show_debug:
                    print('Replaced with: {}'.format(show_semiand(next_semiand)))
            # Haven't taken care of 3. yet
            for i in range(len(next_semiand)):
                # note that next_semiand may itself be `a ; [b | c] ; d`
                new_semiands = list(split_semis_brackets(next_semiand[i], with_tactic))
                if show_debug:
                    print("Dispatching branch {}/{} with semiands {}".format(str(1+i), str(len(next_semiand)), show_semiands(new_semiands)))
                yield from lin(new_semiands + semiands[:], periodands, branch_done(done, nb_subgoals, i))
            if show_debug:
                print("Done dispatching {} tactics".format(str(len(next_semiand))))
            return


        return

    if len(commands) == 0:
        raise "Error: called linearize_proof with empty commands"
    first_tactic = commands.pop(0)
    semiands = list(split_semis_brackets(first_tactic, with_tactic))
    yield from lin(semiands, commands, 0)
