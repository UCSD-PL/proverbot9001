#!/usr/bin/env python3

import argparse
import datetime
import os
import os.path
import queue
import re
import subprocess
import sys
import threading

# This dependency is in pip, the python package manager
from sexpdata import *
from timer import TimerBucket
from traceback import *
from compcert_linearizer_failures import compcert_failures

import serapi_instance
from serapi_instance import (AckError, CompletedError, CoqExn,
                             BadResponse, ParseError, get_stem)

from typing import Optional, List, Iterator

# exception for when things go bad, but not necessarily because of the linearizer
class LinearizerCouldNotLinearize(Exception):
    pass

# exception for when the linearizer trips onto itself
class LinearizerThisShouldNotHappen(Exception):
    pass

measure_time = False

# stop_on_error: whether the program will stop when a linearization fails
# show_trace:    whether the program will show all the Coq commands it outputs
# show_debug:    whether the program will explain everything it's doing
stop_on_error = False
show_trace    = False
show_debug    = False

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
            s = s.replace('...', ".")
        else:
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
        raise LinearizerThisShouldNotHappen("FIXME")

# a semiand is just some pipeands
def show_semiand(pipeands):
    if len(pipeands) == 1:
        return pipeands[0]
    else:
        #print(str(pipeands))
        return '[ {} ]'.format(' | '.join(pipeands))

def show_semiands(semiands):
    return ' ; '.join(map(show_semiand, semiands))

# def recount_open_proofs(coq):
#     goals = coq.query_goals()
#     return count_open_proofs(goals)

def linearize_commands(commands_sequence, coq, filename):
    if show_debug:
        print("Linearizing commands")
    command = next(commands_sequence, None)
    while command:
        # Run up to the next proof
        while coq.count_fg_goals() == 0:
            coq.run_stmt(command)
            if coq.count_fg_goals() == 0:
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
        theorem_statement = serapi_instance.kill_comments(command_batch.pop(0))
        theorem_name = theorem_statement.split(":")[0].strip()
        coq.run_stmt(theorem_statement)
        yield theorem_statement
        if [filename, theorem_name] in compcert_failures:
            print("Skipping {}".format(theorem_name))
            for command in command_batch:
                coq.run_stmt(command)
                yield command
            command = next(commands_sequence, None)
            continue

        # This might not be super robust?
        match = re.fullmatch("\s*Proof with (.*)\.", command_batch[0])
        if match and match.group(1):
            with_tactic = match.group(1)
        else:
            with_tactic = ""

        orig = command_batch[:]
        command_batch = list(split_commas(command_batch))
        try:
            linearized_commands = list(linearize_proof(coq, theorem_name, with_tactic, command_batch))
            leftover_commands = []

            for command in command_batch:
                # goals = coq.query_goals()
                if command and (coq.count_fg_goals() != 0 or
                                serapi_instance.ending_proof(command) or
                                "Transparent" in command):
                    coq.run_stmt(command)
                    leftover_commands.append(command)

            yield from linearized_commands
            for command in leftover_commands:
                yield command
        except (BadResponse, CoqExn, LinearizerCouldNotLinearize, ParseError) as e:
            print("Aborting current proof linearization!")
            print("Proof of:\n{}\nin file {}".format(theorem_name, filename))
            print()
            if stop_on_error:
                raise e
            coq.run_stmt("Abort.")
            coq.run_stmt(theorem_statement)
            for command in orig:
                if command:
                    coq.run_stmt(command)
                    yield command

        command = next(commands_sequence, None)

# semiands   : [[String]]
# ksemiands  : [[[String]]]
# periodands : [String]
# done       : Int

def branch_done(done, nb_subgoals, branch_index):
    return done + nb_subgoals - 1 - branch_index

if measure_time:
    linearizing_timer_bucket = TimerBucket("linearizing", True)

def linearize_proof(coq, theorem_name, with_tactic, commands):

    theorem_name = theorem_name.split(" ")[1]
    if measure_time:
        count_goals_before_timer_bucket = TimerBucket("{} / counting goals before".format(theorem_name), False)
        run_statement_timer_bucket = TimerBucket("{} / running statement".format(theorem_name), False)
        count_goals_after_timer_bucket = TimerBucket("{} / counting goals after".format(theorem_name), False)

    #print("linearize_proof(coq, '{}', {})".format(
    #    with_tactic, str(commands)
    #))

    def linearize_periodands(periodands, done):
        if show_debug:
            print("Linearizing next periodands, done when: {}".format(str(done)))
        if len(periodands) == 0:
            raise LinearizerCouldNotLinearize("Error: ran out of tactic w/o finishing the proof")
        next_tactic = periodands.pop(0)
        if next_tactic == None:
            return
        while next_tactic and re.match("\s*[+\-*]+|\s*{|\s*}",
                                       serapi_instance.kill_comments(next_tactic)):
            if len(periodands) == 0:
                raise LinearizerCouldNotLinearize("Error: ran out of tactics w/o finishing the proof")
            else:
                if show_debug:
                    print("Skipping bullet: {}".format(next_tactic))
                maybe_next_tactic = periodands.pop(0)
                if maybe_next_tactic:
                    next_tactic = maybe_next_tactic
                else:
                    break
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

        if measure_time: stop_timer = count_goals_before_timer_bucket.start_timer("")
        nb_goals_before = coq.count_fg_goals()
        if measure_time: stop_timer()
        if show_debug:
            print("Goals before: {}".format(str(nb_goals_before)))

        if measure_time: stop_timer = run_statement_timer_bucket.start_timer("")
        # This can happen when a semiand was empty for instance, pop a periodand instead
        if len(semiands) == 0:
            yield from linearize_periodands(periodands, done)
            return
        semiand = semiands.pop(0)
        # dispatch is now preprocessed when we have the information on subgoals
        # available, so popped semiands ought to be just one command
        if len(semiand) != 1:
            raise LinearizeThisShouldNotHappen("Error: popped a semiand that was not preprocessed")
        tactic = '\n' + semiand[0].strip() + '.'
        context_before = coq.proof_context
        coq.run_stmt(tactic)
        context_after = coq.proof_context
        if show_trace:
            print('    ' + tactic)
        if (context_before != context_after or
            not (re.match(".*auto", tactic)  or
                 re.match(".*Proof", tactic))):
            yield tactic
        else:
            # print("Skipping {} because it didn't change the context.".format(tactic))
            pass
        if measure_time: stop_timer()

        if measure_time: stop_timer = count_goals_after_timer_bucket.start_timer("")
        nb_goals_after = coq.count_fg_goals()
        if measure_time: stop_timer()
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
            raise LinearizerThisShouldNotHappen("Peeked an empty semiand, this should not happen")
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
                raise LinearizerThisShouldNotHappen("Error: empty next semiand")
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

    if measure_time: stop_timer = linearizing_timer_bucket.start_timer(theorem_name)
    if len(commands) == 0:
        raise LinearizerThisShouldNotHappen("Error: called linearize_proof with empty commands")
    first_tactic = commands.pop(0)
    semiands = list(split_semis_brackets(first_tactic, with_tactic))
    yield from lin(semiands, commands, 0)
    if measure_time: stop_timer()

    if measure_time:
        count_goals_before_timer_bucket.print_statistics()
        run_statement_timer_bucket.print_statistics()
        count_goals_after_timer_bucket.print_statistics()

def split_commas(commands : Iterator[str]) -> Iterator[str]:
    def split_commas_command(command : str) -> Iterator[str]:
        if not "," in command:
            yield command
        else:
            stem = get_stem(command)
            if stem == "rewrite":
                in_match = re.match("\s*(rewrite\s+\S*?),\s+(.*)\s+in(.*\.)", command)
                if in_match:
                    first_command, rest, context = in_match.group(1, 2, 3)
                    print("Splitting {} into {} and {}"
                          .format(command, first_command + " in" + context,
                                  "rewrite " + rest + " in" + context))
                    yield first_command + " in" + context
                    yield from split_commas_command("rewrite " + rest + " in" + context)
                else:
                    parts_match = re.match("\s*(rewrite\s+!?\s*\S*?),\s+(.*)", command)
                    assert parts_match, "Couldn't match \"{}\"".format(command)
                    first_command, rest = parts_match.group(1, 2)
                    print("Splitting {} into {} and {}"
                          .format(command, first_command + ". ", "rewrite " + rest))
                    yield first_command + ". "
                    yield from split_commas_command("rewrite " + rest)
            else:
                yield command
    new_commands : List[str] = []
    for command in commands:
        split_commands = split_commas_command(command)
        # print("Split {} into {}".format(command, list(split_commands)))
        yield from split_commands
