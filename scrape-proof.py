#!/usr/bin/env python3

import argparse
import subprocess
import re
import os
import sys
from coq import *

def get_response(response, response_type):
    result = ""
    if response_type == "context" or response_type == "both":
        result += get_context(response)
    if response_type == "both":
        result += "=====\n"
    if response_type == "goal" or response_type == "both":
        result += get_goal(response)
    return result
        
def lift_inner_lemmas(string):
    new_contents = ""
    commands = re.split("\.\s+", string)
    lemma_contents = ""
    proof_depth = 0
    for command in commands:
        if starting_proof(command):
            proof_depth += 1
        if proof_depth == 0 or proof_depth > 1:
            new_contents += command + ".\n"
        else:
            lemma_contents += command + ".\n"
        if ending_proof(command):
            if proof_depth > 0:
                proof_depth -= 1
            if proof_depth == 0:
                new_contents += lemma_contents
                lemma_contents = ""
    return new_contents

def kill_semis(coqargs, contents, debugfile, options):
    debug = options['debug']
    rollback_only = options['rollback']
    new_contents = ""
    coq = Coq(coqargs)
    coq.print_errors = False
    coq.debug = debug
    coq.pdump = options['pdump']
    coq.start()

    # Split into commands
    commands = re.split("\.\s+", contents)

    # Pull the welcome message
    coq.get_welcome()

    # Wait for the initial prompt
    coq.wait_for_prompt()

    stripping_proof = False
    in_proof = False
    fallback = False
    finished_proof = False

    contexts = []
    # Add the toplevel context
    subcommandss = []
    subgoal_depths = []
    subgoal_indices = []
    strpd_lemma_contents = ""
    orig_lemma_contents = ""

    proofs_stripped = 0
    proofs_unstripped = 0

    for command in commands:
        command = command.strip()
        if command == "":
            continue
        if command == "Proof" and not in_proof:
            print("Problem! Unrecognized proof start.")
            coq.kill()
            os._exit(2)

        # Make sure existentials get resolved properly even when
        # subgoals are finished.
        if "Grab" in command:
            finished_proof = False

        if (finished_proof and
            not ending_proof(command) and
            "Transparent" not in command):
            continue

        # Split commands on semicolons only if we're stripping a
        # proof.
        if in_proof and not fallback:
            command_parts = split_compound_command(command)
        else:
            command_parts = [command]

        try:
            if len(command_parts) == 1:
                # If we just get a normal command, run it normally, and
                # add it to the stripped version.
                if debugfile != None:
                    debugfile.write(command + ".\n")
                coq.run_command(command + ".")
                try:
                    response = coq.eget()
                except CoqError:
                    orig_lemma_contents += command + ".\n"
                    raise
                # Keep track of whether we're in the proof, because we only
                # want to kill semicolons that are in proofs. Otherwise we
                # would break other parts of Coq syntax, like, for instance,
                # tactic definitions.
                if starting_proof(command) or ("===" in response and not in_proof):
                    subcommandss = []
                    subgoal_depths = []
                    subgoal_indices = []
                    strpd_lemma_contents = command + ".\n"
                    orig_lemma_contents = command + ".\n"
                    in_proof = True
                else:
                    # If we just get a normal command, run it normally, and
                    # add it to the stripped version.
                    if in_proof and not fallback:
                        orig_lemma_contents += command + ".\n"
                        strpd_lemma_contents += command + ".\n"
                    else:
                        new_contents += command + ".\n"
            # When we see a semicolon in a proof script, break up the
            # command, and push a new subcommands and subgoal_depth so
            # that the loop below will start processing the new
            # command. subgoal_depth is counted from the leftmost
            # subcommand.
            else:
                subcommandss.append(command_parts)
                subgoal_depths.append(0)
                subgoal_indices.append([])
                orig_lemma_contents += command + ".\n"
            done = False
            while not done:
                done = True
                # If we just finished a sub-proof, we're going to want
                # to print closing brackets until we get to the next
                # uncompleted part of the goal.
                while ("This subproof is complete" in response or
                       ("No more subgoals" in response and
                        len(subgoal_depths) > 0 and
                        subgoal_depths[-1] > 0)):
                    # close a brace
                    coq.run_command("}")
                    response = coq.eget()
                    # We're now one subcommand shallower (counting from
                    # the left) into the last semi-colon command.
                    subgoal_depths[-1] -= 1
                    # If we finish the last semi-colon command by closing
                    # all the braces, pop the command.
                    if subgoal_depths[-1] == 0:
                        subgoal_depths.pop()
                        subcommandss.pop()
                        subgoal_indices.pop()
                    if "No more subgoals" in response and not fallback:
                        finished_proof = True
                # Unless we finish the current subcase with the subcommand
                # we're working on, we're going to want to move on to the
                # next command after this.
                if len(subgoal_depths) > 0:
                    # If we have a compound command on the stack but
                    # haven't run it's subcommands, run any we haven't run.
                    for subcommand in subcommandss[-1][subgoal_depths[-1]:]:
                        # Kill spaces that were used for spacing the
                        # semicolon. Isn't strictly necessary but makes
                        # the -v output prettier.
                        subcommand = subcommand.strip()
                        # If this is the first time we've gotten this deep
                        # in this compound command, we're going to append
                        # a new subgoal index counter for this depth.
                        if len(subgoal_indices[-1]) <= subgoal_depths[-1]:
                            subgoal_indices[-1].append(0)
                        # Otherwise, increment the counter we already have
                        # for this depth.
                        else:
                            subgoal_indices[-1][subgoal_depths[-1]] += 1
                        # Okay, now there's two kinds of commands WITHIN
                        # compound commands: simple tactics invocations,
                        # where we want to just run them on each of our
                        # goals, and [ tac1 | tac2 | ... | tacn ] things,
                        # where we should only run each branch on the
                        # subgoal that matches it's index on this
                        # branch. Then of course, each of those branches
                        # can be it's own compound tactic.
    
                        if re.match("\[.*\]", subcommand):
                            branches = subcommand.strip('[]').split("|")
                            subcommand = branches[subgoal_indices[-1][subgoal_depths[-1]]].strip()
                            # Handle branches being their own compound
                            # commands by pushing a new compound command
                            # entry, and then restarting the outer while
                            # loop ("while not done:").
                            if ";" in subcommand:
                                subcommandss.append(subcommand.split(';'))
                                subgoal_depths.append(0)
                                subgoal_indices.append([])
                                done = False
                                break
                        # Open a new set of brackets for each subcommand
                        # so that we can track when their part of the
                        # proof is finished.
                        coq.run_command("{")
                        coq.eget()
                        # Run the command, and add it to our stripped
                        # version.
                        if debugfile != None:
                            debugfile.write(subcommand + ".\n")
                        coq.run_command(subcommand + ".")
                        strpd_lemma_contents += subcommand + ".\n"
                        # We've run a command, so we're one subcommand
                        # more "deep"
                        subgoal_depths[-1] += 1
                        # Get the response, and check if we finished our
                        # subgoal. If we did, we'll want to jump out of it
                        # until we get to one we haven't finished or we
                        # finish the command, so set done to false so we
                        # can go back to the preceding while loop.
                        response = coq.eget()
                        if "This subproof is complete" in response or "No more subgoals" in response:
                            done = False
                            break

            if ending_proof(command):
                finished_proof = False
                if in_proof:
                    if not fallback:
                        new_contents += strpd_lemma_contents
                        proofs_stripped += 1
                    else:
                        proofs_unstripped += 1
                        fallback = False
                    in_proof = False

                    subcommandss = []
                    subgoal_depths = []
                    subgoal_indices = []
                    strpd_lemma_contents = ""
                    orig_lemma_contents = ""
        except CoqError:
            if rollback_only and int(rollback_only) == proofs_unstripped:
                raise
            fallback = True
            finished_proof = False
            in_proof = True
            if debugfile:
                debugfile.write("Abort.\n")
                debugfile.write("Back.\n")
            coq.run_command("Abort.")
            coq.eget()
            coq.run_command("Back.")
            coq.eget()
            for command in re.split("\.\s+", orig_lemma_contents):
                command = command.strip()
                if command == "":
                    continue
                if ending_proof(command):
                    fallback = False
                    in_proof = False
                    proofs_unstripped += 1
                if debugfile:
                    debugfile.write(command + ".\n")
                coq.run_command(command + ".")
                coq.eget()
                new_contents += command + ".\n"

            subcommandss = []
            subgoal_depths = []
            subgoal_indices = []
            strpd_lemma_contents = ""
            orig_lemma_contents = ""

        # Print some commas to stdout to let the user know we're working.
        sys.stderr.write(",")
        sys.stderr.flush()

    coq.kill()
    print()
    print("Successfully stripped {} out of {} proofs.".format(proofs_stripped, proofs_stripped + proofs_unstripped))

    return new_contents

def process_file(fd, coqargs, fout, options):
    coq = Coq(coqargs)
    coq.start()
    coq.debug = options['debug']

    # We don't start in a proof
    in_proof = False

    # Get the entire input file
    contents = fd.read()
    # Kill comments
    contents = kill_comments(contents)
    contents = kill_brackets(contents)
    # Kill bullets. We have to be careful to not kill stars, plusses,
    # and minuses in normal galina, so we only remove those characters
    # when they are seperated by a preceding period by only whitespace.
    contents = kill_bullets(contents)
    contents = kill_brackets(contents)

    if options['v0'] != None:
        with open(options['v0'], 'w') as sout_file:
            sout_file.write(contents)

    contents = lift_inner_lemmas(contents)

    if options['v1'] != None:
        with open(options['v1'], 'w') as sout_file:
            sout_file.write(contents)

    if options['v2'] != None:
        with open(options['v2'], 'w') as sout_file:
            contents = kill_semis(coqargs, contents, sout_file, options)
    else:
        contents = kill_semis(coqargs, contents, None, options)

    if options['v3'] != None:
        with open(options['v3'], 'w') as sout_file:
            sout_file.write(contents)

    # Split it into commands
    commands = re.split("\.\s+", contents)
    tactics_run = []
    cur_context = ""

    # Pull the welcome message
    coq.get_welcome()

    # Wait for the initial prompt
    coq.wait_for_prompt()
    
    # Execute each command
    for command in commands:
        if command == "":
            continue

        command = command + "."

        # Write out the command
        coq.run_command(command)

        response = coq.eget()
        # If we start a proof block, write out the initial context and
        # start proof mode, and then continue.
        if starting_proof(command) or (not in_proof and "===" in response):
            if not in_proof:
                cur_context = get_response(response, options['print'])
                in_proof = True
            continue
        if command == "Proof.":
            continue
        # If we end a proof block, just flush the response and move
        # on, turning off proof mode.
        elif ending_proof(command):
            in_proof = False
            tactics_run = []
            continue

        # When we're in the proof, write out the commands and context
        # responses.  Ignore responses to non-proof commands.
        if in_proof:
            if options["prev-tactics"]:
                if len(tactics_run) > 5:
                    for tactic in tactics_run[-5:]:
                        fout.write(tactic + "\n")
                else:
                    for tactic in tactics_run:
                        fout.write(tactic + "\n")
                fout.write("+++++\n")
            fout.write(cur_context)
            fout.write("*****\n")
            fout.write(command + "\n")
            tactics_run.append(command)
            if options['debug']:
                print(command)
            fout.write("%%%%%\n")
            cur_context = get_response(response, options['print'])

        # Print some periods to stdout to let the user know we're
        # working.
        if not fout == sys.stdout:
            sys.stderr.write(".")
            sys.stderr.flush()

    coq.kill()

includes=subprocess.Popen(['make', 'print-includes'], stdout=subprocess.PIPE).communicate()[0]

parser = argparse.ArgumentParser(description="scrape the context and tactics from each proof in a file.")
parser.add_argument('-o', '--output', help="output data file name", default=None)
parser.add_argument('--skip-errors', help="Skip the rest of files that error instead of exiting.",
                    default=False, const=True, action='store_const', dest='skip')
parser.add_argument('--log-errors', help="Log the names of files that failed.",
                    default=None, dest="elog")
parser.add_argument('-v0', default=None)
parser.add_argument('-v1', default=None)
parser.add_argument('-v2', default=None)
parser.add_argument('-v3', default=None)
parser.add_argument('--debug', help="print everything.", default=False, const=True, action='store_const', dest='debug')
parser.add_argument('--debug-stripping', help="print everything during stripping.", default=False, const=True, action='store_const', dest='debug_stripping')
parser.add_argument('--rollback-only', default=None, dest='rollback')
parser.add_argument('inputs', nargs='+', help="proof file name (*.v)")
parser.add_argument('--print', choices=['goal', 'context', 'both'], default='goal', dest='which_print')
parser.add_argument('--dump-prompts', default=None, dest='pdump')
parser.add_argument('--prev-tactics', default=False, const=True, action='store_const', dest='prev_tactics')

args = parser.parse_args()
options = {'skip': args.skip, 'debug': args.debug, 'debug-stripping': args.debug_stripping,
           'v0': args.v0, 'v1': args.v1, 'v2':args.v2, 'v3':args.v3,
           'rollback': args.rollback, 'print': args.which_print,
           'pdump': args.pdump, 'prev-tactics': args.prev_tactics}

# The command for running coq
coqargs = ["coqtop"] + includes.split()

if args.output != None:
    fout = open(args.output, 'w')
else:
    fout = sys.stdout

if args.elog != None:
    elog = open(args.elog, 'w')
else:
    elog = None

num_skipped = 0

# Open the input and output files
for idx, infname in enumerate(args.inputs):
    print()
    print("Processing file {} ({} of {})".format(infname, idx + 1,
                                                 len(args.inputs)))
    with open(infname) as fin:
        try:
            process_file(fin, coqargs, fout, options)
        except CoqError:
            print("Proof failed")
            if elog != None:
                elog.write("{}\n".format(infname))
                elog.flush()
                print("Logging failure of {}".format(infname))
            else:
                print("No log.")
            if not options['skip']:
                os._exit(1)
            else:
                num_skipped += 1
if options['skip']:
    print()
    print("Skipped {} files because of errors.".format(num_skipped))

if args.output != None:
    fout.close()
if args.elog != None:
    elog.close()

print()
os._exit(0)
