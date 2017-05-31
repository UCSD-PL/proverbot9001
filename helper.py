#!/usr/bin/env python3

import serapi_instance
import linearize_semicolons
import re

def load_commands(filename):
    with open(filename, 'r') as fin:
        contents = serapi_instance.kill_comments(fin.read())
        commands_orig = serapi_instance.split_commands(contents)
        commands_preprocessed = [newcmd for cmd in commands_orig
                                 for newcmd in serapi_instance.preprocess_command(cmd)]
        return commands_preprocessed

def lifted_vernac(command):
    return re.match("Ltac\s", command)

def lift_and_linearize(commands, coqargs, includes, filename):
    with serapi_instance.SerapiContext(coqargs, includes) as coq:
        result = list(linearize_semicolons.linearize_commands(generate_lifted(commands, coq),
                                                              coq, filename))
        return result

def generate_lifted(commands, coq):
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
        assert(len(lemma_stack) == 0)
    except Exception as e:
        coq.kill()
        raise e
