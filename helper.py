#!/usr/bin/env python3

import serapi_instance
from serapi_instance import AckError, CompletedError, CoqExn, BadResponse
import linearize_semicolons
import re

from typing import List, Match, Any, Optional

def load_commands(filename : str) -> List[str]:
    with open(filename, 'r') as fin:
        contents = serapi_instance.kill_comments(fin.read())
        commands_orig = serapi_instance.split_commands(contents)
        commands_preprocessed = [newcmd for cmd in commands_orig
                                 for newcmd in serapi_instance.preprocess_command(cmd)]
        return commands_preprocessed

def load_commands_preserve(filename : str) -> List[str]:
    with open(filename, 'r') as fin:
        contents = fin.read()
    return read_commands_preserve(contents)

def read_commands_preserve(contents : str) -> List[str]:
    result = []
    cur_command = ""
    comment_depth = 0
    in_quote = False
    for i in range(len(contents)):
        cur_command += contents[i]
        if in_quote:
            if contents[i] == '"' and contents[i-1] != '\\':
                in_quote = False
        else:
            if contents[i] == '"' and contents[i-1] != '\\':
                in_quote = True
            elif comment_depth == 0:
                if (re.match("[\{\}]", contents[i]) and
                      re.fullmatch("\s*", cur_command[:-1])):
                    result.append(cur_command)
                    cur_command = ""
                elif (re.fullmatch("\s*[\+\-\*]+",
                                   serapi_instance.kill_comments(cur_command)) and
                      (len(contents)==i+1 or contents[i] != contents[i+1])):
                    result.append(cur_command)
                    cur_command = ""
                elif (re.match("\.($|\s)", contents[i:i+2]) and
                      (not contents[i-1] == "." or contents[i-2] == ".")):
                    result.append(cur_command)
                    cur_command = ""
            if contents[i:i+2] == '(*':
                comment_depth += 1
            elif contents[i-1:i+1] == '*)':
                comment_depth -= 1
    return result

def lifted_vernac(command : str) -> Optional[Match[Any]]:
    return re.match("Ltac\s", serapi_instance.kill_comments(command).strip())

def lift_and_linearize(commands : List[str], coqargs : List[str], includes : str,
                       prelude : str, filename : str, debug=False) -> List[str]:
    try:
        with serapi_instance.SerapiContext(coqargs, includes, prelude) as coq:
            coq.debug = debug
            result = list(linearize_semicolons.linearize_commands(generate_lifted(commands,
                                                                                  coq),
                                                                  coq, filename))
        return result
    except (CoqExn, BadResponse, AckError, CompletedError):
        print("In file {}".format(filename))
        raise

def generate_lifted(commands : List[str], coq : serapi_instance.SerapiInstance):
    lemma_stack = [] # type: List[List[str]]
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
