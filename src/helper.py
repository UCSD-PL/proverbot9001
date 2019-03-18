#!/usr/bin/env python3.7

import serapi_instance
from serapi_instance import AckError, CompletedError, CoqExn, BadResponse
import linearize_semicolons
import re
import os

from typing import List, Match, Any, Optional, Iterator

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
                       prelude : str, filename : str, skip_nochange_tac : bool, debug=False) -> List[str]:
    try:
        with serapi_instance.SerapiContext(coqargs, includes, prelude) as coq:
            coq.debug = debug
            result = list(linearize_semicolons.linearize_commands(generate_lifted(commands,
                                                                                  coq),
                                                                  coq, filename, skip_nochange_tac))
        return result
    except (CoqExn, BadResponse, AckError, CompletedError):
        print("In file {}".format(filename))
        raise
    except serapi_instance.TimeoutError:
        print("Timed out while lifting commands! Skipping linearization...")
        return commands

def generate_lifted(commands : List[str], coq : serapi_instance.SerapiInstance) \
    -> Iterator[str]:
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

import hashlib
BLOCKSIZE = 65536

def hash_file(filename : str) -> str:
    hasher = hashlib.md5()
    with open(filename, 'rb') as f:
        buf = f.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(BLOCKSIZE)
    return hasher.hexdigest()

def try_load_lin(filename : str, verbose:bool=True) -> Optional[List[str]]:
    if verbose:
        print("Attempting to load cached linearized version from {}"
              .format(filename + '.lin'))
    if not os.path.exists(filename + '.lin'):
        return None
    file_hash = hash_file(filename)
    with open(filename + '.lin', 'r') as f:
        if file_hash == f.readline().strip():
            return read_commands_preserve(f.read())
        else:
            return None

def save_lin(commands : List[str], filename : str) -> None:
    output_file = filename + '.lin'
    with open(output_file, 'w') as f:
        print(hash_file(filename), file=f)
        for command in commands:
            print(command.strip(), file=f)
