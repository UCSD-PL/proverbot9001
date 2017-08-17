#!/usr/bin/env python3

import re

def minimize_whitespace(data):
    return re.sub("\s+", " ", data).strip()

def format_context(prev_tactics, prev_hyps, prev_goal):
    return minimize_whitespace(prev_goal) + "\n+++++\n"

def format_tactic(tactic):
    return minimize_whitespace(tactic) + "\n-----\n"

def read_pair(f_handle):
    context = f_handle.readline()
    if context == None:
        return None
    plusses = f_handle.readline()
    assert(plusses == "+++++\n")
    tactic = f_handle.readline()
    assert(tactic != "")
    minuses = f_handle.readline()
    assert(minuses == "-----\n")
    return (context.strip(), tacitc.strip())
