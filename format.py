#!/usr/bin/env python3

import re

def minimize_whitespace(data):
    return re.sub("\s+", " ", data).strip()

def format_context(prev_tactics, prev_hyps, prev_goal):
    return (format_tactics(prev_tactics) + "*****\n" +
            format_hypothesis(prev_hyps) + "*****\n" +
            format_goal(prev_goal) + "+++++\n")

def format_tactics(tactics):
    return "\n".join([minimize_whitespace(tactic) for tactic in tactics]) + "\n"

def format_hypothesis(prev_hyps):
    return re.sub("[ \t]+", " ", prev_hyps).strip() + "\n"

def format_goal(prev_goal):
    return minimize_whitespace(prev_goal) + "\n"

def format_tactic(tactic):
    return minimize_whitespace(tactic) + "\n-----\n"

def read_pair(f_handle):
    prev_tactics = []
    next_prev_tactic = f_handle.readline()
    if next_prev_tactic == "":
        return None
    while next_prev_tactic != "*****\n":
        assert next_prev_tactic != ""
        prev_tactics.append(next_prev_tactic)
        next_prev_tactic = f_handle.readline()
    stars = next_prev_tactic

    hypotheses = f_handle.readline()
    assert hypotheses != ""
    stars2 = f_handle.readline()
    assert stars2 == "*****\n", "Stars2 line is: {}".format(stars2)

    context = f_handle.readline()
    assert context != ""
    plusses = f_handle.readline()
    assert plusses == "+++++\n", "Plusses line is: {}, context is {}".format(
        plusses, context)
    tactic = f_handle.readline()
    assert tactic != ""
    minuses = f_handle.readline()
    assert minuses == "-----\n", "Minuses line is: {}, context is: {}".format(
        minuses, context)
    return (context.strip(), tactic.strip())
