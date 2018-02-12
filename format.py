#!/usr/bin/env python3

import re

def minimize_whitespace(data):
    return re.sub("\s+", " ", data).strip()

def format_context(prev_tactics, prev_hyps, prev_goal, rel_lemmas):
    return (format_tactics(prev_tactics) + "\n*****\n" +
            format_hypothesis(prev_hyps) + "\n*****\n" +
            # format_lemmas(rel_lemmas) + "*****\n" +
            format_goal(prev_goal) + "\n+++++\n")

def format_tactics(tactics):
    return "\n".join([minimize_whitespace(tactic) for tactic in tactics]) + "\n"

def format_hypothesis(prev_hyps):
    return re.sub("[ \t]+", " ", prev_hyps).strip()

def format_goal(prev_goal):
    return minimize_whitespace(prev_goal)

def format_lemmas(rel_lemmas):
    return re.sub("[ \t]+", " ", rel_lemmas).strip()

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
    assert stars == "*****\n"

    hypotheses = []
    next_hypothesis = f_handle.readline()
    while next_hypothesis != "*****\n":
        assert hypotheses != ""
        hypotheses.append(next_hypothesis)
        next_hypothesis = f_handle.readline()

    stars2 = next_hypothesis
    assert stars2 == "*****\n"

    goal = f_handle.readline()
    assert goal != ""
    plusses = f_handle.readline()
    assert plusses == "+++++\n", "Plusses line is: {}, goal is {}".format(
        plusses, goal)
    tactic = f_handle.readline()
    assert tactic != ""
    minuses = f_handle.readline()
    assert minuses == "-----\n", "Minuses line is: {}, goal is: {}".format(
        minuses, goal)
    return (goal.strip(), tactic.strip())
