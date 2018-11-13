#!/usr/bin/env python3

import re
from typing import List, Tuple, TextIO, Optional

def minimize_whitespace(data : str) -> str:
    return re.sub("\s+", " ", data).strip()

def format_context(prev_tactics : List[str], prev_hyps : List[str], prev_goal : str,
                   rel_lemmas : str) -> str:
    return (format_tactics(prev_tactics) + "\n*****\n" +
            format_hypothesis(prev_hyps) + "\n*****\n" +
            # format_lemmas(rel_lemmas) + "*****\n" +
            format_goal(prev_goal) + "\n+++++\n")

def format_tactics(tactics : List[str]) -> str:
    return "\n".join([minimize_whitespace(tactic) for tactic in tactics]) + "\n"

def format_hypothesis(prev_hyps : List[str]) -> str:
    return "\n".join([re.sub(r"\n", r"\\n", re.sub("[ \t]+", " ", prev_hyp)).strip() for prev_hyp in prev_hyps])

def format_goal(prev_goal : str) -> str:
    return minimize_whitespace(prev_goal)

def format_lemmas(rel_lemmas : str) -> str:
    return re.sub("[ \t]+", " ", rel_lemmas).strip()

def format_tactic(tactic : str):
    return minimize_whitespace(tactic) + "\n-----\n"

def read_tuple(f_handle : TextIO) -> Optional[Tuple[List[str], str, str]]:
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

    hypotheses = [] # type: List[str]
    next_hypothesis = f_handle.readline().strip()
    while next_hypothesis != "*****":
        if next_hypothesis == "":
            next_hypothesis = f_handle.readline().strip()
            continue
        hypotheses.append(next_hypothesis)
        next_hypothesis = f_handle.readline().strip()
    for hyp in hypotheses:
        assert ":" in hyp, "hyps: {}".format(hypotheses)

    stars2 = next_hypothesis
    assert stars2 == "*****"

    goal = f_handle.readline().strip()
    assert goal != "", "Lemma name is {}".format(prev_tactics[0])
    plusses = f_handle.readline()
    assert plusses == "+++++\n", \
        "Plusses line is: {}, goal is {}, hypotheses are {}"\
        .format(plusses, goal, hypotheses)
    tactic = f_handle.readline()
    assert tactic != ""
    minuses = f_handle.readline()
    assert minuses == "-----\n", "Minuses line is: {}, goal is: {}".format(
        minuses, goal)
    return ([hyp.strip() for hyp in hypotheses], goal.strip(), tactic.strip())
