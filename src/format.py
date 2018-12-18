#!/usr/bin/env python3

import re
from typing import List, Tuple, TextIO, Optional, NamedTuple, Union

class ScrapedTactic(NamedTuple):
    prev_tactics : List[str]
    hypotheses : List[str]
    goal : str
    tactic : str

ScrapedCommand = Union[ScrapedTactic, str]

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
    return "\n".join([re.sub(r"\n", r"\\n", re.sub("[ \t]+", " ", prev_hyp.strip())).strip() for prev_hyp in prev_hyps])

def format_goal(prev_goal : str) -> str:
    return minimize_whitespace(prev_goal)

def format_lemmas(rel_lemmas : str) -> str:
    return re.sub("[ \t]+", " ", rel_lemmas).strip()

def format_tactic(tactic : str):
    return minimize_whitespace(tactic) + "\n-----\n"

def read_tuple(f_handle : TextIO) -> Optional[ScrapedCommand]:
    lines : List[str] = []
    next_line = f_handle.readline()
    while next_line != "-----\n" and next_line != "":
        lines.append(next_line)
        next_line = f_handle.readline()
    if len(lines) == 0:
        return None
    elif len(lines) == 1:
        return "\n" + re.sub(r"\\n", r"\n", lines[0])
    else:
        prev_tactics : List[str] = []
        lines_it = iter(lines)
        for line in lines_it:
            if line == "*****\n":
                break
            elif line.strip() == "":
                continue
            else:
                prev_tactics.append(line.strip())
        hyps : List[str] = []
        for line in lines_it:
            if line == "*****\n":
                break
            elif line.strip() == "":
                continue
            else:
                hyps.append(line.strip())
        goal = next(lines_it)
        assert next(lines_it) == "+++++\n"
        tactic = next(lines_it)
        return ScrapedTactic(prev_tactics=prev_tactics, hypotheses=hyps,
                             goal=goal, tactic=tactic)

def read_tactic_tuple(f_handle : TextIO) -> Optional[ScrapedTactic]:
    next_tuple = read_tuple(f_handle)
    while(isinstance(next_tuple, str)):
        next_tuple = read_tuple(f_handle)
    return next_tuple
