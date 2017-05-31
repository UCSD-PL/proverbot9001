#!/usr/bin/env python3

import re

def minimize_whitespace(data):
    return re.sub("\s+", " ", data)

def format_context(prev_tactics, prev_hyps, prev_goal):
    return "*****\n" + minimize_whitespace(prev_goal) + "\n"

def format_tactic(tactic):
    return "+++++\n" + minimize_whitespace(tactic) + "\n"
