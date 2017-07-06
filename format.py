#!/usr/bin/env python3

import re

def minimize_whitespace(data):
    return re.sub("\s+", " ", data).strip()

def format_context(prev_tactics, prev_hyps, prev_goal):
    return minimize_whitespace(prev_goal) + "\n+++++\n"

def format_tactic(tactic):
    return minimize_whitespace(tactic) + "\n-----\n"
