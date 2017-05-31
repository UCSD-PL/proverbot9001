#!/usr/bin/env python3

import re

def minimize_whitespace(data):
    return re.sub("\s+", " ", data)

def format_command_record(prev_tactics, prev_hyps, prev_goal,
                          tactic, post_hyps, post_goal):
    buf = ""
    buf += "*****\n"
    buf += minimize_whitespace(prev_goal) + "\n"
    buf += "+++++\n"
    buf += minimize_whitespace(tactic) + "\n"
    return buf
