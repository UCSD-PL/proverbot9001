#!/usr/bin/env python3

def format_command_record(prev_tactics, prev_hyps, prev_goal,
                          tactic, post_hyps, post_goal):
    buf = ""
    buf += "*****\n"
    buf += prev_goal + "\n"
    buf += "+++++\n"
    buf += tactic + "\n"
    return buf
