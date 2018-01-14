#!/usr/bin/env python3

import re

def next_char(c):
    return chr(ord(c) + 1)

def make_fresh():
    next = chr(128)#"\uAC00" # Hangul syllables (~11k symbols)
    def fresh():
        nonlocal next
        curr = next
        next = next_char(next)
        return curr
    return fresh

fresh = make_fresh()

patterns = [
    "apply",
    "assert",
    "auto",
    "case",
    "clear",
    "destruct",
    "discriminate",
    "eapply",
    "first",
    "generalize",
    "induction",
    "intro",
    "intros",
    "intuition",
    "inversion",
    "reflexivity",
    "revert",
    "rewrite",
    "transitivity",
    "unfold",
]

num_tokenizer_patterns = len(patterns)

tokens = map(lambda p: (p, fresh()), patterns)

# Two dictionaries for fast lookup both ways:

dict_pattern_to_token = {}
dict_token_to_pattern = {}
for (p, t) in tokens:
    dict_pattern_to_token[p] = t
    dict_token_to_pattern[t] = p

def pattern_to_token(s):
    for k in dict_pattern_to_token:
        s = re.sub(fr"(^|(?<=[ ])){k}(?=[ ]|;|.)", dict_pattern_to_token[k], s)
    return s

def token_to_pattern(s):
    for k in dict_token_to_pattern:
        s = re.sub(fr"(^|(?<=[ ])){k}(?=[ ]|;|.)", dict_token_to_pattern[k], s)
    return s
