#!/usr/bin/env python3

import re

debug_tokenizer = False

def next_char(c):
    return chr(ord(c) + 1)

def make_fresh():
    next = "\uAC00" # Hangul syllables (~11k symbols)
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
    "eauto",
    "auto",
    "case",
    "clear",
    "destruct",
    "discriminate",
    "eapply",
    "first",
    "generalize",
    "induction",
    "intros",
    "intro",
    "intuition",
    "inversion",
    "inv",
    "reflexivity",
    "revert",
    "rewrite",
    "transitivity",
    "unfold",
    "with",
    "set",
    "simpl",
    "try",
    "congruence",
    "omega",
    "repeat"
    "as",
    "using",
    "exact",
]

tokens = map(lambda p: (p, fresh()), patterns)

# Two dictionaries for fast lookup both ways:

dict_pattern_to_token = {}
dict_token_to_pattern = {}
for (p, t) in tokens:
    dict_pattern_to_token[p] = t
    dict_token_to_pattern[t] = p

def pattern_to_token(s):
    s_in = s
    for k in dict_pattern_to_token:
        s = re.sub("(^|(?<=[ ])){}(?=[ ]|;|.)".format(k), dict_pattern_to_token[k], s)
    if debug_tokenizer:
        print("{} -> {}".format(s_in, [ord(c) for c in s]))
    return s

def token_to_pattern(s):
    s_in = s
    for k in dict_token_to_pattern:
        s = re.sub("(^|(?<=[ ])){}(?=[ ]|;|.)".format(k), dict_token_to_pattern[k], s)
    if debug_tokenizer:
        print("{} -> {}".format([ord(c) for c in s_in], s))
    return s

def encode_tactic(tactic):
    tactic = pattern_to_token(tactic)
    tokenlist = translate(tactic)
    return tokenlist
def encode_context(context):
    return translate(context)

def decode_tactic(tokenlist):
    tactic = untranslate(tokenlist)
    tactic = token_to_pattern(tactic)
    return tactic
def decode_context(context):
    return untranslate(context)

num_reserved_tokens = 2 # We want '0' to be reserved for "end of stream" and '1' to be reserved for "start of stream"

char_to_num = {}
num_to_char = {}

def get_encoder_state():
    return list(char_to_num.items())

def set_encoder_state(keypairs):
    global char_to_num
    global num_to_char
    char_to_num = {}
    num_to_char = {}
    for k, v in keypairs:
        assert isinstance(k, str)
        assert isinstance(v, int)
        char_to_num[k] = v
        num_to_char[v] = k

def text_vocab_size():
    return num_reserved_tokens + len(char_to_num)# + len(patterns)

def translate(string):
    result = []
    for c in string:
        if c in char_to_num:
            result += [char_to_num[c]]
        else:
            new_id = len(char_to_num) + num_reserved_tokens
            char_to_num[c] = new_id
            num_to_char[new_id] = c
            result += [new_id]
    return result

def untranslate(tokenlist):
    return "".join([num_to_char[t] for t in tokenlist])
