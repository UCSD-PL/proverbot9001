#!/usr/bin/env python3

import re
from typing import Dict, List, Tuple, Callable, Union

TokenizerState = Tuple[List[Tuple[str, int]], List[str], int]

class Tokenizer:
    def __init__(self, distinguished_strings : List[str], num_reserved_tokens : int = 0) \
        -> None:
        self.num_reserved_tokens = num_reserved_tokens
        self.distinguished_strings = distinguished_strings
        self.next_mangle_ord = num_reserved_tokens + len(distinguished_strings)
        self.mangle_dict = {} # type: Dict[str, int]
        self.unmangle_dict = {} # type: Dict[int, str]
        pass

    def _mangle(self, string : str) -> str:
        for c in string:
            if not c in self.mangle_dict:
                self.mangle_dict[c] = self.next_mangle_ord
                self.unmangle_dict[self.next_mangle_ord] = c
                self.next_mangle_ord += 1
        return "".join([chr(self.mangle_dict[c]) for c in string])

    def toTokenList(self, string : str) -> List[int]:
        mangled_string = self._mangle(string)

        for idx, token_string in enumerate(self.distinguished_strings,
                                           start=self.num_reserved_tokens):
            mangled_string = re.sub(self._mangle(token_string), chr(idx), mangled_string)

        for c in mangled_string:
            assert ord(c) < self.numTokens()
        tokenlist = [ord(c) for c in mangled_string]
        assert self.toString(tokenlist) == string
        return tokenlist

    def toString(self, idxs : List[int]) -> str:
        result = ""
        for t in idxs:
            assert t >= self.num_reserved_tokens, "Cannot decode a tokenlist containing a reserved token!"
            if t < len(self.distinguished_strings) + self.num_reserved_tokens:
                result += self.distinguished_strings[t - self.num_reserved_tokens]
            else:
                result += self.unmangle_dict[t]
        return result
    def numTokens(self) -> int:
        return self.next_mangle_ord

    def getState(self) -> TokenizerState:
        return list(self.mangle_dict.items()), self.distinguished_strings, self.next_mangle_ord
    def setState(self, state : TokenizerState):
        dict_items, self.distinguished_strings, self.next_mangle_ord = state
        for k, v in dict_items:
            self.mangle_dict[k] = v
            self.unmangle_dict[v] = k


context_tokens = [
    "forall",
]
tactic_tokens = [
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

def encode_tactic(tactic : str) -> List[int]:
    return tacticTokenizer.toTokenList(tactic)
def encode_context(context : str) -> List[int]:
    return contextTokenizer.toTokenList(context)

def decode_tactic(tokenlist : List[int]) -> str:
    return tacticTokenizer.toString(tokenlist)
def decode_context(context : List[int]) -> str:
    return contextTokenizer.toString(context)

def get_encoder_state() -> Tuple[TokenizerState, TokenizerState]:
    return tacticTokenizer.getState(), contextTokenizer.getState()
def context_vocab_size() -> int:
    return contextTokenizer.numTokens()
def tactic_vocab_size() -> int:
    return tacticTokenizer.numTokens()

def set_encoder_state(state : Tuple[TokenizerState, TokenizerState]) -> None:
    tactic_state, context_state = state
    tacticTokenizer.setState(tactic_state)
    contextTokenizer.setState(context_state)

def enable_keywords() -> None:
    global contextTokenizer
    global tacticTokenizer
    contextTokenizer = Tokenizer(context_tokens, 2)
    tacticTokenizer = Tokenizer(tactic_tokens, 2)
def disable_keywords() -> None:
    global contextTokenizer
    global tacticTokenizer
    contextTokenizer = Tokenizer([], 2)
    tacticTokenizer = Tokenizer([], 2)

enable_keywords()
