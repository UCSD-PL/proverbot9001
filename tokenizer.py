#!/usr/bin/env python3

import re
from typing import Dict, List, Tuple, Callable, Union, Iterable

class Tokenizer:
    def toTokenList(self, string : str) -> List[int]:
        assert False, "Can't use base class, must override method"
        pass
    def toString(self, tokenlist : List[int]) -> str:
        assert False, "Can't use base class, must override method"
        pass
    def numTokens(self) -> int:
        assert False, "Can't use base class, must override method"
        pass
    def getState(self) -> 'TokenizerState':
        assert False, "Can't use base class, must override method"
        pass
    def setState(self, state : 'TokenizerState') -> None:
        assert False, "Can't use base class, must override method"
        pass

def get_words(string : str) -> List[str]:
    return re.split(r'\W|\.+|:', string)

def get_topk_keywords(exampleSentences : Iterable[str], k : int) -> List[str]:
    counts = {} # type: Dict[str, int]
    for example in exampleSentences:
        for token in get_words(example):
            if token not in counts:
                counts[token] = 1
            else:
                counts[token] += 1
    keywords = [x[0] for x in sorted(counts.items(),
                                     key=lambda x: x[1])[:k]]
    return keywords

KeywordTokenizerState = Tuple[List[Tuple[str, int]], List[str], int]

class KeywordTokenizer(Tokenizer):
    def __init__(self, keywords : List[str], num_reserved_tokens : int = 0) \
        -> None:
        self.num_reserved_tokens = num_reserved_tokens
        self.keywords = keywords
        self.next_mangle_ord = num_reserved_tokens + len(keywords)
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

        for idx, token_string in enumerate(self.keywords,
                                           start=self.num_reserved_tokens):
            mangled_string = mangled_string.replace(self._mangle(token_string), chr(idx))

        for c in mangled_string:
            assert ord(c) < self.numTokens()
        tokenlist = [ord(c) for c in mangled_string]
        assert self.toString(tokenlist) == string
        return tokenlist

    def toString(self, idxs : List[int]) -> str:
        result = ""
        for t in idxs:
            assert t >= self.num_reserved_tokens, "Cannot decode a tokenlist containing a reserved token!"
            if t < len(self.keywords) + self.num_reserved_tokens:
                result += self.keywords[t - self.num_reserved_tokens]
            else:
                result += self.unmangle_dict[t]
        return result
    def numTokens(self) -> int:
        return self.next_mangle_ord

    def getState(self) -> KeywordTokenizerState:
        return list(self.mangle_dict.items()), self.keywords, self.next_mangle_ord
    def setState(self, state : KeywordTokenizerState):
        dict_items, self.keywords, self.next_mangle_ord = state
        for k, v in dict_items:
            self.mangle_dict[k] = v
            self.unmangle_dict[v] = k


context_keywords = [
    "forall",
    "eq",
    "Some",
    "None",
    "if",
    "then",
    "else",
]
tactic_keywords = [
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

contextTokenizer : KeywordTokenizer
tacticTokenizer : KeywordTokenizer

def tokenize_tactic(tactic : str) -> List[int]:
    return tacticTokenizer.toTokenList(tactic)
def tokenize_context(context : str) -> List[int]:
    return contextTokenizer.toTokenList(context)

def untokenize_tactic(tokenlist : List[int]) -> str:
    return tacticTokenizer.toString(tokenlist)
def untokenize_context(context : List[int]) -> str:
    return contextTokenizer.toString(context)

TokenizerState = Union[KeywordTokenizerState]

def get_tokenizer_state() -> Tuple[TokenizerState, TokenizerState]:
    return tacticTokenizer.getState(), contextTokenizer.getState()
def context_vocab_size() -> int:
    return contextTokenizer.numTokens()
def tactic_vocab_size() -> int:
    return tacticTokenizer.numTokens()

def set_tokenizer_state(state : Tuple[TokenizerState, TokenizerState]) -> None:
    tactic_state, context_state = state
    tacticTokenizer.setState(tactic_state)
    contextTokenizer.setState(context_state)

def enable_keywords() -> None:
    global contextTokenizer
    global tacticTokenizer
    contextTokenizer = KeywordTokenizer(context_keywords, 2)
    tacticTokenizer = KeywordTokenizer(tactic_keywords, 2)
def disable_keywords() -> None:
    global contextTokenizer
    global tacticTokenizer
    contextTokenizer = KeywordTokenizer([], 2)
    tacticTokenizer = KeywordTokenizer([], 2)

enable_keywords()
