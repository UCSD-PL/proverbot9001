#!/usr/bin/env python3

import re
from typing import Dict, List, Tuple, Callable, Union, Iterable, cast

class Tokenizer:
    def toTokenList(self, string : str) -> List[int]:
        assert False, "Can't use base class, must override method"
        pass
    def toString(self, tokenlist : List[int]) -> str:
        assert False, "Can't use base class, must override method"
        pass
    def freezeTokenList(self):
        pass
    def numTokens(self) -> int:
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
    keywords_and_counts = sorted(counts.items(),
                                 reverse=True,
                                 key=lambda x: x[1])[:k]
    keywords = [x[0] for x in keywords_and_counts]
    return keywords

CompleteTokenizerState = Tuple[List[str], int]

class CompleteTokenizer(Tokenizer):
    def __init__(self, keywords : List[str], num_reserved_tokens : int = 0) \
        -> None:
        self.keywords = keywords
        self.num_reserved_tokens = num_reserved_tokens
        pass
    def toTokenList(self, string : str) -> List[int]:
        words = get_words(string)
        tokens : List[int] = []
        for word in words:
            if word in self.keywords:
                tokens.append(self.num_reserved_tokens + self.keywords.index(word))
            else:
                tokens.append(self.num_reserved_tokens + len(self.keywords))
        return tokens
    def toString(self, tokenlist : List[int]) -> str:
        result = ""
        for token in tokenlist:
            assert token >= self.num_reserved_tokens and \
                token <= self.num_reserved_tokens + len(self.keywords)
            if result != "":
                result += " "
            if token == self.num_reserved_tokens + len(self.keywords):
                result += "UNKNOWN"
            else:
                result += self.keywords[token - self.num_reserved_tokens]
        return result
    def numTokens(self) -> int:
        return self.num_reserved_tokens + len(self.keywords) + 1
        pass

KeywordTokenizerState = Tuple[List[Tuple[str, int]], List[str], int]

class KeywordTokenizer(Tokenizer):
    def __init__(self, keywords : List[str], num_reserved_tokens : int = 0) \
        -> None:
        self.num_reserved_tokens = num_reserved_tokens + 1
        self.unknown_ordinal = num_reserved_tokens
        self.keywords = keywords
        self.next_mangle_ord = self.num_reserved_tokens + len(keywords)
        self.mangle_dict = {} # type: Dict[str, int]
        self.unmangle_dict = {} # type: Dict[int, str]
        self.unmangle_dict[self.unknown_ordinal] = "UNKNOWN"
        self._frozen = False
        pass

    def freezeTokenList(self):
        self._frozen = True

    def _mangle(self, string : str) -> str:
        for c in string:
            if not c in self.mangle_dict:
                if self._frozen:
                    self.mangle_dict[c] = self.unknown_ordinal
                else:
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
            assert ord(c) < self.next_mangle_ord
        tokenlist = [ord(c) for c in mangled_string]
        return tokenlist

    def toString(self, idxs : List[int]) -> str:
        result = ""
        for t in idxs:
            if t < len(self.keywords) + self.num_reserved_tokens and \
               t >= self.num_reserved_tokens:
                result += self.keywords[t - self.num_reserved_tokens]
            else:
                result += self.unmangle_dict[t]
        return result
    def numTokens(self) -> int:
        assert self._frozen,\
            "Can't get number of tokens until the tokenizer is frozen! "\
            "It still might change"
        return self.next_mangle_ord

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
TokenizerState = Union[KeywordTokenizerState, CompleteTokenizerState]

tokenizers = {
    "no-fallback" : CompleteTokenizer,
    "chars-fallback" : KeywordTokenizer,
}
