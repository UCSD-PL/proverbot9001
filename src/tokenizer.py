#!/usr/bin/env python3

import re
import math
from typing import Dict, List, Tuple, Callable, Union, Iterable, cast, Set, Any, Counter
from abc import ABCMeta, abstractmethod
import collections
import multiprocessing
from util import *

class Tokenizer(metaclass=ABCMeta):
    @abstractmethod
    def toTokenList(self, string : str) -> List[int]:
        pass
    @abstractmethod
    def toString(self, tokenlist : List[int]) -> str:
        pass
    def freezeTokenList(self):
        pass
    @abstractmethod
    def numTokens(self) -> int:
        pass

def get_words(string : str) -> List[str]:
    return [word for word in re.sub('(,|\.+|:|\)|\()', r' \1 ', string).split()
            if word.strip() != '']

def get_symbols(string : str) -> List[str]:
    return [word for word in re.sub('(,|:|\)|\()', r' \1 ', string).split()
            if word.strip() != '']

def get_topk_keywords_worker__(sentence_list : List[str]) -> collections.Counter:
    counts : Counter[str] = collections.Counter()
    for example in sentence_list:
        counts.update(get_words(example))
    return counts

def get_topk_keywords(exampleSentences : Iterable[str], k : int) -> List[str]:
    with multiprocessing.Pool(None) as pool:
        sub_counts = pool.imap_unordered(get_topk_keywords_worker__,
                                         chunks(exampleSentences, 32768))
        counts : Counter[str] = collections.Counter()
        for sub_count in sub_counts:
            counts.update(sub_count)
    return [word for word, count in counts.most_common(k)]

def get_relevant_k_keywords(examplePairs : Iterable[Tuple[str, int]], k : int) \
    -> List[str]:
    words : Set[str] = set()
    for input, output in examplePairs:
        words = words | set(get_words(input))

    words_and_entropies = sorted([(word, word_partitioned_entropy(examplePairs, word)) for
                                  word in words],
                                 reverse=True,
                                 key=lambda x: x[1])[:k]
    tokens = [x[0] for x in words_and_entropies]
    print("Highest information tokens are {}".format(tokens))
    return tokens

def word_partitioned_entropy(examplePairs : Iterable[Tuple[str, int]], word : str) \
    -> float:
    return ((entropy([output for input, output in examplePairs
                      if word in get_words(input)]) +
             entropy([output for input, output in examplePairs
                      if word in get_words(input)])) / 2)

def entropy(outputs : List[int]) -> float:
    output_counts : Dict[int, int] = {}
    total_count = 0
    for output in outputs:
        total_count += 1
        if output in output_counts:
            output_counts[output] += 1
        else:
            output_counts[output] = 1

    entropy = 0.
    for output, count in output_counts.items():
        probability = count / total_count
        entropy += probability * math.log(probability, 2)
    return (- entropy)

CompleteTokenizerState = Tuple[List[str], int]

class CharsTokenizer(Tokenizer):
    def __init__(self, keywords : List[str], num_reserved_tokens : int = 0) -> None:
        self.unknown_ord = num_reserved_tokens
        self.next_ord = num_reserved_tokens + 1
        self.mangle_dict = {} # type: Dict[str, int]
        self.unmangle_dict = {} # type: Dict[int, str]
        self._frozen = False
    def freezeTokenList(self):
        self._frozen = True
    def toTokenList(self, string : str) -> List[int]:
        for c in string:
            if not c in self.mangle_dict:
                if self._frozen:
                    self.mangle_dict[c] = self.unknown_ord
                else:
                    self.mangle_dict[c] = self.next_ord
                    self.unmangle_dict[self.next_ord] = c
                    self.next_ord += 1
        return [self.mangle_dict[c] for c in string]
    def toString(self, tokenlist : List[int]) -> str:
        return "".join([self.unmangle_dict[t] for t in tokenlist])
    def numTokens(self) -> int:
        return self.next_ord

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

def make_keyword_tokenizer(data : List[str],
                           tokenizer_type : Callable[[List[str], int], Tokenizer],
                           num_keywords : int,
                           num_reserved_tokens : int) -> Tokenizer:
    keywords = get_topk_keywords(data, num_keywords)
    tokenizer = tokenizer_type(keywords, num_reserved_tokens)
    return tokenizer


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
    "chars-only" : CharsTokenizer,
}
