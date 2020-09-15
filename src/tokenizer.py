#!/usr/bin/env python3.7
##########################################################################
#
#    This file is part of Proverbot9001.
#
#    Proverbot9001 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Proverbot9001 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Proverbot9001.  If not, see <https://www.gnu.org/licenses/>.
#
#    Copyright 2019 Alex Sanchez-Stern and Yousef Alhessi
#
##########################################################################

import re
import math
import collections
import multiprocessing
import functools
from typing import Dict, List, Tuple, Callable, Union, Iterable, cast, \
    Set, Any, Counter, Sequence, Optional
from abc import ABCMeta, abstractmethod

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
    def numTokens(self) -> int:
        return len(self.listTokens())
    @abstractmethod
    def listTokens(self) -> List[str]:
        pass

def get_words(string : str) -> List[str]:
    return [word for word in
            re.sub('(,|\.+|(?::(?!=))|(?::=)|\)|\()', r' \1 ', string).split()
            if word.strip() != '']

def get_symbols(string: str) -> List[str]:
    return [word for word in re.sub('(,|(?::(?!=))|(?::=)|\)|\(|;)',
                                    r' \1 ', string).split()
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

def get_relevant_k_keywords_worker__(examplePairs : List[Tuple[str, int]],
                                     word : str):
    return (word, word_partitioned_entropy(examplePairs, word))

def get_relevant_k_keywords(examplePairs : Iterable[Tuple[str, int]], k : int) \
    -> List[str]:
    words : Set[str] = set()
    for input, output in examplePairs:
        words = words | set(get_words(input))

    with multiprocessing.Pool(None) as pool:
        words_and_entropies = sorted(list(
            pool.imap_unordered(functools.partial(get_relevant_k_keywords_worker__,
                                                  examplePairs),
                                words)),
                                    reverse=False,
                                    key=lambda x: x[1])[:k]

    tokens = [x[0] for x in words_and_entropies]
    return tokens

def get_relevant_k_keywords2(examplePairs : Iterable[Tuple[str, int]], k : int,
                             num_threads : Optional[int]) \
    -> List[str]:
    def leader_entropy(pool : List[Tuple[str, int]]) -> Tuple[int, float]:
        if len(pool) == 0:
            return 0, 0
        tactic_counter : Counter[int] = collections.Counter()
        for context, tactic in pool:
            tactic_counter[tactic] += 1
        leader_tactic, leader_count = tactic_counter.most_common(1)[0]
        return leader_tactic, entropy([1 if tactic == leader_tactic else 0
                                       for context, tactic in pool])

    # Given a pools list, split each pool into two pools based on the
    # presence of the word 'word' in the samples, dropping pools with
    # no entropy (only one tactic).
    def split_pools(pools : List[Tuple[List[Tuple[str, int]], int, float]], word : str):
        new_pools : List[Tuple[List[Tuple[str, int]], int, float]] = []
        for old_pool, old_leader, old_entropy in pools:
            subpools = \
                multipartition(old_pool,
                               lambda ctxt_and_tactic:
                               1 if word in get_words(ctxt_and_tactic[0])
                               else 0)
            for subpool in subpools:
                leader, entropy = leader_entropy(subpool)
                if entropy > 0:
                    new_pools.append((subpool, leader, entropy))
        return new_pools

    pairs_list = list(examplePairs)

    # Get a starting set of "potential" tokens from the k^2 most common words
    words_counter : Counter[str] = collections.Counter()
    for context, tactic in examplePairs:
        words_counter.update(get_words(context))
    common_words = [word for word, count in words_counter.most_common(k**2)]

    # Set up the initial pool
    total_leader, total_leader_entropy = leader_entropy(pairs_list)
    pools : List[Tuple[List[Tuple[str, int]], int, float]] \
        = [(pairs_list, total_leader, total_leader_entropy)]
    keywords : List[str] = []

    common_keywords_and_counts = words_counter.most_common(int(k / 4))
    for word, count in common_keywords_and_counts:
        common_words.remove(word)
        keywords.append(word)

    while len(keywords) < k:
        if len(pools) == 0:
            print("Returning early with {} keywords: "
                  "ran out of  pools".format(len(keywords)))
            return keywords
        highest_entropy_pool, leader, pool_entropy = \
            max(pools, key=lambda pool_pair: pool_pair[-1])

        with multiprocessing.Pool(num_threads) as process_pool:
            word_entropy_pairs = list(
                process_pool.imap_unordered(
                    functools.partial(
                        get_relevant_k_keywords_worker__,
                        [(context, 1
                         if tactic == leader
                         else 0)
                         for context, tactic in highest_entropy_pool]),
                    common_words))
            word, word_partitioned_entropy = min(word_entropy_pairs,
                                                 key=lambda x: x[1])
            if word_partitioned_entropy >= pool_entropy:
                pools.remove((highest_entropy_pool, leader, pool_entropy))
                continue
        if word in keywords:
            print("Returning early with {} keywords: "
                  "ran out of samples that could be differentiated "
                  "with the presence of keywords in {} most common"
                  .format(len(keywords), k**2))
            return keywords
        keywords.append(word)
        pools = split_pools(pools, word)
    return keywords

def word_partitioned_entropy(examplePairs : Sequence[Tuple[str, int]], word : str) \
    -> float:
    has_word = [output for input, output in examplePairs if word in get_words(input)]
    entropy1 = entropy(has_word)
    not_has_word = [output for input, output in examplePairs
                    if word not in get_words(input)]
    entropy2 = entropy(not_has_word)
    scaled_entropy1 = entropy1 * len(has_word)
    scaled_entropy2 = entropy2 * len(not_has_word)
    answer = (scaled_entropy1 + scaled_entropy2) / len(examplePairs)
    assert answer <= 1
    return answer

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
    def listTokens(self) -> List[str]:
        return list(self.mangle_dict.keys())

class CompleteTokenizer(Tokenizer):
    def __init__(self, keywords : List[str], num_reserved_tokens : int = 0,
                 use_unknowns : bool = True) -> None:
        self.keywords = keywords
        self.num_reserved_tokens = num_reserved_tokens
        self.use_unknowns = use_unknowns
        pass
    def toTokenList(self, string : str) -> List[int]:
        string = unescape_periods(string)
        words = get_words(string)
        tokens : List[int] = []
        for word in words:
            if word in self.keywords:
                tokens.append(self.num_reserved_tokens + self.keywords.index(word))
            elif self.use_unknowns:
                tokens.append(self.num_reserved_tokens + len(self.keywords))
        return tokens
    def toString(self, tokenlist : List[int]) -> str:
        result = ""
        for token in tokenlist:
            assert token <= self.num_reserved_tokens + len(self.keywords)
            if result != "":
                result += " "
            if token == self.num_reserved_tokens + len(self.keywords):
                result += "UNKNOWN"
            else:
                if token < self.num_reserved_tokens:
                    result += "RES"
                else:
                    result += self.keywords[token - self.num_reserved_tokens]
        return result
    def numTokens(self) -> int:
        return self.num_reserved_tokens + len(self.keywords) + 1
    def listTokens(self) -> List[str]:
        return self.keywords

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

    def listTokens(self) -> List[str]:
        return self.keywords


def unescape_periods(s: str) -> str:
    return s.replace("\\.", ".")


def make_keyword_tokenizer_relevance(data : List[Tuple[str, int]],
                                     tokenizer_type : Callable[[List[str], int],
                                                               Tokenizer],
                                     num_keywords : int,
                                     num_reserved_tokens : int,
                                     num_threads : Optional[int]=None) -> Tokenizer:
    keywords = get_relevant_k_keywords2(data, num_keywords, num_threads)
    tokenizer = tokenizer_type(keywords, num_reserved_tokens)
    return tokenizer

def make_keyword_tokenizer_topk(data : List[str],
                                tokenizer_type : Callable[[List[str], int], Tokenizer],
                                num_keywords : int,
                                num_reserved_tokens : int) -> Tokenizer:
    keywords = get_topk_keywords(data, num_keywords)
    tokenizer = tokenizer_type(keywords, num_reserved_tokens)
    return tokenizer

def limitNumTokens(term : str, num_tokens : int):
    return ' '.join(get_symbols(term)[:num_tokens])

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
    "no-unknowns" : lambda *args, **kwargs: \
    CompleteTokenizer(*args, **kwargs, use_unknowns=False), # type: ignore
    "chars-fallback" : KeywordTokenizer,
    "chars-only" : CharsTokenizer,
} # type: Dict[str, Callable[[List[str], int], Tokenizer]]
