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
import itertools
import multiprocessing
import functools
import sys
import time
import io
import os
from abc import ABCMeta
from dataclasses import dataclass

from argparse import Namespace
from itertools import chain
from sparse_list import SparseList  # type: ignore
import random
import torch

from tokenizer import (Tokenizer, TokenizerState,
                       make_keyword_tokenizer_relevance,
                       make_keyword_tokenizer_topk, tokenizers)
from format import (read_tactic_tuple, ScrapedTactic, ScrapedCommand,
                    read_tuple, TacticContext, strip_scraped_output)
from models.components import SimpleEmbedding
import serapi_instance

from typing import (Tuple, NamedTuple, List, Callable, Optional,
                    Sized, Sequence, Dict, Generic, Iterable, TypeVar,
                    Any)
from util import eprint, chunks, split_by_char_outside_matching
from context_filter import get_context_filter, ContextFilter
from serapi_instance import get_stem
from pathlib_revised import Path2
TOKEN_START = 2
SOS_token = 1
EOS_token = 0

Sentence = List[int]
Bag = List[int]
ClassifySequenceDataset = List[Tuple[Sentence, int]]
SequenceSequenceDataset = List[Tuple[Sentence, Sentence]]
ClassifyBagDataset = List[Tuple[Bag, int]]
TermDataset = List[Sentence]


class Dataset(Sized, metaclass=ABCMeta):
    pass


class DatasetMetadata(metaclass=ABCMeta):
    pass


SampleType = TypeVar('SampleType')

@dataclass(init=True, repr=True)
class ListDataset(Dataset, Generic[SampleType]):
    data : List[SampleType]
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i : Any):
        return self.data[i]

@dataclass(init=True, repr=True)
class RawDataset(Dataset, Sequence[ScrapedTactic]):
    data : List[ScrapedTactic]
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i : Any):
        return self.data[i]

class EmbeddedSample(NamedTuple):
    relevant_lemmas : List[str]
    prev_tactics : List[str]
    hypotheses : List[str]
    goal : str
    tactic : int

@dataclass(init=True, repr=True)# type: ignore
class EmbeddedDataset(Dataset, Iterable[EmbeddedSample], metaclass=ABCMeta):
    pass

@dataclass(init=True, repr=True)
class StrictEmbeddedDataset(EmbeddedDataset, Sequence[EmbeddedSample]):
    data : List[EmbeddedSample]
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i : Any):
        return self.data[i]
@dataclass(init=True, repr=True)
class LazyEmbeddedDataset(EmbeddedDataset):
    data : Iterable[EmbeddedSample]
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)

class TokenizedSample(NamedTuple):
    relevant_lemmas : List[str]
    prev_tactics : List[str]
    hypotheses : List[str]
    goal : Sentence
    tactic : int

@dataclass(init=True, repr=True)
class TokenizedDataset(Dataset):
    data : Iterable[TokenizedSample]
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)


NGram = List[int]

class NGramSample(NamedTuple):
    goal : NGram
    tactic : int

@dataclass(init=True, repr=True)
class NGramDataset(Dataset):
    data : List[NGramSample]
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i : Any):
        return self.data[i]

def getTokenbagVector(goal : Sentence) -> Bag:
    tokenbag: List[int] = []
    for t in goal:
        if t >= len(tokenbag):
            tokenbag = extend(tokenbag, t+1)
        tokenbag[t] += 1
    return tokenbag

def getNGramTokenbagVector(n : int, num_tokens : int, goal : Sentence) -> Bag:
    tokenbag: SparseList[int] = SparseList(num_tokens ** n, 0)
#    tokenbag: List[int] = extend([], num_tokens ** n)
    for i in range(n-1, len(goal)):
        v_index = goal[i]
        for j in range(1, n):
            v_index *= num_tokens
            v_index += goal[i-j]
        tokenbag[v_index] += 1
#    pdb.set_trace()
    return tokenbag

def extend(vector : List[int], length : int):
    assert len(vector) <= length
    return vector + [0] * (length - len(vector))

def file_chunks(filepath : Path2, chunk_size : int):
    with filepath.open(mode='r') as f:
        while True:
            chunk = list(itertools.islice(f, chunk_size))
            if len(chunk) == 0:
                return
            yield chunk

MixedDataset = Iterable[ScrapedCommand]

def read_all_text_data_worker__(lines : List[str]) -> MixedDataset:
    def worker_generator():
        with io.StringIO("".join(lines)) as f:
            t = read_tuple(f)
            while t:
                yield t
                t = read_tuple(f)
    return list(worker_generator())
def read_all_text_data(data_path : Path2) -> MixedDataset:
    line_chunks = file_chunks(data_path, 32768)
    data_chunks = lazy_multiprocessing_imap(read_all_text_data_worker__, line_chunks)
    yield from itertools.chain.from_iterable(data_chunks)
def read_text_data_worker__(lines : List[str]) -> RawDataset:
    def worker_generator() -> Iterable[ScrapedTactic]:
        with io.StringIO("".join(lines)) as f:
            t = read_tactic_tuple(f)
            while t:
                yield t
                t = read_tactic_tuple(f)
    return RawDataset(list(worker_generator()))

T = TypeVar('T')
O = TypeVar('O')

def lazy_multiprocessing_imap(worker: Callable[[T], O], in_data : Iterable[T],
                              num_threads : int=os.cpu_count(),
                              chunk_size : Optional[int]=None) -> Iterable[O]:
    if chunk_size == None:
        chunk_size = num_threads * 10
    with multiprocessing.Pool(num_threads) as pool:
        for chunk in chunks(in_data, chunk_size):
            yield from list(pool.imap(worker, chunk))

def read_text_data(data_path: Path2) \
                  -> Iterable[ScrapedTactic]:
    line_chunks = file_chunks(data_path, 32768)
    data_chunks = lazy_multiprocessing_imap(read_text_data_worker__, line_chunks)
    yield from itertools.chain.from_iterable(data_chunks)

@dataclass
class StateScore:
    state : TacticContext
    score : float

def preprocess_data(arg_values: Namespace, dataset_iter:
                    Iterable[ScrapedTactic]) \
                    -> Iterable[ScrapedTactic]:
    with multiprocessing.Pool(arg_values.num_threads) as pool:
        if arg_values.truncate_semicolons:
            dataset_iter = pool.imap(truncate_tactic_semicolons,
                                     dataset_iter)
        if arg_values.use_substitutions:
            substitutions = {"auto": "eauto.",
                             "intros until": "intros.",
                             "intro": "intros.",
                             "constructor": "econstructor."}
            dataset_iter = pool.imap(
                functools.partial(tactic_substitutions, substitutions),
                dataset_iter)
        dataset_iter = pool.imap(
            serapi_instance.normalizeNumericArgs, dataset_iter)
        yield from dataset_iter


def get_text_data(arg_values: Namespace) -> RawDataset:
    def _print(*args, **kwargs):
        eprint(*args, **kwargs, guard=arg_values.verbose)

    start = time.time()
    _print("Reading dataset...", end="")
    sys.stdout.flush()
    raw_data = read_text_data(arg_values.scrape_file)
    filtered_data = RawDataset(list(
        itertools.islice(
            filter_data(preprocess_data(arg_values, raw_data),
                        get_context_filter(arg_values.
                                           context_filter),
                        arg_values),
            arg_values.max_tuples)))
    _print("{:.2f}s".format(time.time() - start))
    _print("Got {} input-output pairs ".format(len(filtered_data)))
    return filtered_data

class StateEvaluationDataset(ListDataset[StateScore]):
    pass

def filter_data(data: RawDataset, pair_filter: ContextFilter,
                arg_values: Namespace) -> Iterable[ScrapedTactic]:
    return (scraped
            for (scraped, next_scraped) in
            zip(data, itertools.chain(itertools.islice(data, 1, None),
                                      [([], [], [], "", "")]))
            if pair_filter(strip_scraped_output(scraped), scraped.tactic,
                           strip_scraped_output(next_scraped), arg_values))

def encode_seq_seq_data(data : RawDataset,
                        context_tokenizer_type : Callable[[List[str], int], Tokenizer],
                        tactic_tokenizer_type : Callable[[List[str], int], Tokenizer],
                        num_keywords : int,
                        num_reserved_tokens : int) \
    -> Tuple[SequenceSequenceDataset, Tokenizer, Tokenizer]:
    context_tokenizer = make_keyword_tokenizer_topk([context for
                                                     prev_tactics, hyps, context, tactic
                                                     in data],
                                                    context_tokenizer_type,
                                                    num_keywords, num_reserved_tokens)
    tactic_tokenizer = make_keyword_tokenizer_topk([tactic for prev_tactics, hyps,
                                                    context, tactic in data],
                                                   tactic_tokenizer_type,
                                                   num_keywords, num_reserved_tokens)
    result = [(context_tokenizer.toTokenList(context),
               tactic_tokenizer.toTokenList(tactic))
              for prev_tactics, hyps, context, tactic in data]
    context_tokenizer.freezeTokenList()
    tactic_tokenizer.freezeTokenList()
    return result, context_tokenizer, tactic_tokenizer

def _tokenize(t : Tokenizer, s : str):
    return t.toTokenList(s)


def tokenize_data(tokenizer : Tokenizer, data : EmbeddedDataset,
                  num_threads : int) \
    -> TokenizedDataset:
    with multiprocessing.Pool(num_threads) as pool:
        result=TokenizedDataset(list(chain.from_iterable(pool.imap(
            functools.partial(tokenize_worker__, tokenizer),
            chunks(list(data), 1024)))))
    tokenizer.freezeTokenList()
    return result

def tokenize_worker__(tokenizer : Tokenizer,
                      chunk : EmbeddedDataset) -> TokenizedDataset:
    return TokenizedDataset([TokenizedSample(relevant_lemmas,
                                             prev_tactics,
                                             hypotheses,
                                             tokenizer.toTokenList(goal),
                                             tactic)
                             for relevant_lemmas, prev_tactics, hypotheses, goal, tactic
                             in chunk])

def encode_seq_classify_data(data : RawDataset,
                             tokenizer_type : Callable[[List[str], int], Tokenizer],
                             num_keywords : int,
                             num_reserved_tokens : int,
                             save_tokens : Optional[str] = None,
                             load_tokens : Optional[str] = None,
                             num_relevance_samples : int = 1000) \
    -> Tuple[ClassifySequenceDataset, Tokenizer, SimpleEmbedding]:
    embedding = SimpleEmbedding()
    subset = RawDataset(random.sample(data, num_relevance_samples))
    if load_tokens:
        print("Loading tokens from {}".format(load_tokens))
        tokenizer = torch.load(load_tokens)
    else:
        start = time.time()
        print("Picking tokens...", end="")
        sys.stdout.flush()
        tokenizer = make_keyword_tokenizer_relevance([(context,
                                                       embedding.encode_token(
                                                           get_stem(tactic)))
                                                      for prev_tactics, hyps,
                                                      context, tactic
                                                      in subset],
                                                     tokenizer_type,
                                                     num_keywords, num_reserved_tokens)
        print("{}s".format(time.time() - start))
    if save_tokens:
        print("Saving tokens to {}".format(save_tokens))
        torch.save(tokenizer, save_tokens)
    with multiprocessing.Pool(None) as pool:
        result = [(goal, embedding.encode_token(tactic)) for goal, tactic in
                  chain.from_iterable(pool.imap(functools.partial(
                      encode_seq_classify_data_worker__, tokenizer),
                                                          chunks(data, 1024)))]
    tokenizer.freezeTokenList()
    return result, tokenizer, embedding
def encode_seq_classify_data_worker__(tokenizer : Tokenizer,
                                      chunk : List[Tuple[List[str], List[str], str, str]])\
    -> List[Tuple[Sentence, str]]:
    return [(tokenizer.toTokenList(goal), get_stem(tactic))
            for prev_tactics, hyps, goal, tactic in chunk]


def encode_bag_classify_data(data : RawDataset,
                             tokenizer_type : Callable[[List[str], int], Tokenizer],
                             num_keywords : int,
                             num_reserved_tokens : int) \
    -> Tuple[ClassifyBagDataset, Tokenizer, SimpleEmbedding]:
    seq_data, tokenizer, embedding = encode_seq_classify_data(data, tokenizer_type,
                                                              num_keywords,
                                                              num_reserved_tokens)
    bag_data = [(extend(getTokenbagVector(context), tokenizer.numTokens()), tactic)
                for context, tactic in seq_data]
    return bag_data, tokenizer, embedding

def encode_bag_classify_input(context : str, tokenizer : Tokenizer ) \
    -> Bag:
    return extend(getTokenbagVector(tokenizer.toTokenList(context)), tokenizer.numTokens())

def encode_ngram_classify_data(data : RawDataset,
                               num_grams : int,
                               tokenizer_type : Callable[[List[str], int], Tokenizer],
                               num_keywords : int,
                               num_reserved_tokens : int) \
                               -> Tuple[ClassifyBagDataset, Tokenizer, SimpleEmbedding]:
    seq_data, tokenizer, embedding = encode_seq_classify_data(data, tokenizer_type,
                                                              num_keywords,
                                                              num_reserved_tokens)
    print("Getting grams")
    inputs, outputs = zip(*seq_data)
    with multiprocessing.Pool(None) as pool:
        bag_data = list(pool.imap(functools.partial(
            getNGramTokenbagVector, num_grams, tokenizer.numTokens()), inputs))
    return list(zip(bag_data, outputs)), tokenizer, embedding

def encode_ngram_classify_input(context : str, num_grams : int, tokenizer : Tokenizer ) \
    -> Bag:
    return getNGramTokenbagVector(num_grams, tokenizer.numTokens(), tokenizer.toTokenList(context))

def term_data(data : RawDataset,
              tokenizer_type : Callable[[List[str], int], Tokenizer],
              num_keywords : int,
              num_reserved_tokens : int) -> Tuple[TermDataset, Tokenizer]:
    term_strings = list(itertools.chain.from_iterable(
        [[hyp.split(":")[1].strip() for hyp in hyps] + [goal]
         for prev_tactics, hyps, goal, tactic in data]))
    tokenizer = make_keyword_tokenizer_topk(term_strings, tokenizer_type,
                                            num_keywords, num_reserved_tokens)
    return [tokenizer.toTokenList(term_string) for term_string in term_strings], \
        tokenizer

def normalizeSentenceLength(sentence : Sentence, max_length : int) -> Sentence:
    if len(sentence) > max_length:
        sentence = sentence[:max_length]
    elif len(sentence) < max_length:
        sentence.extend([EOS_token] * (max_length - len(sentence)))
    return sentence

def stemmify_data(point : ScrapedTactic) -> ScrapedTactic:
    relevant_lemmas, prev_tactics, hypotheses, goal, tactic = point
    return ScrapedTactic(relevant_lemmas, prev_tactics, hypotheses, goal, get_stem(tactic))

def tactic_substitutions(substitutions : Dict[str, str], sample : ScrapedTactic) \
    -> ScrapedTactic:
    relevant_lemmas, prev_tactics, hyps, goal, tactic = sample
    return ScrapedTactic(relevant_lemmas, prev_tactics, hyps, goal,
                         tactic if get_stem(tactic) not in substitutions
                         else substitutions[get_stem(tactic)])


def truncate_tactic_semicolons(sample: ScrapedTactic) \
        -> ScrapedTactic:
    rl, pt, hyp, goal, tactic = sample
    newtac = tactic
    outer_parens_match = re.fullmatch("\((.*)\)", newtac.strip())
    if outer_parens_match:
        newtac = outer_parens_match.group(1)
    splitresult = split_by_char_outside_matching(
        "\(|\[", "\)|\]", ";", newtac)
    if splitresult:
        before_semi, after_semi = splitresult
        newtac = before_semi.strip() + "."
    return ScrapedTactic(rl, pt, hyp, goal, newtac)
