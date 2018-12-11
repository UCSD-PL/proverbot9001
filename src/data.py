#!/usr/bin/env python3

import pdb
import re
import itertools
import multiprocessing
import functools
from itertools import chain
from sparse_list import SparseList # type: ignore
import random
from tokenizer import Tokenizer, TokenizerState, \
    make_keyword_tokenizer_relevance, make_keyword_tokenizer_topk
from format import read_tuple
from models.components import SimpleEmbedding

from typing import Tuple, List, Callable, Optional
from util import *
from context_filter import ContextFilter, get_context_filter
from serapi_instance import get_stem

TOKEN_START = 2
SOS_token = 1
EOS_token = 0

Sentence = List[int]
Bag = List[int]
RawDataset = Iterable[Tuple[List[str], str, str]]
ClassifySequenceDataset = List[Tuple[Sentence, int]]
SequenceSequenceDataset = List[Tuple[Sentence, Sentence]]
ClassifyBagDataset = List[Tuple[Bag, int]]
TermDataset = List[Sentence]

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

def file_chunks(filepath : str, chunk_size : int):
    with open(filepath, 'r') as f:
        while True:
            chunk = list(itertools.islice(f, chunk_size))
            if len(chunk) == chunk_size:
                while chunk[-1] != "-----\n":
                    nextline = f.readline()
                    if not nextline:
                        break
                    chunk += [nextline]
                    assert len(chunk) < chunk_size * 2
            elif len(chunk) == 0:
                return
            yield chunk

def read_text_data_worker__(lines : List[str]) -> RawDataset:
    def worker_generator():
        with io.StringIO("".join(lines)) as f:
            t = read_tuple(f)
            while t:
                yield t
                t = read_tuple(f)
    return list(worker_generator())

def read_text_data(data_path : str) -> RawDataset:
    with multiprocessing.Pool(None) as pool:
        line_chunks = file_chunks(data_path, 32768)
        data_chunks = pool.imap_unordered(read_text_data_worker__, line_chunks)
        result = itertools.chain.from_iterable(data_chunks)
        yield from result
def get_text_data(data_path : str, context_filter_name : str,
                  max_tuples : Optional[int]=None, verbose : bool = False) -> RawDataset:
    def _print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    _print("Reading dataset...")
    raw_data = read_text_data(data_path)
    filtered_data = list(itertools.islice(filter_data(raw_data, get_context_filter(context_filter_name)), max_tuples))
    _print("Got {} input-output pairs ".format(len(filtered_data)))
    return filtered_data

def filter_data(data : RawDataset, pair_filter : ContextFilter) -> RawDataset:
    return ((hyps, goal, tactic)
            for ((hyps, goal, tactic), (next_hyps, next_goal, next_tactic)) in
            zip(data, itertools.islice(data, 1, None))
            if pair_filter({"goal": goal, "hyps" : hyps}, tactic,
                           {"goal": next_goal, "hyps" : next_hyps}))

def encode_seq_seq_data(data : RawDataset,
                        context_tokenizer_type : Callable[[List[str], int], Tokenizer],
                        tactic_tokenizer_type : Callable[[List[str], int], Tokenizer],
                        num_keywords : int,
                        num_reserved_tokens : int) \
    -> Tuple[SequenceSequenceDataset, Tokenizer, Tokenizer]:
    context_tokenizer = make_keyword_tokenizer_topk([context for hyps, context, tactic in data],
                                                    context_tokenizer_type,
                                                    num_keywords, num_reserved_tokens)
    tactic_tokenizer = make_keyword_tokenizer_topk([tactic for hyps, context, tactic in data],
                                                   tactic_tokenizer_type,
                                                   num_keywords, num_reserved_tokens)
    result = [(context_tokenizer.toTokenList(context),
               tactic_tokenizer.toTokenList(tactic))
              for hyps, context, tactic in data]
    context_tokenizer.freezeTokenList()
    tactic_tokenizer.freezeTokenList()
    return result, context_tokenizer, tactic_tokenizer

def _tokenize(t : Tokenizer, s : str):
    return t.toTokenList(s)

def encode_seq_classify_data_worker__(tokenizer : Tokenizer,
                                      chunk : List[Tuple[List[str], str, str]])\
    -> List[Tuple[Sentence, str]]:
    return [(tokenizer.toTokenList(goal), get_stem(tactic))
            for hyps, goal, tactic in chunk]

def encode_seq_classify_data(data : RawDataset,
                             tokenizer_type : Callable[[List[str], int], Tokenizer],
                             num_keywords : int,
                             num_reserved_tokens : int) \
    -> Tuple[ClassifySequenceDataset, Tokenizer, SimpleEmbedding]:
    embedding = SimpleEmbedding()
    print("Making tokenizer...")
    subset : RawDataset = random.sample(list(data), 1000)
    tokenizer = make_keyword_tokenizer_relevance([(context,
                                                   embedding.encode_token(
                                                       get_stem(tactic)))
                                                  for hyps, context, tactic
                                                  in subset],
                                                 tokenizer_type,
                                                 num_keywords, num_reserved_tokens)
    print("Tokenizing/embedding data...")
    with multiprocessing.Pool(None) as pool:
        result = [(goal, embedding.encode_token(tactic)) for goal, tactic in
                  chain.from_iterable(pool.imap_unordered(functools.partial(
                      encode_seq_classify_data_worker__, tokenizer),
                                                          chunks(data, 1024)))]
    tokenizer.freezeTokenList()
    return result, tokenizer, embedding

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
        bag_data = list(pool.imap_unordered(functools.partial(
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
         for hyps, goal, tactic in data]))
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
