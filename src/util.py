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

import time
import io
import math
import re
import itertools
import argparse
import fcntl

from typing import (List, Tuple, Iterable, Any, overload, TypeVar,
                    Callable, Optional, Pattern, Match, Union)

import torch
import torch.cuda
import torch.autograd as autograd

from sexpdata import Symbol
from dataloader import rust_parse_sexp_one_level
from pathlib import Path


def maybe_cuda(component):
    if use_cuda:
        return component.to(device=torch.device(cuda_device))
    else:
        return component

def LongTensor(*args : Any) -> torch.LongTensor:
    if use_cuda:
        return torch.cuda.LongTensor(*args)
    else:
        return torch.LongTensor(*args)

def FloatTensor(*args : Any) -> torch.FloatTensor:
    if use_cuda:
        return torch.cuda.FloatTensor(*args)
    else:
        return torch.FloatTensor(*args)

def ByteTensor(*args : Any) -> torch.ByteTensor:
    if use_cuda:
        return torch.cuda.ByteTensor(*args)
    else:
        return torch.ByteTensor(*args)

def asMinutes(s : float) -> str:
    m = math.floor(s / 60)
    s -= m * 60
    return "{:3}m {:5.2f}s".format(m, s)

def stringified_percent(total : float, outof : float) -> str:
    if outof == 0:
        return "NaN"
    else:
        return "{:10.2f}".format(total * 100 / outof)

def timeSince(since : float, percent : float) -> str:
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return "{} (- {})".format(asMinutes(s), asMinutes(rs))

def str_1d_long_tensor(tensor : torch.LongTensor):
    if (type(tensor) == autograd.Variable):
        tensor = tensor.data
    tensor = tensor.view(-1)
    return str(list(tensor))

def str_1d_float_tensor(tensor : torch.FloatTensor):
    if (type(tensor) == autograd.Variable):
        tensor = tensor.data
    tensor = tensor.view(-1)
    output = io.StringIO()
    print("[", end="", file=output)
    if tensor.size()[0] > 0:
        print("{:.4f}".format(tensor[0]), end="", file=output)
    for f in tensor[1:]:
        print(", {:.4f}".format(f), end="", file=output)
    print("]", end="", file=output)
    result = output.getvalue()
    output.close()
    return result

@overload
def _inflate(tensor : torch.LongTensor, times : int) -> torch.LongTensor: ...
@overload
def _inflate(tensor : torch.FloatTensor, times : int) -> torch.FloatTensor: ...

def _inflate(tensor : torch.Tensor, times : int) -> torch.Tensor:
    tensor_dim = len(tensor.size())
    if tensor_dim == 3:
        b = tensor.size(1)
        return tensor.repeat(1, 1, times).view(tensor.size(0), b * times, -1)
    elif tensor_dim == 2:
        return tensor.repeat(1, times)
    elif tensor_dim == 1:
        b = tensor.size(0)
        return tensor.repeat(times).view(b, -1)
    else:
        raise ValueError("Tensor can be of 1D, 2D, or 3D only. "
                         "This one is {}D.".format(tensor_dim))

T = TypeVar('T')
def chunks(l : Iterable[T], chunk_size : int) -> Iterable[List[T]]:
    i = iter(l)
    next_chunk = list(itertools.islice(i, chunk_size))
    while next_chunk:
        yield next_chunk
        next_chunk = list(itertools.islice(i, chunk_size))

def list_topk(lst : List[T], k : int, f : Optional[Callable[[T], float]] = None) \
    -> Tuple[List[int], List[T]]:
    if f == None:
        f = lambda x: float(x) # type: ignore
    l = sorted(enumerate(lst), key=lambda x:f(x[1]), reverse=True) # type: ignore
    lk = l[:k]
    return tuple(zip(*lk)) # type: ignore

def topk_with_filter(t : torch.FloatTensor, k : int, f : Callable[[float, int], bool]) \
    -> Tuple[torch.FloatTensor, torch.LongTensor]:
    all_certainties, all_idxs = t.topk(t.size()[0])
    certainties = []
    idxs = []
    for certainty, idx in zip(all_certainties, all_idxs):
        if f(certainty.item(), idx.item()):
            certainties.append(certainty)
            idxs.append(idx)
            if len(certainties) == k:
                break
    return FloatTensor(certainties), LongTensor(idxs)

def multipartition(xs : List[T], f : Callable[[T], int]) -> List[List[T]]:
    result : List[List[T]] = []
    for x in xs:
        assert x != None
        i = f(x)
        while i >= len(result):
            result += [[]]
        result[i] += [x]
    return result

def escape_filename(filename : str) -> str:
    return re.sub("/", "Zs", re.sub("\.", "Zd", re.sub("Z", "ZZ", filename)))
def escape_lemma_name(lemma_name : str) -> str:
    subs = [("Z", "ZZ"),
            ("/", "Zs"),
            ("\.", "Zd")]
    for k, v in subs:
        lemma_name = re.sub(k, v, lemma_name)
    return lemma_name

import hashlib
BLOCKSIZE = 65536

def hash_file(filename : str) -> str:
    hasher = hashlib.md5()
    with open(filename, 'rb') as f:
        buf = f.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(BLOCKSIZE)
    return hasher.hexdigest()

import sys
def eprint(*args, **kwargs):
    if "guard" not in kwargs or kwargs["guard"]:
        print(*args, file=sys.stderr, **{i:kwargs[i] for i in kwargs if i!='guard'})
        sys.stderr.flush()

import contextlib

class DummyFile:
    def write(self, x): pass
    def flush(self): pass

@contextlib.contextmanager
def nostdout():
    try:
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        yield
    finally:
        sys.stdout = save_stdout
@contextlib.contextmanager
def nostderr():
    try:
        save_stderr = sys.stderr
        sys.stderr = DummyFile()
        yield
    finally:
        sys.stderr = save_stderr

@contextlib.contextmanager
def silent():
    save_stderr = sys.stderr
    save_stdout = sys.stdout
    sys.stderr = DummyFile()
    sys.stdout = DummyFile()
    try:
        yield
    finally:
        sys.stderr = save_stderr
        sys.stdout = save_stdout

with silent():
    use_cuda = torch.cuda.is_available()
    cuda_device = "cuda:0"
    # assert use_cuda

import signal as sig
@contextlib.contextmanager
def sighandler_context(signal, f):
    old_handler = sig.signal(signal, f)
    yield
    sig.signal(signal, old_handler)

@contextlib.contextmanager
def print_time(msg : str, guard=True):
    start = time.time()
    eprint(msg + "...", end="", guard=guard)
    try:
        yield
    finally:
        eprint("{:.2f}s".format(time.time() - start), guard=guard)

mybarfmt = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'


def split_to_next_matching(openpat : str, closepat : str, target : str) \
    -> Tuple[str, str]:
    counter = 1
    openp = re.compile(openpat)
    closep = re.compile(closepat)
    firstmatch = openp.search(target)
    assert firstmatch, "Coudn't find an opening pattern!"
    curpos = firstmatch.end()
    while counter > 0:
        nextopenmatch = openp.search(target, curpos)
        nextopen = nextopenmatch.end() if nextopenmatch else len(target)

        nextclosematch = closep.search(target, curpos)
        nextclose = nextclosematch.end() if nextclosematch else len(target)
        if nextopen < nextclose:
            counter += 1
            assert nextopen + 1 > curpos, (target, curpos, nextopen)
            curpos = nextopen
        else:
            counter -= 1
            assert nextclose + 1 > curpos
            curpos = nextclose
    return target[:curpos], target[curpos:]

def multisplit_matching(openpat : str, closepat : str,
                        splitpat : str, target : str) \
                        -> List[str]:
    splits = []
    nextsplit = split_by_char_outside_matching(openpat, closepat, splitpat, target)
    rest = None
    while nextsplit:
        before, rest = nextsplit
        splits.append(before)
        nextsplit = split_by_char_outside_matching(openpat, closepat, splitpat, rest[1:])
    if rest:
        splits.append(rest[1:])
    else:
        splits.append(target)
    return splits


def split_by_char_outside_matching(openpat: str, closepat: str,
                                   splitpat: str, target: str) \
        -> Optional[Tuple[str, str]]:
    counter = 0
    curpos = 0
    with silent():
        openp = re.compile(openpat)
        closep = re.compile(closepat)
        splitp = re.compile(splitpat)

    def search_pat(pat: Pattern) -> Tuple[Optional[Match], int]:
        match = pat.search(target, curpos)
        return match, match.end() if match else len(target) + 1

    while curpos < len(target) + 1:
        _, nextopenpos = search_pat(openp)
        _, nextclosepos = search_pat(closep)
        nextsplitchar, nextsplitpos = search_pat(splitp)

        if nextopenpos < nextclosepos and nextopenpos < nextsplitpos:
            counter += 1
            assert nextopenpos > curpos
            curpos = nextopenpos
        elif nextclosepos < nextopenpos and \
                (nextclosepos < nextsplitpos or
                 (nextclosepos == nextsplitpos and counter > 0)):
            counter -= 1
            assert nextclosepos > curpos
            curpos = nextclosepos
        else:
            if counter <= 0:
                if nextsplitpos > len(target):
                    return None
                assert nextsplitchar
                return target[:nextsplitchar.start()], target[nextsplitchar.start():]
            else:
                assert nextsplitpos > curpos
                curpos = nextsplitpos
    return None


def get_possible_arg(args: argparse.Namespace, argname: str,
                     default: Any) -> Any:
    try:
        return getattr(args, argname)
    except AttributeError:
        return default


def parseSexpOneLevel(sexp_str: str) -> Union[List[str], int, Symbol]:
    if sexp_str[0] == '(':
        result = rust_parse_sexp_one_level(sexp_str)
        return result
    elif re.fullmatch(r"\s*\d+\s*", sexp_str):
        return int(sexp_str.strip())
    elif re.fullmatch(r'\s*\w+\s*', sexp_str):
        return Symbol(sexp_str)
    else:
        assert False, f"Couldn't parse {sexp_str}"


def unwrap(a: Optional[T]) -> T:
    assert a is not None
    return a


def progn(*args):
    return args[-1]


def safe_abbrev(filename: Path, all_files: List[Path]) -> str:
    if filename.stem in [f.stem for f in all_files if f != filename]:
        return escape_filename(str(filename))
    else:
        return filename.stem

class FileLock:
    def __init__(self, file_handle):
        self.file_handle = file_handle

    def __enter__(self):
        while True:
            try:
                fcntl.flock(self.file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
               time.sleep(0.01)
        return self

    def __exit__(self, type, value, traceback):
        fcntl.flock(self.file_handle, fcntl.LOCK_UN)
