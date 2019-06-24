#!/usr/bin/env python3.7

import time
import io
import math
import re
import itertools

import torch
import torch.cuda
import torch.autograd as autograd

from typing import List, Tuple, Iterable, Any, overload, TypeVar, Callable, Optional

use_cuda = torch.cuda.is_available()
assert use_cuda

def maybe_cuda(component):
    if use_cuda:
        return component.cuda()
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

def chunks(l : Iterable[Any], chunk_size : int) -> Iterable[List[Any]]:
    i = iter(l)
    next_chunk = list(itertools.islice(i, chunk_size))
    while next_chunk:
        yield next_chunk
        next_chunk = list(itertools.islice(i, chunk_size))

T = TypeVar('T')
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

import contextlib

class DummyFile:
    def write(self, x): pass
    def flush(self): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
@contextlib.contextmanager
def nostderr():
    save_stderr = sys.stderr
    sys.stderr = DummyFile()
    yield
    sys.stderr = save_stderr

@contextlib.contextmanager
def silent():
    save_stderr = sys.stderr
    save_stdout = sys.stdout
    sys.stderr = DummyFile()
    sys.stdout = DummyFile()
    yield
    sys.stderr = save_stderr
    sys.stdout = save_stdout

import signal as sig
@contextlib.contextmanager
def sighandler_context(signal, f):
    old_handler = sig.signal(signal, f)
    yield
    sig.signal(signal, old_handler)
