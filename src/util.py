#!/usr/bin/env python3

import time
import io
import math
import re

import torch
import torch.cuda
import torch.autograd as autograd

from serapi_instance import kill_comments

from typing import List, Iterable, Any, overload

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

def chunks(l : Iterable[Any], chunk_size : int):
    rest_list = l
    while len(rest_list) > 0:
        chunk = rest_list[:chunk_size]
        rest_list = rest_list[chunk_size:]
        yield chunk
