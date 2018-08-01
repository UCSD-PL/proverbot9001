#!/usr/bin/env python3

import time
import io
import math
import re

import torch
import torch.cuda
import torch.autograd as autograd

from serapi_instance import kill_comments

from typing import List, Any

Sentence = List[int]
DataSet = List[List[Sentence]]

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

def get_stem(tactic : str) -> str:
    tactic = kill_comments(tactic).strip()
    if re.match("[-+*\{\}]", tactic):
        return tactic
    if re.match(".*;.*", tactic):
        return tactic
    match = re.match("^\(?(\w+).*", tactic)
    assert match, "tactic \"{}\" doesn't match!".format(tactic)
    return match.group(1)
