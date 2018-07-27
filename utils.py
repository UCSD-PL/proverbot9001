#!/usr/bin/env python3

import re
import math
import time

from typing import Any

import torch
import torch.cuda

use_cuda = torch.cuda.is_available()
assert use_cuda

def maybe_cuda(component):
    if use_cuda:
        return component.cuda()
    else:
        return component

def get_stem(tactic : str) -> str:
    if re.match("[-+*\{\}]", tactic):
        return tactic
    if re.match(".*;.*", tactic):
        return tactic
    match = re.match("^\(?(\w+).*", tactic)
    assert match, "tactic \"{}\" doesn't match!".format(tactic)
    return match.group(1)

def LongTensor(arr : Any) -> torch.LongTensor:
    if use_cuda:
        return torch.cuda.LongTensor(arr)
    else:
        return torch.LongTensor(arr)

def FloatTensor(*args) -> torch.FloatTensor:
    if use_cuda:
        return torch.cuda.FloatTensor(*args)
    else:
        return torch.FloatTensor(*args)

def asMinutes(s : float) -> str:
    m = math.floor(s / 60)
    s -= m * 60
    return "{}m {:.2f}s".format(m, s)

def timeSince(since : float, percent : float) -> str:
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return "{} (- {})".format(asMinutes(s), asMinutes(rs))
