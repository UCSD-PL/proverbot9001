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
import json
from typing import List, TextIO, Optional, NamedTuple, Union


class ScrapedTactic(NamedTuple):
    relevant_lemmas: List[str]
    prev_tactics: List[str]
    hypotheses: List[str]
    goal: str
    tactic: str


class TacticContext(NamedTuple):
    relevant_lemmas: List[str]
    prev_tactics : List[str]
    hypotheses : List[str]
    goal : str

ScrapedCommand = Union[ScrapedTactic, str]

def strip_scraped_output(scraped : ScrapedTactic) -> TacticContext:
    relevant_lemmas, prev_tactics, hypotheses, goal, output = scraped
    assert prev_tactics != None
    assert hypotheses != None
    assert goal != None
    assert output != None
    return TacticContext(relevant_lemmas, prev_tactics, hypotheses, goal)

def read_tuple(f_handle : TextIO) -> Optional[ScrapedCommand]:
    line = f_handle.readline()
    try:
        obj = json.loads(line)
    except:
        print(f"line: {line}")
        raise
    if isinstance(obj, str):
        return obj
    else:
        return ScrapedTactic(obj["relevant_lemmas"],
                             obj["prev_tactics"],
                             obj["prev_hyps"],
                             obj["prev_goal"],
                             obj["tactic"])

def read_tactic_tuple(f_handle : TextIO) -> Optional[ScrapedTactic]:
    next_tuple = read_tuple(f_handle)
    while(isinstance(next_tuple, str)):
        next_tuple = read_tuple(f_handle)
    return next_tuple
