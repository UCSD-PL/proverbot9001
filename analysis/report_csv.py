#!/usr/bin/env python3

import csv
import re
import itertools
from io import TextIOWrapper
from typing import NamedTuple, Tuple, Union, List, Dict, Iterator, cast

from context_filter import ContextFilter
from format import format_goal, format_hypothesis

class PredictionResult(NamedTuple):
    prediction : str
    grade : str
class TacticRow(NamedTuple):
    command : str
    hyps : str
    goal : str
    predictions : List[PredictionResult]
class CommandRow(NamedTuple):
    command : str

FileRow = Union[CommandRow, TacticRow]

def read_options(f : TextIOWrapper) -> Dict[str, str]:
    next_option_line = f.tell()
    options : Dict[str, str] = {}
    next_line = f.readline()
    while(next_line.startswith("#")):
        option_match = re.match("# (.*): (.*)", next_line)
        assert option_match
        key, value = option_match.group(1, 2)
        options[key] = value
        next_option_line = f.tell()
        next_line = f.readline()
    f.seek(next_option_line)
    return options

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def read_csvfile_rest(csvfile_handle : TextIOWrapper) -> Iterator[FileRow]:
    reader = csv.reader(csvfile_handle)
    for row in reader:
        if len(row) == 1:
            yield CommandRow(command=row[0])
        else:
            command, hyps, goal, *predictions = row
            yield TacticRow(command=command, hyps=hyps, goal=goal,
                            predictions=[PredictionResult(predictions[i], predictions[i+1])
                                         for i in range(0, len(predictions) - 1, 2)])
    csvfile_handle.close()

def read_csvfile(filename : str) -> Tuple[Dict[str, str], Iterator[FileRow]]:
    csvfile = cast(TextIOWrapper, open(filename, 'r', newline=''))
    options = read_options(csvfile)
    reader = csv.reader(csvfile)
    rows = read_csvfile_rest(csvfile)
    return options, rows

def tactics_only(rows : Iterator[FileRow]) -> Iterator[TacticRow]:
    for row in rows:
        if isinstance(row, CommandRow):
            continue
        else:
            yield row

def check_cfilter_row(cfilter : ContextFilter, curRow : FileRow, nextRow : FileRow):
    if not nextRow or isinstance(nextRow, CommandRow):
        new_hyps, new_goal = "", ""
    else:
        new_hyps, new_goal = nextRow.hyps, nextRow.goal
    if isinstance(curRow, CommandRow):
        return True
    else:
        return cfilter({"goal": format_goal(curRow.goal),
                        "hyps": format_hypothesis(curRow.hyps)},
                       curRow.command,
                       {"goal": format_goal(new_goal),
                        "hyps": format_hypothesis(new_hyps)})

def filter_rows(rows : Iterator[FileRow], cfilter : ContextFilter) -> Iterator[FileRow]:
    for row, nextRow in pairwise(rows):
        if check_cfilter_row(cfilter, row, nextRow):
            yield row
