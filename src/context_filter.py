#!/usr/bin/env python3.7

import re
import functools
import argparse

from typing import Dict, Callable, Union, List, cast, Tuple, Iterable

from tokenizer import get_symbols
import serapi_instance

ContextData = Dict[str, Union[str, List[str]]]
ContextFilter = Callable[[ContextData, str, ContextData, argparse.Namespace], bool]

def filter_and(*args : ContextFilter) -> ContextFilter:
    def filter_and2(f1 : ContextFilter, f2 : ContextFilter) -> ContextFilter:
        return lambda in_data, tactic, next_in_data, args: \
            (f1(in_data, tactic, next_in_data, args) and
             f2(in_data, tactic, next_in_data, args))
    if len(args) == 1:
        return args[0]
    else:
        return filter_and2(args[0], filter_and(*args[1:]))
def filter_or(*args : ContextFilter) -> ContextFilter:
    def filter_or2(f1 : ContextFilter, f2 : ContextFilter) -> ContextFilter:
        return lambda in_data, tactic, next_in_data, args:\
            (f1(in_data, tactic, next_in_data, args) or
             f2(in_data, tactic, next_in_data, args))
    if len(args) == 1:
        return args[0]
    else:
        return filter_or2(args[0], filter_or(*args[1:]))

def no_compound_or_bullets(in_data : ContextData, tactic : str,
                           next_in_data : ContextData,
                           arg_values : argparse.Namespace) -> bool:
    return (not re.match("\s*[\{\}\+\-\*].*", tactic, flags=re.DOTALL) and
            not re.match(".*;.*", tactic, flags=re.DOTALL))
def not_proof_keyword(in_data : ContextData, tactic : str,
                      next_in_data : ContextData,
                      arg_values : argparse.Namespace) -> bool:
    return not re.match("Proof", tactic)
def not_background_subgoal(in_data : ContextData, tactic : str,
                           next_in_data : ContextData,
                           arg_values : argparse.Namespace) -> bool:
    return not re.match("\d*:.*", tactic)
def not_vernac(in_data : ContextData, tactic : str,
               next_in_data : ContextData,
               arg_values : argparse.Namespace) -> bool:
    return not (re.match("\s*Opaque", tactic))

def goal_changed(in_data : ContextData, tactic : str,
                 next_in_data : ContextData,
                 arg_values : argparse.Namespace) -> bool:
    return in_data["goal"] != next_in_data["goal"]

def hyps_changed(in_data : ContextData, tactic : str,
                 next_in_data : ContextData,
                 arg_values : argparse.Namespace) -> bool:
    return in_data["hyps"] != next_in_data["hyps"]

def no_args(in_data : ContextData, tactic : str,
            next_in_data : ContextData,
            arg_values : argparse.Namespace) -> bool:
    return re.match("\s*\S*\.", tactic) != None

def args_vars_in_context(in_data : ContextData, tactic : str,
                         next_in_data : ContextData,
                         arg_values : argparse.Namespace) -> bool:
    stem, args_string  = serapi_instance.split_tactic(tactic)
    args = args_string[:-1].split()
    if not serapi_instance.tacticTakesHypArgs(stem) and len(args) > 0:
        return False
    var_names = serapi_instance.get_vars_in_hyps(cast(List[str], in_data["hyps"]))
    for arg in args:
        if not arg in var_names:
            return False
    return True

def tactic_literal(tactic_to_match : str,
                   in_data: ContextData, tactic : str,
                   new_in_data : ContextData,
                   arg_values : argparse.Namespace) -> bool:
    return re.match("\s*{}(\s.+)?\.".format(tactic_to_match), tactic) != None
def tactic_eliteral(tactic_to_match : str,
                    in_data: ContextData, tactic : str,
                    new_in_data : ContextData,
                    arg_values : argparse.Namespace) -> bool:
    return re.match("\s*e?{}(\s.+)?\.".format(tactic_to_match), tactic) != None
def max_args(num_str : str,
             in_data: ContextData, tactic : str,
             new_in_data : ContextData,
             arg_values : argparse.Namespace) -> bool:
    stem, args_string  = serapi_instance.split_tactic(tactic)
    args = args_string.strip()[:-1].split()
    return len(args) <= int(num_str)

def numeric_args(in_data : ContextData, tactic : str,
                 next_in_data : ContextData,
                 arg_values : argparse.Namespace) -> bool:
    goal = in_data["goal"]
    goal_words = get_symbols(cast(str, goal))
    stem, rest = serapi_instance.split_tactic(tactic)
    args = get_subexprs(rest.strip("."))
    for arg in args:
        if not re.match("\d+", arg):
            return False
    return True

def args_token_in_goal(in_data : ContextData, tactic : str,
                       next_in_data : ContextData,
                       arg_values : argparse.Namespace) -> bool:
    goal = in_data["goal"]
    goal_words = get_symbols(cast(str, goal))[:arg_values.max_length]
    stem, rest = serapi_instance.split_tactic(tactic)
    args = get_subexprs(rest.strip("."))
    for arg in args:
        if not arg in goal_words:
            return False
    return True

def get_subexprs(text : str) -> List[str]:
    def inner() -> Iterable[str]:
        cur_expr = ""
        paren_depth = 0
        for c in text:
            if c == "(":
                paren_depth += 1
            cur_expr += c
            if c == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    yield cur_expr.strip()
                    cur_expr = ""
            if c == " " and paren_depth == 0:
                if cur_expr != "":
                    yield cur_expr.strip()
                cur_expr = ""
        if cur_expr != "":
            yield cur_expr.strip()
    return list(inner())

def split_toplevel(specstr : str) -> List[str]:
    paren_depth = 0
    operators = ["%", "+"]
    pieces : List[str] = []
    curPiece = ""
    for c in specstr:
        if paren_depth > 0:
            if c == ")":
                paren_depth -= 1
            elif c == "(":
                paren_depth += 1
            if paren_depth > 0:
                curPiece += c
            else:
                if curPiece != '':
                    pieces.append(curPiece)
                curPiece = ""
        elif c == "(":
            if curPiece != '':
                pieces.append(curPiece)
            curPiece = ""
            paren_depth += 1
        elif c in operators:
            if curPiece != '':
                pieces.append(curPiece)
            pieces.append(c)
            curPiece = ""
        else:
            curPiece += c
    assert paren_depth == 0
    if curPiece != "":
        pieces.append(curPiece)
    return pieces

def get_context_filter(specstr : str) -> ContextFilter:
    pieces = split_toplevel(specstr)
    if not "+" in specstr and not "%" in specstr:
        for prefix, func, arg_str in special_prefixes:
            match = re.match("^{}(.*)".format(prefix), specstr)
            if match:
                return functools.partial(func, match.group(1))
        assert specstr in context_filters, "Invalid atom {}! Valid atoms are {}"\
            .format(specstr, list(context_filters.keys()) +
                    [get_prefix_argstr(prefix_entry) for prefix_entry in special_prefixes])
        return context_filters[specstr]
    if len(pieces) == 1:
        return get_context_filter(pieces[0])
    else:
        assert len(pieces) % 2 == 1, "Malformed subexpression {}! {}".format(specstr, pieces)
        if pieces[1] == "%":
            assert all([operator == "%" for operator in pieces[1::2]])
            return filter_and(*[get_context_filter(substr) for substr in pieces[::2]])
        else:
            assert pieces[1] == "+", pieces
            assert all([operator == "+" for operator in pieces[1::2]])
            return filter_or(*[get_context_filter(substr) for substr in pieces[::2]])

ParamterizedFilterFunc = Callable[[str, ContextData, str, ContextData, argparse.Namespace], bool]
PrefixEntry = Tuple[str, ParamterizedFilterFunc, str]
special_prefixes : List[PrefixEntry] \
    = [
        ("tactic:", tactic_literal, "<tacticname>"),
        ("etactic:", tactic_eliteral, "<tacticname>"),
        ("~tactic:", lambda *args: not tactic_literal(*args), "<tacticname>"),
        ("~etactic:", lambda *args: not tactic_eliteral(*args), "<tacticname>"),
        ("maxargs:", max_args, "<number>"),
    ]

def get_prefix_argstr(prefix_entry : PrefixEntry):
    prefix, func, argstr = prefix_entry
    return "{}{}".format(prefix, argstr)

context_filters : Dict[str, ContextFilter] = {
    "default": filter_and(no_compound_or_bullets,
                          not_proof_keyword,
                          not_background_subgoal,
                          not_vernac),
    "none": lambda *args: False,
    "all": lambda *args: True,
    "goal-changes": filter_and(goal_changed, no_compound_or_bullets),
    "hyps-change": filter_and(hyps_changed, no_compound_or_bullets),
    "something-changes":filter_and(filter_or(goal_changed, hyps_changed),
                                   no_compound_or_bullets),
    "no-args": filter_and(no_args, no_compound_or_bullets),
    "hyp-args":filter_and(args_vars_in_context, no_compound_or_bullets),
    "goal-args" : args_token_in_goal,
    "numeric-args" : numeric_args,
}
