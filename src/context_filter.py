#!/usr/bin/env python3

from typing import Dict, Callable, Union, List, cast
import re

ContextData = Dict[str, Union[str, List[str]]]
ContextFilter = Callable[[ContextData, str, ContextData], bool]

def filter_and(f1 : ContextFilter, f2 : ContextFilter) -> ContextFilter:
    return lambda in_data, tactic, next_in_data: (f1(in_data, tactic, next_in_data) and
                                                  f2(in_data, tactic, next_in_data))
def filter_or(f1 : ContextFilter, f2 : ContextFilter) -> ContextFilter:
    return lambda in_data, tactic, next_in_data: (f1(in_data, tactic, next_in_data) or
                                                  f2(in_data, tactic, next_in_data))

def no_compound_or_bullets(in_data : ContextData, tactic : str,
                           next_in_data : ContextData) -> bool:
    return (not re.match("\s*[\{\}\+\-\*].*", tactic, flags=re.DOTALL) and
            not re.match(".*;.*", tactic, flags=re.DOTALL))

def goal_changed(in_data : ContextData, tactic : str,
                 next_in_data : ContextData) -> bool:
    return in_data["goal"] != next_in_data["goal"]

def hyps_changed(in_data : ContextData, tactic : str,
                 next_in_data : ContextData) -> bool:
    return in_data["hyps"] != next_in_data["hyps"]

def no_args(in_data : ContextData, tactic : str,
            next_in_data : ContextData) -> bool:
    return re.match("\s*\S*\.", tactic) != None

def get_vars_in_hyps(hyps : str) -> List[str]:
    hyps_replaced = re.sub("forall.*?,", "",
                           re.sub("fun.*?=>", "", hyps, flags=re.DOTALL),
                           flags=re.DOTALL)
    var_terms = re.findall("(\S+(?:, \S+)*) (?::=.*?)?: .*?",
                           hyps_replaced, flags=re.DOTALL)
    var_names = [name.strip() for term in var_terms for name in term.split(",")]
    return var_names

def args_vars_in_context(in_data : ContextData, tactic : str,
                         next_in_data : ContextData) -> bool:
    stem, *args = tactic[:-1].split()
    var_names = get_vars_in_hyps(cast(str, in_data["hyps"]))
    for arg in args:
        if not arg in var_names:
            return False
    return True

def apply_lemma(in_data : ContextData, tactic : str,
                next_in_data : ContextData) -> bool:
    return re.match("\s*e?apply\s*\S+\.", tactic) != None

def get_context_filter(specstr : str) -> ContextFilter:
    if "+" in specstr:
        curFilter = context_filters["none"]
        for cfilter in specstr.split("+"):
            assert cfilter in context_filters, \
                "Not a valid atom! {}\nValid atoms are {}"\
                .format(cfilter, context_filters.keys())
            curFilter = filter_or(curFilter, context_filters[cfilter])
        return curFilter
    else:
        assert specstr in context_filters, \
            "Not a valid atom! {}\nValid atoms are {}"\
            .format(cfilter, context_filters.keys())
        return context_filters[specstr]

context_filters : Dict[str, ContextFilter] = {
    "default": no_compound_or_bullets,
    "none": lambda *args: False,
    "all": lambda *args: True,
    "goal-changes": filter_and(goal_changed, no_compound_or_bullets),
    "hyps-change": filter_and(hyps_changed, no_compound_or_bullets),
    "something-changes":filter_and(filter_or(goal_changed, hyps_changed),
                                   no_compound_or_bullets),
    "no-args": filter_and(no_args, no_compound_or_bullets),
    "context-var-args":filter_and(args_vars_in_context, no_compound_or_bullets),
    "apply":apply_lemma,
}
