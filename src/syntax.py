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
from dataclasses import dataclass
from typing import List, Union, Iterable
from util import unwrap

vernacular_binder = [
    "Definition",
    "Inductive",
    "Fixpoint",
    "Theorem",
    "Function",
    "Remark",
    "Hypothesis",
    "Lemma",
    "Example",
    "Ltac",
    "Record",
    "Variable",
    "Variables",
    "Section",
    "End",
    "Instance",
    "Module",
    "Context"
]
vernacular_words = vernacular_binder + [
    "Proof",
    "Qed",
    "Defined",
    "Require",
    "Import",
    "Export",
    "Print",
    "Assumptions",
    "Local",
    "Open",
    "Scope",
    "Admitted",
    "Notation",
    "Set",
    "Unset",
    "Implicit",
]

local_binder = [
    "forall",
    "fun"
]

syntax_words = local_binder + [
    "Type",
    "Set",
    "Prop",
    "if",
    "then",
    "else",
    "match",
    "with",
    "end",
    "as",
    "in",
    "return",
    "using",
    "let"
]

vernacular_color = "#a020f0"
syntax_color = "#0027a6"
global_bound_color = "#3b10ff"
local_bound_color = "#a0522d"
comment_color = "#004800"

def color_word(color : str, word : str) -> str:
    return f"<span style=\"color:{color}\">{word}</span>"

@dataclass
class ColoredString:
    contents : str
    color : str

def highlight_comments(code : str) -> List[Union[str, ColoredString]]:
    def generate() -> Iterable[Union[str, ColoredString]]:
        cur_string = ""
        comment_depth = 0
        cur_pos = 0
        openp = re.compile(r"\(\*", re.DOTALL)
        closep = re.compile(r"\*\)", re.DOTALL)

        while cur_pos < len(code):
            next_open_match = openp.search(code, cur_pos)
            next_close_match = closep.search(code, cur_pos)
            if next_open_match == None and next_close_match == None:
                if comment_depth <= 0:
                    yield cur_string + code[cur_pos:]
                else:
                    yield ColoredString(cur_string + code[cur_pos:], comment_color)
                break
            if next_close_match == None or \
            (next_open_match != None and
             unwrap(next_open_match).start() < unwrap(next_close_match).start()):
                next_open_match = unwrap(next_open_match)
                cur_string += code[cur_pos:next_open_match.start()]
                if comment_depth == 0:
                    yield cur_string
                    cur_string = ""
                cur_string += next_open_match.group(0)
                cur_pos = next_open_match.end()
                comment_depth += 1
            else:
                next_close_match = unwrap(next_close_match)
                cur_string += code[cur_pos:next_close_match.end()]
                cur_pos = next_close_match.end()
                comment_depth -= 1
                if comment_depth == 0:
                    yield ColoredString(cur_string, comment_color)
                    cur_string = ""
    return list(generate())

def highlight_word(color: str, word : str, colored_text : List[Union[str, ColoredString]]) \
    -> List[Union[str, ColoredString]]:
    word_pat = re.compile(rf"\b{word}\b")
    def generate() -> Iterable[Union[str, ColoredString]]:
        for block in colored_text:
            if isinstance(block, ColoredString):
                yield block
            else:
                text_left = block
                match = word_pat.search(text_left)
                while match:
                    yield text_left[:match.start()]
                    yield ColoredString(word, color)
                    text_left = text_left[match.end():]
                    match = word_pat.search(text_left)
                yield text_left
    return list(generate())

def highlight_words(color : str, words : List[str], text : List[Union[str, ColoredString]]) \
    -> List[Union[str, ColoredString]]:
    result = text
    for word in words:
        result = highlight_word(color, word, result)
    return result

def syntax_highlight(code : str) -> List[Union[str, ColoredString]]:
    return highlight_words(syntax_color, syntax_words,
                           highlight_words(vernacular_color, vernacular_words,
                                           highlight_comments(code)))

def strip_comments(command : str) -> str:
    result = ""
    comment_depth = 0
    for i in range(len(command)):
        if command[i:i+2] == "(*":
            comment_depth += 1
        if comment_depth < 1:
            result += command[i]
        if command[i-1:i+1] == "*)":
            comment_depth -= 1
    return result
