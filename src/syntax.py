#!/usr/bin/env python3.7

import re

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
syntax_color = "#228b22"
global_bound_color = "#3b10ff"
local_bound_color = "#a0522d"
comment_color = "#004800"

def color_word(color : str, word : str) -> str:
    return f"<span style=\"color:{color}\">{word}</span>"

from typing import Pattern

def highlight_comments(page : str) -> str:
    result = ""
    comment_depth = 0
    curpos = 0
    openp = re.compile(r"\(\*")
    closep = re.compile(r"\*\)")
    def search_pat(pat : Pattern) -> int:
        match = pat.search(page, curpos)
        return match.end() if match else len(page) + 1

    while curpos < len(page):
        nextopenpos = search_pat(openp)
        nextclosepos = search_pat(closep)

        if nextopenpos < nextclosepos:
            result += page[curpos:nextopenpos]
            if comment_depth == 0:
                result += "<span style=\"color:{}\">".format(comment_color)
            curpos = nextopenpos
            comment_depth += 1
        elif nextclosepos < nextopenpos:
            result += page[curpos:nextclosepos]
            curpos = nextclosepos
            comment_depth -= 1
            if comment_depth == 0:
                result += "</span>"
        elif nextclosepos == nextopenpos:
            assert nextclosepos == len(page) + 1 and \
                nextopenpos == len(page) + 1
        result += page[curpos:]
    return result

def syntax_highlight(page : str) -> str:
    for vernac in vernacular_words:
        colored_vernac = color_word(vernacular_color, vernac)
        pat = re.compile(rf"\b{vernac}\b")
        page = pat.sub(colored_vernac, page)
    result = highlight_comments(page)
    return result

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
