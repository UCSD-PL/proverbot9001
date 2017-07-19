#!/usr/bin/env python3

import re

vernacular_binder = [
    "Definition",
    "Inductive",
    "Fixpoint",
    "Theorem",
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

def color_word(color, word):
    return "<span style=\"color:{}\">{}</span>".format(color, word)

def highlight_comments(page):
    result = ""
    comment_depth = 0
    for i in range(len(page)):
        if(page[i:i+2] == "(*"):
            comment_depth += 1
            if comment_depth == 1:
                result += "<span style=\"color:{}\">".format(comment_color)
        result += page[i]
        if(page[i-1:i+1] == "*)"):
            comment_depth -= 1
            if comment_depth == 0:
                result += "</span>"
    return result;

def syntax_highlight(page):
    for vernac in vernacular_words:
        page = re.sub(vernac,
                      color_word(vernacular_color, vernac),
                      page)
    return highlight_comments(page);
