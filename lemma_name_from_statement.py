#!/usr/bin/env python3

import re
import sys

def main() -> None:
    for line in sys.stdin:
        print(lemma_name_from_statement(line))

def lemma_name_from_statement(stmt : str) -> str:
    lemma_match = re.match("\s*\S+\s+([\w']+)", stmt)
    assert lemma_match, stmt
    lemma_name = lemma_match.group(1)
    assert ":" not in lemma_name, stmt
    return lemma_name

if __name__ == "__main__":
    main()
