#!/usr/bin/env python3

import re
import sys

import coq_serapy

def main() -> None:
    for line in sys.stdin:
        print(coq_serapy.lemma_name_from_statement(coq_serapy.kill_comments(line.strip())))

if __name__ == "__main__":
    main()
