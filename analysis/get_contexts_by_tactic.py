#!/usr/bin/env python3

import csv
import argparse
import re

from typing import Dict, Any, List, Tuple

Row = Tuple[List[str], str, str]

def get_matching_rows(filenames : List[str],
                      correct_tactic : Optional[str],
                      predicted_tactic : Optiona[str],
                      max_rows : float = float("Inf")) -> \
    List[Row]:
    pass

def main() -> None:
    parser = argparse.ArgumentParser(description="Get all contexts in the data which match the given correct and predicted tactic")
    parser.add_argument("filenames", nargs="+",
                        help="Search contexts in a dataset to find ones matching "\
                        "a particular correct and/or predicted tactic")
