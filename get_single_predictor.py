#!/usr/bin/env python

import argparse
import json
import random
import re
import time
import contextlib
import math
import pandas as pd
import sys
from pathlib import Path
import os
import itertools
import json
import glob

CLI=argparse.ArgumentParser()
CLI.add_argument(
    "--preds", # set of directories holding predictor outputs
    nargs="*",  
    type=str,
    default=None,  
    )
CLI.add_argument(
    "--subset-size",  # size of subset, subset is only output if the size is greater than 0
    type=int,
    default=0,  # default if nothing is provided
    )

args = CLI.parse_args()
#df = pd.read_csv(sys.argv[1])
list_of_predictors = args.preds
predictor = list_of_predictors[0]
df = None
i = 0

files = glob.glob("./" + predictor + "/**/*", recursive=True)
for each_file in files:
    # check only text files
    if each_file.endswith('-proofs.txt'):
        with open(each_file, 'r') as proof_file:
            for aline in proof_file.readlines():
                if not aline[0] != "[":
                    json_line = json.loads(aline)
                    if json_line[1]["status"] == 'SUCCESS':
                        df_row = {"project": json_line[0][0], "file": json_line[0][1], "class": json_line[0][2], "lemma": json_line[0][3], ("s" + predictor): True}
                    else:
                        df_row = {"project": json_line[0][0], "file": json_line[0][1], "class": json_line[0][2], "lemma": json_line[0][3], ("s" + predictor): False}
                    if json_line[1]["status"] == 'SUCCESS':
                        df_row["ns" + predictor] = True
                    else:
                        df_row["ns" + predictor] = False
                    df = pd.concat([df, pd.DataFrame([df_row])], ignore_index=True)

df['allstatusone'] = False
for predictor in [list_of_predictors[0]]:
    df['allstatusone'] = df['allstatusone'] | df['s' + predictor]

total_solved_one = df[df.allstatusone == True].shape[0]
print(f"{total_solved_one:.4f}")
