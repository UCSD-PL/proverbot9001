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
    "--pred", # set of directories holding predictor outputs
    type=str,
    default=None,  
    )
CLI.add_argument(
    "--subset-size",  # size of subset, subset is only output if the size is greater than 0
    type=int,
    default=0,  # default if nothing is provided
    )

args = CLI.parse_args()
predictor = args.pred
df = None
i = 0

df = pd.DataFrame()

num_proofs = 0
num_steps = 0 

files = glob.glob("./" + predictor + "/**/*", recursive=True)
for each_file in files:
    if each_file.endswith('-proofs.txt'):
        with open(each_file, 'r') as proof_file:
            for aline in proof_file.readlines():
                if not aline[0] != "[":
                    json_line = json.loads(aline)
                    if json_line[1]["status"] == 'SUCCESS':
                        json_line = json.loads(aline)
                        num_steps += int(json_line[1]["steps_taken"])
                        num_proofs += 1


avg_steps = num_steps/num_proofs
print("Avg steps taken")
print(avg_steps)
