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
#print('list of predictors')
#print(list_of_predictors)
first_predictor = list_of_predictors[0]
df = None
i = 0

for predictor in list_of_predictors[0:]:
    tmp_df = pd.DataFrame()

    #for each_file in os.listdir("./" + predictor + "/*/"):
    #print("proofs solved predictor " + str(i))
    files = glob.glob("./" + predictor + "/**/*", recursive=True)
    for each_file in files:
        # check only text files
        # print(each_file.split("/")[-1])
        if each_file.endswith('-proofs.txt'):
            with open(each_file, 'r') as proof_file:
                for aline in proof_file.readlines():
                    if not aline[0] != "[":
                        json_line = json.loads(aline)
                        if json_line[1]["status"] == 'SUCCESS':
                            df_row = {"project": json_line[0][0], "file": json_line[0][1], "class": json_line[0][2], "lemma": json_line[0][3], ("s" + predictor): True, ("s" + predictor + "steps"): json_line[1]["steps_taken"]}
                        else:
                            df_row = {"project": json_line[0][0], "file": json_line[0][1], "class": json_line[0][2], "lemma": json_line[0][3], ("s" + predictor): False, ("s" + predictor + "steps"): json_line[1]["steps_taken"]}
                        tmp_df = pd.concat([tmp_df, pd.DataFrame([df_row])], ignore_index=True)
    # Find common values for each key
    if df is None:
        df = tmp_df
    else:
        common_projects = set(df['project']).intersection(tmp_df['project'])
        common_files = set(df['file']).intersection(tmp_df['file'])
        common_classes = set(df['class']).intersection(tmp_df['class'])
        common_lemmas = set(df['lemma']).intersection(tmp_df['lemma'])

        df = pd.merge(df, tmp_df, on=["project", "file", "class", "lemma"], how='left')
        df.drop_duplicates(keep='first', inplace=True, ignore_index=True)


    tmp_correct = 0
    i = i + 1

df['all_predictors_status'] = False
df = df.fillna(False)
for predictor in list_of_predictors:
    df['all_predictors_status'] = df['all_predictors_status'] | df['s' + predictor]

df['total_steps'] = 0
for predictor in list_of_predictors:
    df['total_steps'] += pd.to_numeric(df[f's{predictor}steps']).astype(int)

print(df['all_predictors_status'])

avg_steps = df.loc[df['all_predictors_status'], 'total_steps'].mean()

print("average steps")
print(avg_steps)

