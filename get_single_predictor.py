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
    # print(each_file.split("/")[-1])
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

#df.replace({'CRASHED': True, 'FAILURE': False,'SKIPPED': False, 'CRASHED': False, 'INCOMPLETE': False}, inplace=True)

#filtered_df = df[df.allstatus == True]
#s_columns = df.filter(regex='^s')
#s_true_count = s_columns.any(axis=1)
#df['crashandsuccess'] = df['ns' + list_of_predictors[-1]] & s_true_count

#for index, row in filtered_df.iterrows():
#    print(row)
#df.to_csv("tiny_test.csv")

df['allstatusone'] = False
for predictor in [list_of_predictors[0]]:
    df['allstatusone'] = df['allstatusone'] | df['s' + predictor]
#s_columns = df.filter(regex='^s')
#s_true_count = s_columns.any(axis=1)
#df['crashandsuccess'] = df['ns' + list_of_predictors[-1]] & s_true_count

#for index, row in filtered_df.iterrows():
#    print(row)
total_solved_one = df[df.allstatusone == True].shape[0]
#df.to_csv("tiny_test.csv")
print(total_solved_one)
print(df.shape[0])
#print(df.shape[0])

#best_whole = 0
#if args.subset_size > 0:
#    for subset in itertools.combinations(status_columns,args.subset_size):
#        tot_col = df[subset[0]]
#        for index in subset[1:]:
#            next_col = df[index]
#            tot_col = tot_col | next_col
#        total_solved = (tot_col).sum()
#        if total_solved > best_whole:
#            best_whole = total_solved
#            best_subset = subset

#    print("best subset")
#    for index in best_subset:
#        print(index.split("status")[1])
#    print("total proofs solved")
#    print(str(best_whole) + "/507")
#    print("percentage of proof solved")
#    print(str(100 * best_whole/507) + "%")
