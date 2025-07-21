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
                            df_row = {"project": json_line[0][0], "file": json_line[0][1], "class": json_line[0][2], "lemma": json_line[0][3], ("s" + predictor): True}
                        else:
                            df_row = {"project": json_line[0][0], "file": json_line[0][1], "class": json_line[0][2], "lemma": json_line[0][3], ("s" + predictor): False}
                        #if json_line[1]["status"] == 'SUCCESS':
                        #    df_row["ns" + predictor] = True
                        #else:
                        #    df_row["ns" + predictor] = False
                        tmp_df = pd.concat([tmp_df, pd.DataFrame([df_row])], ignore_index=True)
    # Find common values for each key
    if df is None:
        df = tmp_df
    else:
        common_projects = set(df['project']).intersection(tmp_df['project'])
        common_files = set(df['file']).intersection(tmp_df['file'])
        common_classes = set(df['class']).intersection(tmp_df['class'])
        common_lemmas = set(df['lemma']).intersection(tmp_df['lemma'])

        # Filter both DataFrames to include only rows with common key values
        #tmp_df_filtered = tmp_df[
        #    tmp_df['project'].isin(common_projects) &
        #    tmp_df['file'].isin(common_files) &
        #    tmp_df['class'].isin(common_classes) &
        #    tmp_df['lemma'].isin(common_lemmas)
        #]
        #df_filtered = df[
        #    df['project'].isin(common_projects) &
        #    df['file'].isin(common_files) &
        #    df['class'].isin(common_classes) &
        #    df['lemma'].isin(common_lemmas)
        #]

        # Perform the merge on the filtered DataFrames
        #df = pd.merge(df_filtered, tmp_df_filtered, on = ["project", "file", "class", "lemma"], how='inner')
        df = pd.merge(df, tmp_df, on=["project", "file", "class", "lemma"], how='left')
        df.drop_duplicates(keep='first', inplace=True, ignore_index=True)


    tmp_correct = 0
    i = i + 1

#df.replace({'CRASHED': True, 'FAILURE': False,'SKIPPED': False, 'CRASHED': False, 'INCOMPLETE': False}, inplace=True)

#filtered_df = df[df.allstatus == True]
df = df.fillna(False)
df['allstatusone'] = False
for predictor in [list_of_predictors[0]]:
    df['allstatusone'] = df['allstatusone'] | df['s' + predictor]
df['allstatus'] = False
for predictor in list_of_predictors[1:]:
    df['allstatus'] = df['allstatus'] | df['s' + predictor]
df['allstatustwo'] = False
for predictor in list_of_predictors[0:]:
    df['allstatustwo'] = df['allstatustwo'] | df['s' + predictor]
#print(df)
filtered_rows = df[(df['s'+list_of_predictors[0]] == True)]
filtered_rows = filtered_rows[(filtered_rows['s'+list_of_predictors[1]] == False)]
filtered_rows = filtered_rows.drop("allstatus", axis=1)
filtered_rows = filtered_rows.drop("allstatusone", axis=1)
filtered_rows = filtered_rows.drop("allstatustwo", axis=1)
#print(filtered_rows.to_string())
#s_columns = df.filter(regex='^s')
#s_true_count = s_columns.any(axis=1)
#df['crashandsuccess'] = df['ns' + list_of_predictors[-1]] & s_true_count

#for index, row in filtered_df.iterrows():
#    print(row)
total_solved_one = df[df.allstatusone == True].shape[0]
total_solved = df[df.allstatus == True].shape[0]
total_solved_two = df[df.allstatustwo == True].shape[0]
#df.to_csv("tiny_test.csv")
#print("total solved first pred: " + str(total_solved_one))
#print("percentage")
#print(str(100 * total_solved_one/df.shape[0]) + "%")
#print("total solved rest of preds: " + str(total_solved))
#print("percentage")
#print(str(100 * total_solved/df.shape[0]) + "%")
#print("alltogether " + str(total_solved_two))
#print("percentage")
#print(str(100 * total_solved_two/df.shape[0]) + "%", flush=True)

print(total_solved_two)
print(df.shape[0])


#filtered_df = df[(~df["s" + list_of_predictors[0]]) &
#                 (~df["s" + list_of_predictors[1]]) &
#                 (df["s" + list_of_predictors[2]])]
#print(filtered_df.to_string())

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
