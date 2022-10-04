from collections import deque
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import time, argparse, random
from pathlib_revised import Path2
import dataloader
import coq_serapy as serapi_instance
from coq_serapy import load_commands, kill_comments, get_hyp_type, get_indexed_vars_dict, get_stem, split_tactic, contextSurjective
from coq_serapy.contexts import truncate_tactic_context, FullContext
from search_file import loadPredictorByFile
from search_strategies import completed_proof
from train_encoder import EncoderRNN, DecoderRNN, Lang, tensorFromSentence, EOS_token
from tokenizer import get_symbols, get_words,tokenizers
import pickle
import gym
import fasttext
import os, sys
from util import nostderr, unwrap, eprint, mybarfmt
from collections import defaultdict
from coqproofenv import ProofEnv
import tqdm




parser = argparse.ArgumentParser()
parser.add_argument("--proof_file", type=Path2)
parser.add_argument("--max-tuples", default=None, type=int)
parser.add_argument("--tokenizer",
                        choices=list(tokenizers.keys()), type=str,
                        default=list(tokenizers.keys())[0])
parser.add_argument("--num-keywords", default=100, type=int)
parser.add_argument("--lineend", action="store_true")
parser.add_argument('--wandb_log', action= 'store_true')
parser.add_argument('--weightsfile', default = "data/polyarg-weights.dat", type=Path2)
parser.add_argument("--max_term_length", type=int, default=256)
parser.add_argument("--max_attempts", type=int, default=5)
parser.add_argument('--prelude', default=".")
parser.add_argument('--run_name', type=str, default=None)
args = parser.parse_args()


env = ProofEnv(args.proof_file.path, args.prelude, args.wandb_log)

s = env.reset()


while True :
    print("Proof context : ")
    print("    Foreground goals :" )
    for i in env.coq.context.fg_goals :
        print("           Hypothesis : ", i.hypotheses)
        print("           Goals : ", i.goal)
    print("    Background goals :" )
    for i in env.coq.context.bg_goals :
        print("           Hypothesis : ", i.hypotheses)
        print("           Goals : ", i.goal)
    print("    Shelved goals :" )
    for i in env.coq.context.shelved_goals :
        print("           Hypothesis : ", i.hypotheses)
        print("           Goals : ", i.goal)
    print("    Given up goals :" )
    for i in env.coq.context.given_up_goals :
        print("           Hypothesis : ", i.hypotheses)
        print("           Goals : ", i.goal)       
    print("The tactic : ", env.coq.tactic)
    print()
    print()
    a = input()
    if a == "" :
        s = env.solve_curr_from_file()
    else :
        s = env.step(a)