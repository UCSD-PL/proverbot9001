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
from coq_serapy import load_commands, kill_comments, get_hyp_type, get_indexed_vars_dict, get_stem, split_tactic, contextSurjective, summarizeContext
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


proof_file = args.proof_file.path
commands = load_commands(proof_file, progress_bar=True)
coq = serapi_instance.SerapiInstance(['sertop', '--implicit'],None, prelude = args.prelude)
coq.verbose = 3
coq.quiet = True


def is_same_context(context1, context2) :
    print("Context Surjectives")
    # print(contextSurjective(context2, context1))
    return contextSurjective(context1, context2) and contextSurjective(context2, context1)

    
for command in commands :
    if not "Lemma store_init_data_list_aligned:\n  forall b il m p m',\n  store_init_data_list ge m b p il = Some m' ->\n  init_data_list_aligned p il" in command :
        coq.run_stmt(command, timeout= 100)
    else :
        coq.run_stmt(command, timeout= 100)
        # quit()
        new_commands = ['Proof.', 'induction b.', '{', 'simpl.', 'try congruence.', 'intros.', 'exploit store_init_data_list_unchanged.', '{', 'eexact H.', '}', '{', 'simpl.', 'eauto.', 'intros.', 'simpl.', 'red.', 'eauto.', 'eauto.', 'eauto.', 'eauto.', 'intros.', 'inv H.','simpl in H3.']
        contexts = []
        for new_command in new_commands :
            coq.run_stmt(new_command.lstrip().rstrip(), timeout= 100)
            contexts.append(coq.proof_context)
        
        coq.cancel_last()
        contexts.append(coq.proof_context)
        print(is_same_context(contexts[-1], contexts[-3]))
        # print(is_same_context(contexts[-1], contexts[-2]))
        
        quit()
    