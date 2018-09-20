#!/usr/bin/env python3

import argparse
import time
from typing import Dict, List, Union, Any, Tuple, Iterable, cast, Callable

from models.encdecrnn_predictor import inputFromSentence
from tokenizer import Tokenizer, tokenizers, get_symbols
from models.tactic_predictor import TacticPredictor
from models.args import take_std_args, optimizers
from models.components import SimpleEmbedding
from context_filter import get_context_filter

from data import read_text_data, filter_data, RawDataset, make_keyword_tokenizer, Sentence
import serapi_instance

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim import Optimizer
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
import torch.utils.data as data
import torch.cuda

no_args_tactics = ["auto", "eauto", "constructor", "econstructor",
                   "simpl", "intro", "intros", "xomega", "simpl",
                   "intro", "intros"]
ind_star_args_tactics = ["induction", "einduction", "intros until"]
star_args_tactics = ["subst"]
one_arg_tactics = ["inv", "inversion", "exists", "induction",
                   "rewrite", "erewrite", "rewrite <-", "erewrite <-",
                   "apply", "eapply", "simpl in", "destruct"]

class EncStructPredictor(TacticPredictor):
    def load_saved_state(self, filename : str) -> None:
        pass
    def __init__(self, options : Dict[str, Any]) -> None:
        pass
    def predictDistribution(self, in_data : Dict[str, str]) -> torch.FloatTensor:
        pass
    def predictKTactics(self, in_data : Dict[str, str], k : int) \
        -> List[Tuple[str, float]]:
        pass
    def predictKTacticsWithLoss(self, in_data : Dict[str, str], k : int,
                                correct : str) -> Tuple[List[Tuple[str, float]], float]:
        pass
    def getOptions(self) -> List[Tuple[str, str]]:
        pass

Checkpoint = Tuple[Dict[Any, Any], float]

def train(dataset : Any,
          input_vocab_size : int, output_vocab_size : int, hidden_size : int,
          learning_rate : float, num_encoder_layers : int,
          max_length : int, num_epochs : int, batch_size : int,
          print_every : int, optimizer_f : Callable[..., Optimizer]) \
          -> Iterable[Checkpoint]:

    return

TacticStructure = Tuple[int, List[int]]
StructDataset = List[Tuple[List[Sentence], Sentence, TacticStructure]]

def encode_seq_structural_data(data : RawDataset,
                               context_tokenizer_type : \
                               Callable[[List[str], int], Tokenizer],
                               num_keywords : int,
                               num_reserved_tokens: int) -> \
                               Tuple[StructDataset, Tokenizer, SimpleEmbedding]:
    hyps_and_goals = [hyp_or_goal
                      for hyp_and_goal in [hyps + [goal] for hyps, goal, tactic in data]
                      for hyp_or_goal in hyp_and_goal]
    context_tokenizer = make_keyword_tokenizer(hyps_and_goals, context_tokenizer_type,
                                               num_keywords, num_reserved_tokens)
    embedding = SimpleEmbedding()

    encodedData = []
    for hyps, goal, tactic in data:
        stem, rest = serapi_instance.split_tactic(tactic)
        encodedData.append(([context_tokenizer.toTokenList(hyp) for hyp in hyps],
                            context_tokenizer.toTokenList(goal),
                            (embedding.encode_token(stem),
                            [hyp_index(hyps, arg) for arg in get_symbols(rest)])))

    return encodedData, context_tokenizer, embedding

# Returns *one* indexed hypothesis indices, zero for "not found"
def hyp_index(hyps : List[str], arg : str) -> int:
    for i, hyp in enumerate(hyps, start=1):
        if arg in hyp.split(":")[0]:
            return i
    return 0

def main(arg_list : List[str]) -> None:
    args = take_std_args(arg_list, "a structural pytorch model for proverbot")
    print("Reading dataset...")

    raw_data = read_text_data(args.scrape_file)
    print("Read {} raw input-output pairs".format(len(raw_data)))
    print("Filtering data based on predicate...")
    filtered_data = filter_data(raw_data, get_context_filter(args.context_filter))

    print("{} input-output pairs left".format(len(filtered_data)))
    print("Encoding data...")
    start = time.time()

    dataset, tokenizer, embedding = encode_seq_structural_data(filtered_data,
                                                               tokenizers[args.tokenizer],
                                                               args.num_keywords, 2)
    timeTaken = time.time() - start
    print("Encoded data in {:.2f}".format(timeTaken))

    checkpoints = train(dataset, tokenizer.numTokens(), embedding.num_tokens(),
                        args.hidden_size,
                        args.learning_rate, args.num_encoder_layers,
                        args.max_length, args.num_epochs, args.batch_size,
                        args.print_every, optimizers[args.optimizer])

    for epoch, (encoder_state, training_loss) in enumerate(checkpoints):
        state = {'epoch':epoch,
                 'training-loss': training_loss,
                 'tokenizer':tokenizer,
                 'tokenizer-name':args.tokenizer,
                 'optimizer':args.optimizer,
                 'learning-rate':args.learning_rate,
                 'embedding': embedding,
                 'neural-encoder':encoder_state,
                 'num-encoder-layers':args.num_encoder_layers,
                 'max-length': args.max_length,
                 'hidden-size' : args.hidden_size,
                 'num-keywords' : args.num_keywords,
                 'context-filter' : args.context_filter,
        }
        with open(args.save_file, 'wb') as f:
            print("=> Saving checkpoint at epoch {}".
                  format(epoch))
            torch.save(state, f)
