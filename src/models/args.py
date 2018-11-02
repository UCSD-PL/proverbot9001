#!/usr/bin/env python3

import argparse
from typing import List

from tokenizer import Tokenizer, tokenizers
from context_filter import get_context_filter

from torch import optim

optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
}
def add_std_args(parser : argparse.ArgumentParser) -> None:
    parser.add_argument("scrape_file")
    parser.add_argument("save_file")
    parser.add_argument("--num-epochs", dest="num_epochs", default=15, type=int)
    parser.add_argument("--batch-size", dest="batch_size", default=256, type=int)
    parser.add_argument("--max-length", dest="max_length", default=100, type=int)
    parser.add_argument("--print-every", dest="print_every", default=10, type=int)
    parser.add_argument("--hidden-size", dest="hidden_size", default=128, type=int)
    parser.add_argument("--learning-rate", dest="learning_rate",
                        default=.7, type=float)
    parser.add_argument("--num-encoder-layers", dest="num_encoder_layers",
                        default=3, type=int)
    parser.add_argument("--num-keywords", dest="num_keywords", default=100, type=int)
    parser.add_argument("--tokenizer",
                        choices=list(tokenizers.keys()), type=str,
                        default=list(tokenizers.keys())[0])
    parser.add_argument("--optimizer",
                        choices=list(optimizers.keys()), type=str,
                        default=list(optimizers.keys())[0])
    parser.add_argument("--context-filter", dest="context_filter",
                        type=str, default="default")

def start_std_args(args : List[str], description : str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_std_args(parser)
    return parser

def take_std_args(args : List[str], description : str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    add_std_args(parser)
    return parser.parse_args(args)
