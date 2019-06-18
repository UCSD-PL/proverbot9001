#!/usr/bin/env python3
import pdb
import signal
import sys
from tokenizer import tokenizers
import search_report
import dynamic_report
import static_report
import argparse
import data
import itertools
from predict_tactic import trainable_modules

from typing import Dict, Callable, List

def exit_early(signal, frame):
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, exit_early)
    parser = argparse.ArgumentParser(description=
                                     "Proverbot9001 toplevel. Used for training "
                                     "and making reports")
    parser.add_argument("command", choices=list(modules.keys()))
    args = parser.parse_args(sys.argv[1:2])
    modules[args.command](sys.argv[2:])

def train(args):
    parser = argparse.ArgumentParser(description=
                                     "Proverbot9001 training module")
    parser.add_argument("model", choices=list(trainable_modules.keys()))
    args_values = parser.parse_args(args[:1])
    trainable_modules[args_values.model](args[1:])

def get_data(args : List[str]) -> None:
    parser = argparse.ArgumentParser(description=
                                     "Parse datafiles into multiple formats")
    parser.add_argument("format", choices=["terms", "goals", "hyps+goal",
                                           "hyps+goal+tactic"])
    parser.add_argument("datafile_path", type=str)
    parser.add_argument("--tokenizer",
                        choices=list(tokenizers.keys()), type=str,
                        default=list(tokenizers.keys())[0])
    parser.add_argument("--max-tuples", dest="max_tuples", default=None, type=int)
    parser.add_argument("--num-keywords", dest="num_keywords", default=100, type=int)
    parser.add_argument("--max-length", dest="max_length", default=None, type=int)
    parser.add_argument("--lineend", dest="lineend", default=False, const=True,
                        action='store_const')
    parser.add_argument("--context-filter", dest="context_filter", default="default")
    arg_values = parser.parse_args(args)
    if arg_values.format == "terms":
        terms, tokenizer = data.term_data(
            data.RawDataset(list(itertools.islice(data.read_text_data(arg_values.datafile_path),
                                             arg_values.max_tuples))),
            tokenizers[arg_values.tokenizer],
            arg_values.num_keywords, 2)
        if arg_values.max_length:
            terms = [data.normalizeSentenceLength(term, arg_values.max_length)
                     for term in terms]
        for term in terms:
            print(tokenizer.toString(
                list(itertools.takewhile(lambda x: x != data.EOS_token, term))),
                  end="\\n\n" if arg_values.lineend else "\n")
    elif arg_values.format == "goals":
        dataset = data.get_text_data(arg_values)
        for prev_tactics, hyps, goal, tactic in dataset:
            print(goal)
    elif arg_values.format =="hyps+goal":
        dataset = data.get_text_data(arg_values)
        for prev_tactics, hyps, goal, tactic in dataset:
            for hyp in hyps:
                print(hyp)
            print("================================")
            print(goal)
    elif arg_values.format =="hyps+goal+tactic":
        dataset = data.get_text_data(arg_values)
        for prev_tactics, hyps, goal, tactic in dataset:
            for hyp in hyps:
                print(hyp)
            print("================================")
            print(goal)
            print("====> {}".format(tactic))
        pass

modules = {
    "train" : train,
    "search-report":search_report.main,
    "dynamic-report":dynamic_report.main,
    "static-report":static_report.main,
    "data": get_data,
}

if __name__ == "__main__":
    main()
