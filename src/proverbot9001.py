#!/usr/bin/env python3
##########################################################################
#
#    This file is part of Proverbot9001.
#
#    Proverbot9001 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Proverbot9001 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Proverbot9001.  If not, see <https://www.gnu.org/licenses/>.
#
#    Copyright 2019 Alex Sanchez-Stern and Yousef Alhessi
#
##########################################################################

import signal
import sys
from tokenizer import tokenizers
import search_report
import dynamic_report
import static_report
import evaluator_report
import argparse
import data
import itertools
import serapi_instance
import features
from util import eprint
from format import strip_scraped_output
from models.components import SimpleEmbedding
import predict_tactic
import evaluate_state
from pathlib_revised import Path2

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
    parser.add_argument("model", choices=list(predict_tactic.trainable_modules.keys()) +
                        list(evaluate_state.trainable_modules.keys()))
    args_values = parser.parse_args(args[:1])
    module = predict_tactic.trainable_modules.get(args_values.model)
    if not module:
        module = evaluate_state.trainable_modules[args_values.model]
    module(args[1:])

def get_data(args : List[str]) -> None:
    parser = argparse.ArgumentParser(description=
                                     "Parse datafiles into multiple formats")
    parser.add_argument("format", choices=["terms", "goals", "hyps+goal",
                                           "hyps+goal+tactic", "tacvector"])
    parser.add_argument("scrape_file", type=Path2)
    parser.add_argument("--tokenizer",
                        choices=list(tokenizers.keys()), type=str,
                        default=list(tokenizers.keys())[0])
    parser.add_argument("--max-tuples", dest="max_tuples", default=None, type=int)
    parser.add_argument("--num-keywords", dest="num_keywords", default=100, type=int)
    parser.add_argument("--num-head-keywords", dest="num_head_keywords", type=int,
                        default=100)
    parser.add_argument("--num-tactic-keywords", dest="num_tactic_keywords", type=int,
                        default=50)
    parser.add_argument("--print-keywords", dest="print_keywords", action='store_true')
    parser.add_argument("--no-truncate_semicolons", dest="truncate_semicolons",
                        action='store_false')
    parser.add_argument("--max-length", dest="max_length", default=None, type=int)
    parser.add_argument("--lineend", dest="lineend", default=False, const=True,
                        action='store_const')
    parser.add_argument("--context-filter", dest="context_filter", default="default")
    parser.add_argument("--verbose", action="store_true")
    arg_values = parser.parse_args(args)
    if arg_values.format == "terms":
        terms, tokenizer = data.term_data(
            data.RawDataset(list(itertools.islice(data.read_text_data(arg_values.scrape_file),
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
        for relevant_lemmas, prev_tactics, hyps, goal, tactic in dataset:
            print(goal)
    elif arg_values.format =="hyps+goal":
        dataset = data.get_text_data(arg_values)
        for relevant_lemmas, prev_tactics, hyps, goal, tactic in dataset:
            for hyp in hyps:
                print(hyp)
            print("================================")
            print(goal)
    elif arg_values.format =="hyps+goal+tactic":
        dataset = data.get_text_data(arg_values)
        for relevant_lemmas, prev_tactics, hyps, goal, tactic in dataset:
            for hyp in hyps:
                print(hyp)
            print("================================")
            print(goal)
            print("====> {}".format(tactic))
        pass
    elif arg_values.format == "tacvector":
        dataset = data.get_text_data(arg_values)
        embedding = SimpleEmbedding()
        eprint("Encoding tactics...", guard=arg_values.verbose)
        answers = [embedding.encode_token(serapi_instance.get_stem(datum.tactic))
                   for datum in dataset]
        stripped_data = [strip_scraped_output(scraped) for scraped in dataset]
        eprint("Constructing features...", guard=arg_values.verbose)
        word_feature_functions = [word_feature_constructor(stripped_data, arg_values) # type: ignore
                                  for word_feature_constructor in features.word_feature_constructors]
        vec_features_functions = [vec_feature_constructor(stripped_data, arg_values)
                                  for vec_feature_constructor in features.vec_feature_constructors]
        eprint("Extracting features...", guard=arg_values.verbose)
        word_features = [[feature(c) for feature in word_feature_functions]
                         for c in stripped_data]
        vec_features = [[feature_val for feature in
                         vec_features_functions
                         for feature_val in feature(c)]
                        for c in stripped_data]
        eprint("Done", guard=arg_values.verbose)
        for word_feat, vec_feat, tactic in zip(word_features, vec_features, answers):
            print(",".join(list(map(str, word_feat)) + list(map(str, vec_feat))
                           + [str(tactic)]))

modules = {
    "train" : train,
    "search-report":search_report.main,
    "dynamic-report":dynamic_report.main,
    "static-report":static_report.main,
    "evaluator-report":evaluator_report.main,
    "data": get_data,
}

if __name__ == "__main__":
    main()
