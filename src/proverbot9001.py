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
from collections import Counter
from tokenizer import tokenizers
import search_file
import dynamic_report
import static_report
import evaluator_report
import argparse
import data
import itertools
import coq_serapy
import features
import json
from util import eprint, print_time
from models.components import SimpleEmbedding
import predict_tactic 
import evaluate_state
import interactive_predictor
from pathlib import Path
from pathlib_revised import Path2
import dataloader
from models.tactic_predictor import Prediction, TacticPredictor
from search_worker import ReportJob, SearchWorker, get_files_jobs, get_predictor, get_random_predictor, project_dicts_from_args, get_predictor_by_path
import pandas as pd
from predict_tactic import loadPredictorByFile
from coq_serapy.contexts import TacticContext, FullContext, ProofContext, truncate_tactic_context, strip_scraped_output

from models.tactic_predictor import TrainablePredictor


from typing import List


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
    arg_values = parser.parse_args(args[:1])
    if arg_values.model in predict_tactic.trainable_modules.keys():
        predict_tactic.loadTrainablePredictor(arg_values.model)(args[1:])
    elif arg_values.model in evaluate_state.trainable_modules.keys():
        evaluate_state.trainable_modules[arg_values.model](args[1:])
    else:
        assert False, f"Couldn't find the module {arg_values.model} to train!"

def get_data(args : List[str]) -> None:
    parser = argparse.ArgumentParser(description=
                                     "Parse datafiles into multiple formats")
    parser.add_argument("format", choices=["terms", "goals", "hyps+goal",
                                           "hyps+goal+tactic", "tacvector",
                                           "scrapefile-rd", "scrapefile"])
    parser.add_argument("scrape_file", type=Path)
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
    parser.add_argument("--no-truncate-semicolons", dest="truncate_semicolons",
                        action='store_false')
    parser.add_argument("--max-length", dest="max_length", default=30, type=int)
    parser.add_argument("--lineend", action="store_true")
    parser.add_argument("--context-filter", default="default")
    parser.add_argument('-v', "--verbose", action="count")
    parser.add_argument("--num-threads", "-j", type=int, default=None)
    parser.add_argument("--no-use-substitutions", action='store_false',
                        dest='use_substitutions')
    parser.add_argument("--no-normalize-numeric-args", action='store_false',
                        dest='normalize_numeric_args')
    parser.add_argument("--sort", action='store_true')
    arg_values = parser.parse_args(args)
    if arg_values.format == "terms":
        scraped_tactics = dataloader.scraped_tactics_from_file(str(arg_values.scrape_file),
                                                               arg_values.context_filter,
                                                               arg_values.max_length,
                                                               arg_values.max_tuples)
        terms = [term
                 for scraped in scraped_tactics if len(scraped.context.fg_goals) > 0
                 for term in [coq_serapy.get_hyp_type(premise) for premise in
                  scraped.relevant_lemmas + scraped.context.fg_goals[0].hypotheses]
                 + [scraped.context.fg_goals[0].goal]]
        for term in terms:
            print(term, end="\\n\n" if arg_values.lineend else "\n")
    else:
        dataset = data.get_text_data(arg_values)
        if arg_values.sort:
            dataset = data.RawDataset(sorted(dataset, key=lambda d: len(d.hypotheses), reverse=True))
        if arg_values.format == "goals":
            for relevant_lemmas, prev_tactics, hyps, goal, tactic in dataset:
                print(goal)
        elif arg_values.format == "hyps+goal":
            for relevant_lemmas, prev_tactics, hyps, goal, tactic in dataset:
                for hyp in hyps:
                    print(hyp)
                print("================================")
                print(goal)
        elif arg_values.format == "hyps+goal+tactic":
            for relevant_lemmas, prev_tactics, hyps, goal, tactic in dataset:
                for hyp in hyps:
                    print(hyp)
                print("================================")
                print(goal)
                print("====> {}".format(tactic))
            pass
        elif arg_values.format == "tacvector":
            embedding = SimpleEmbedding()
            eprint("Encoding tactics...", guard=arg_values.verbose)
            answers = [embedding.encode_token(coq_serapy.get_stem(datum.tactic))
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
        elif arg_values.format == "scrapefile-rd":
            for point in dataset:
                print(json.dumps({"relevant_lemmas": point.relevant_lemmas,
                                  "prev_tactics": point.prev_tactics,
                                  "context": {"fg_goals":
                                              [{"hypotheses": point.hypotheses,
                                                "goal": point.goal}],
                                              "bg_goals": [],
                                              "shelved_goals": [],
                                              "given_up_goals": []},
                                  "tactic": point.tactic}))
        elif arg_values.format == "scrapefile":
            for point in dataset:
                print(json.dumps({"relevant_lemmas": point.relevant_lemmas,
                                  "prev_tactics": point.prev_tactics,
                                  "prev_hyps": point.hypotheses,
                                  "prev_goal": point.goal,
                                  "tactic": point.tactic}))

import random
import contextlib
from pathlib_revised import Path2
from tokenizer import get_relevant_k_keywords2

def get_tactics(args: List[str]):
    parser = argparse.ArgumentParser(description="Pick a set of common tactics")
    parser.add_argument("-v", "--verbose", action='count', default=0)
    parser.add_argument("-n", "--num-tactics", type=int, default=128)
    parser.add_argument("-j", "--num-threads", default=None, type=int)
    parser.add_argument("--no-truncate-semicolons", dest="truncate_semicolons",
                        action='store_false')
    parser.add_argument("--no-use-substitutions", action='store_false',
                        dest='use_substitutions')
    parser.add_argument("--no-normalize-numeric-args", action='store_false',
                        dest='normalize_numeric_args')
    parser.add_argument("--context-filter", default="default")
    parser.add_argument("--max-tuples", dest="max_tuples", default=None, type=int)
    parser.add_argument("--max-term-length", default=30, type=int)
    parser.add_argument("scrape_file", type=Path)
    parser.add_argument("dest")
    arg_values = parser.parse_args(args)

    with print_time("Getting data"):
        # raw_data = list(data.get_text_data(arg_values))
        raw_data = dataloader.scraped_tactics_from_file(str(arg_values.scrape_file),
                                                        arg_values.context_filter,
                                                        arg_values.max_term_length,
                                                        arg_values.max_tuples)

    count: Counter[str] = Counter()
    for scraped in raw_data:
        stem = coq_serapy.get_stem(scraped.tactic)
        if stem != "":
            count[stem] += 1
    with (open(arg_values.dest, mode='w') if arg_values.dest != "-"
          else contextlib.nullcontext(sys.stdout)) as f:
        for tactic, tac_count in count.most_common(arg_values.num_tactics):
            f.write(tactic + "\n")

def get_tokens(args: List[str]):
    parser = argparse.ArgumentParser(description="Pick a set of tokens")
    parser.add_argument("--type", choices=["mixed"], default="mixed")
    parser.add_argument("-v", "--verbose", action='count', default=0)
    parser.add_argument("-n", "--num-keywords", type=int, default=120)
    parser.add_argument("-s", "--num-samples", type=int, default=2000)
    parser.add_argument("-j", "--num-threads", type=int, default=None)
    parser.add_argument("--context-filter", default="default")
    parser.add_argument("--max-term-length", default=30, type=int)
    parser.add_argument("scrapefile", type=Path)
    parser.add_argument("dest")
    arg_values = parser.parse_args(args)

    with print_time("Reading scraped data", guard=arg_values.verbose):
        raw_data = dataloader.scraped_tactics_from_file(str(arg_values.scrapefile),
                                                        arg_values.context_filter,
                                                        arg_values.max_term_length,
                                                        None)
    embedding = SimpleEmbedding()
    subset = random.sample(raw_data, arg_values.num_samples)
    relevance_pairs = [(scraped.context.fg_goals[0].goal,
                        embedding.encode_token(
                            coq_serapy.get_stem(scraped.tactic)))
                       for scraped in subset]
    with print_time("Calculating keywords", guard=arg_values.verbose):
        keywords = get_relevant_k_keywords2(relevance_pairs,
                                            arg_values.num_keywords,
                                            arg_values.num_threads)

    with (open(arg_values.dest, mode='w') if arg_values.dest != "-"
          else contextlib.nullcontext(sys.stdout)) as f:
        for keyword in keywords:
            f.write(keyword + "\n")

def predictor_data(args: List[str], **kwargs):
    parser = argparse.ArgumentParser(description="Get correctness training data for a predictor")
    parser.add_argument("--type", choices=["mixed"], default="mixed")
    parser.add_argument("-v", "--verbose", action='count', default=0)
    parser.add_argument("-n", "--num-keywords", type=int, default=120)
    parser.add_argument("-s", "--num-samples", type=int, default=2000)
    parser.add_argument("-j", "--num-threads", type=int, default=None)
    parser.add_argument("--context-filter", default="default")
    parser.add_argument("--max-attempts", default=10)
    parser.add_argument("--no-prev-tactic", action='store_true')
    parser.add_argument("--no-goal-head", action='store_true')
    parser.add_argument("--no-hyp-head", action='store_true')
    parser.add_argument("--no-hyp-scores", action='store_true')
    parser.add_argument("--blacklisted-tactics", default=[])
    parser.add_argument("--max-term-length", default=30, type=int)
    parser.add_argument("--weightsfile", type=str)
    parser.add_argument("--weightsfiles",  nargs='+', type=str)
    parser.add_argument("--scrapefile", type=Path)
    parser.add_argument("--dest")
    arg_values = parser.parse_args(args)

    my_predictor = get_predictor(arg_values)

    with print_time("Reading scraped data", guard=arg_values.verbose):
        raw_data = dataloader.scraped_tactics_from_file(str(arg_values.scrapefile),
                                                        arg_values.context_filter,
                                                        arg_values.max_term_length,
                                                        None)
    predictor_list = []
    for weightsfile in arg_values.weightsfiles:
        temp_predictor = loadPredictorByFile(weightsfile)
        predictor_list.append(temp_predictor)

    print(predictor_list)

    i = 0
    with open("howmany.txt","r") as f: 
        for line in raw_data:
            correct_tactic = line.tactic.strip()
            correct_tactic = ' '.join(correct_tactic.split())
            relevant_lemmas =  line.relevant_lemmas
            prev_tactics = line.prev_tactics
            hypotheses = line.context.fg_goals[0].hypotheses
            goal = line.context.fg_goals[0].goal
            #goal = ' '.join(line.context.fg_goals[0].goal.split())
            the_context = TacticContext(relevant_lemmas, prev_tactics, hypotheses, goal)
            fingoal = ' '.join(goal.split())
            data = {'goal': fingoal}

            predictor_num = 0
            for predictor in predictor_list:
                tactics = predictor.predictKTactics(arg_values, truncate_tactic_context(the_context, arg_values.max_term_length), arg_values.max_attempts,blacklist=arg_values.blacklisted_tactics)
                predicted_tactic_list = [' '.join(the_tactic.prediction.split()) for the_tactic in tactics]

                rank = 0
                flag = False
                for a_tactic in predicted_tactic_list:
                    if correct_tactic == a_tactic:
                        final_rank = ((10 - rank)/10)
                        flag = True
                        break
                    rank = rank + 1
                if not flag:
                    final_rank = 0.0
                data[predictor_num] = final_rank
                predictor_num = predictor_num + 1
       
            data = pd.Series(data)
            if not i == 0:
                predictor_list_dataframe = pd.concat([predictor_list_dataframe, data.to_frame().T], axis=0)
            else:
                predictor_list_dataframe = pd.DataFrame.from_dict(data).T
            i = i + 1
            f.write(i)
            #if i > 10000:
            #    break
        fulldest = arg_values.dest
        predictor_list_dataframe.to_csv(fulldest)
        f.close()

modules = {
    "train": train,
    "search-report": search_file.main,
    "dynamic-report": dynamic_report.main,
    "static-report": static_report.main,
    "evaluator-report": evaluator_report.main,
    "data": get_data,
    "tokens": get_tokens,
    "tactics": get_tactics,
    "predict": interactive_predictor.predict,
    "predictor-data": predictor_data,
}

if __name__ == "__main__":
    main()
