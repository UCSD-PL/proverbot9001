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

from format import TacticContext
from tokenizer import get_symbols, limitNumTokens
from util import eprint
import serapi_instance

import typing
from typing import List, Dict, Set, Any
from abc import ABCMeta, abstractmethod
from collections import Counter
from difflib import SequenceMatcher
from pathlib_revised import Path2
import re
import argparse
import math
import torch

class Feature(metaclass=ABCMeta):
    def __init__(self, init_dataset : List[TacticContext],
                 args : argparse.Namespace) -> None:
        pass
    @classmethod
    def add_feature_arguments(self,
                              parser : argparse.ArgumentParser,
                              feature_args_set : Set[str],
                              default_values : Dict[str, Any] = {}) -> Set[str]:
        return set()
        pass


def maybe_add_argument(parser: argparse.ArgumentParser,
                       default_values: Dict[str, Any],
                       name: str, t: type,
                       default_value: Any,
                       feature_args_set: Set[str]) -> None:
    if name not in feature_args_set:
        parser.add_argument(f"--{name}", type=t,
                            default=default_values.get(name, default_value))

class VecFeature(Feature, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, context : TacticContext) -> List[float]:
        pass
    @abstractmethod
    def feature_size(self) -> int:
        pass

class WordFeature(Feature, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, context : TacticContext) -> int:
        pass
    @abstractmethod
    def vocab_size(self) -> int:
        pass

class ConstFeature(VecFeature):
    def __call__(self, context : TacticContext) -> List[float]:
        return [0.5]
    def feature_size(self) -> int:
        return 1
class ConstFeatureW(WordFeature):
    def __call__(self, context : TacticContext) -> int:
        return 0
    def vocab_size(self) -> int:
        return 1
class NumEvarsInGoal(VecFeature):
    def __call__(self, context : TacticContext) -> List[float]:
        return [float(len(re.findall("\s\?\w", context.goal)))]
    def feature_size(self) -> int:
        return 1

class NumEqualitiesInHyps(VecFeature):
    def __call__(self, context : TacticContext) -> List[float]:
        return [float(sum([re.match("\w+ : eq ", hypothesis) != None
                           for hypothesis in context.hypotheses]))]
    def feature_size(self) -> int:
        return 1

class TopLevelTokenInGoalV(VecFeature):
    def __init__(self, init_dataset : List[TacticContext],
                 args : argparse.Namespace) -> None:
        headTokenCounts : typing.Counter[str] = Counter()
        for relevant_lemmas, prev_tactics, hyps, goal in init_dataset:
            headToken = get_symbols(goal)[0]
            headTokenCounts[headToken] += 1
        if args.load_head_keywords and Path2(args.load_head_keywords).exists():
            self.headKeywords = torch.load(args.load_head_keywords)
        else:
            self.headKeywords = [word for word, count in
                                 headTokenCounts.most_common(args.num_head_keywords)]
        if args.save_head_keywords:
            torch.save(self.headKeywords, args.save_head_keywords)
        eprint("Head keywords are {}".format(self.headKeywords),
               guard=args.print_keywords)
    def __call__(self, context : TacticContext) -> List[float]:
        headToken = get_symbols(context.goal)[0]
        oneHotHeads = [0.] * len(self.headKeywords)
        if headToken in self.headKeywords:
            oneHotHeads[self.headKeywords.index(headToken)] = 1.
        return oneHotHeads
    def feature_size(self) -> int:
        return len(self.headKeywords)
    @classmethod
    def add_feature_arguments(self,
                              parser : argparse.ArgumentParser,
                              feature_args_set : Set[str],
                              default_values : Dict[str, Any] = {}) -> Set[str]:
        maybe_add_argument(parser, default_values, "num-head-keywords", int,
                           100, feature_args_set)
        maybe_add_argument(parser, default_values, "save-head-keywords", str,
                           "data/head-keywords.dat", feature_args_set)
        maybe_add_argument(parser, default_values, "load-head-keywords", str,
                           "data/head-keywords.dat", feature_args_set)
        return {"num-head-keywords", "save-head-keywords", "load-head-keywords"}
        pass
class TopLevelTokenInGoal(WordFeature):
    def __init__(self, init_dataset : List[TacticContext],
                 args : argparse.Namespace) -> None:
        headTokenCounts : typing.Counter[str] = Counter()
        for relevant_lemmas, prev_tactics, hyps, goal in init_dataset:
            headToken = get_symbols(goal)[0]
            headTokenCounts[headToken] += 1
        if args.load_head_keywords and Path2(args.load_head_keywords).exists():
            self.headKeywords = torch.load(args.load_head_keywords)
        else:
            self.headKeywords = [word for word, count in
                                 headTokenCounts.most_common(args.num_head_keywords)]
        if args.save_head_keywords:
            torch.save(self.headKeywords, args.save_head_keywords)
        eprint("Goal head keywords are {}".format(self.headKeywords),
               guard=args.print_keywords)
    def __call__(self, context : TacticContext) -> int:
        headToken = get_symbols(context.goal)[0]
        if headToken in self.headKeywords:
            return self.headKeywords.index(headToken) + 1
        else:
            return 0
    def vocab_size(self) -> int:
        return len(self.headKeywords) + 1
    @classmethod
    def add_feature_arguments(self,
                              parser : argparse.ArgumentParser,
                              feature_args_set : Set[str],
                              default_values : Dict[str, Any] = {}) -> Set[str]:
        maybe_add_argument(parser, default_values, "num-head-keywords", int,
                           100, feature_args_set)
        maybe_add_argument(parser, default_values, "save-head-keywords", str,
                           "data/head-keywords.dat", feature_args_set)
        maybe_add_argument(parser, default_values, "load-head-keywords", str,
                           "data/head-keywords.dat", feature_args_set)
        return {"num-head-keywords", "save-head-keywords", "load-head-keywords"}
        pass

class TopLevelTokenInBestHyp(WordFeature):
    def __init__(self, init_dataset : List[TacticContext],
                 args : argparse.Namespace) -> None:
        self.max_length = args.max_length
        headTokenCounts : typing.Counter[str] = Counter()
        for relevant_lemmas, prev_tactics, hyps, goal in init_dataset:
            for hyp in hyps:
                headToken = get_symbols(serapi_instance.get_hyp_type(hyp))[0]
                headTokenCounts[headToken] += 1
        if args.load_head_keywords and Path2(args.load_head_keywords).exists():
            self.headKeywords = torch.load(args.load_head_keywords)
        else:
            self.headKeywords = [word for word, count in
                                 headTokenCounts.most_common(args.num_head_keywords)]
        eprint("Hypothesis head keywords are {}".format(self.headKeywords),
               guard=args.print_keywords)
    def __call__(self, context : TacticContext) -> int:
        if len(context.hypotheses) == 0:
            return 0
        hyp_types = [limitNumTokens(serapi_instance.get_hyp_type(hyp),
                                    self.max_length)
                     for hyp in context.hypotheses]
        goal = limitNumTokens(context.goal, self.max_length)
        closest_hyp_type = max(hyp_types,
                               key=lambda x:
                               SequenceMatcher(None, goal, x).ratio() * len(get_symbols(x)))
        headToken = get_symbols(closest_hyp_type)[0]
        if headToken in self.headKeywords:
            return self.headKeywords.index(headToken) + 1
        else:
            return 0
    def vocab_size(self) -> int:
        return len(self.headKeywords) + 1
    @classmethod
    def add_feature_arguments(self,
                              parser : argparse.ArgumentParser,
                              feature_args_set : Set[str],
                              default_values : Dict[str, Any] = {}) -> Set[str]:
        maybe_add_argument(parser, default_values, "num-head-keywords", int,
                           100, feature_args_set)
        maybe_add_argument(parser, default_values, "save-head-keywords", str,
                           "data/head-keywords.dat", feature_args_set)
        maybe_add_argument(parser, default_values, "load-head-keywords", str,
                           "data/head-keywords.dat", feature_args_set)
        return {"num-head-keywords", "save-head-keywords", "load-head-keywords"}
        pass

class BestHypScore(VecFeature):
    def __init__(self, init_dataset : List[TacticContext],
                 args : argparse.Namespace) -> None:
        pass
    def __call__(self, context : TacticContext) -> List[float]:
        if len(context.hypotheses) == 0:
            return [0.]
        hyp_types = [serapi_instance.get_hyp_type(hyp)[:100] for hyp in context.hypotheses]
        best_hyp_score = max([SequenceMatcher(None, context.goal,hyp).ratio() * len(hyp)
                        for hyp in hyp_types])
        return [best_hyp_score / 100]
    def feature_size(self) -> int:
        return 1

class PrevTacticV(VecFeature):
    def __init__(self, init_dataset : List[TacticContext],
                 args : argparse.Namespace) -> None:
        prevTacticsCounts : typing.Counter[str] = Counter()
        for relevant_lemmas, prev_tactics, hyps, goal in init_dataset:
            if len(prev_tactics) > 2:
                prevTacticsCounts[serapi_instance.get_stem(prev_tactics[-1])] += 1
        if args.load_tactic_keywords and Path2(args.load_tactic_keywords).exists():
            self.tacticKeywords = torch.load(args.load_tactic_keywords)
        else:
            self.tacticKeywords = ["Proof"] + \
                [word for word, count in
                 prevTacticsCounts.most_common(args.num_tactic_keywords)]
        eprint("Tactic keywords are {}".format(self.tacticKeywords),
               guard=args.print_keywords)
    def __call__(self, context : TacticContext) -> List[float]:
        prev_tactic = (serapi_instance.get_stem(context.prev_tactics[-1]) if
                       len(context.prev_tactics) > 1 else "Proof")
        oneHotPrevs= [0.] * len(self.tacticKeywords)
        if prev_tactic in self.tacticKeywords:
            oneHotPrevs[self.tacticKeywords.index(prev_tactic)] = 1.
        return oneHotPrevs
    def feature_size(self) -> int:
        return len(self.tacticKeywords)
    @classmethod
    def add_feature_arguments(self,
                              parser : argparse.ArgumentParser,
                              feature_args_set : Set[str],
                              default_values : Dict[str, Any] = {}) -> Set[str]:
        maybe_add_argument(parser, default_values, "num-tactic-keywords", int,
                           50, feature_args_set)
        maybe_add_argument(parser, default_values, "save-tactic-keywords", str,
                           "data/tactic-keywords.dat", feature_args_set)
        maybe_add_argument(parser, default_values, "load-tactic-keywords", str,
                           "data/tactic-keywords.dat", feature_args_set)
        return {"num-tactic-keywords", "save-tactic-keywords", "load-tactic-keywords"}
class PrevTactic(WordFeature):
    def __init__(self, init_dataset : List[TacticContext],
                 args : argparse.Namespace) -> None:
        prevTacticsCounts : typing.Counter[str] = Counter()
        for relevant_lemmas, prev_tactics, hyps, goal in init_dataset:
            if len(prev_tactics) > 2:
                prevTacticsCounts[serapi_instance.get_stem(prev_tactics[-1])] += 1
        if args.load_tactic_keywords and Path2(args.load_tactic_keywords).exists():
            self.tacticKeywords = torch.load(args.load_tactic_keywords)
        else:
            self.tacticKeywords = ["Proof"] + \
                [word for word, count in
                 prevTacticsCounts.most_common(args.num_tactic_keywords)]
        eprint("Tactic keywords are {}".format(self.tacticKeywords),
               guard=args.print_keywords)
    def __call__(self, context : TacticContext) -> int:
        prev_tactic = (serapi_instance.get_stem(context.prev_tactics[-1]) if
                       len(context.prev_tactics) > 1 else "Proof")
        if prev_tactic in self.tacticKeywords:
            return self.tacticKeywords.index(prev_tactic) + 1
        else:
            return 0
    def vocab_size(self) -> int:
        return len(self.tacticKeywords) + 1
    @classmethod
    def add_feature_arguments(self,
                              parser : argparse.ArgumentParser,
                              feature_args_set : Set[str],
                              default_values : Dict[str, Any] = {}) -> Set[str]:
        maybe_add_argument(parser, default_values, "num-tactic-keywords", int,
                           50, feature_args_set)
        maybe_add_argument(parser, default_values, "save-tactic-keywords", str,
                           "data/tactic-keywords.dat", feature_args_set)
        maybe_add_argument(parser, default_values, "load-tactic-keywords", str,
                           "data/tactic-keywords.dat", feature_args_set)
        return {"num-tactic-keywords", "save-tactic-keywords", "load-tactic-keywords"}
        pass

class NumUnboundIdentifiersInGoal(VecFeature):
    def __call__(self, context : TacticContext) -> List[float]:
        identifiers = get_symbols(context.goal)
        locallyBoundInHyps = serapi_instance.get_vars_in_hyps(context.hypotheses)
        binders = ["forall\s+(.*)(?::.*)?,",
                   "fun\s+(.*)(?::.*)?,",
                   "let\s+\S+\s+:="]
        punctuation = ["(", ")", ":", ",", "_", ":=", "=>", "{|", "|}"]
        locallyBoundInTerm = [var
                              for binder_pattern in binders
                              for varString in re.findall(binder_pattern, context.goal)
                              for var in re.findall("\((\S+)\s+:", varString)
                              if var not in punctuation]
        globallyBoundIdentifiers = \
            [ident for ident in identifiers
             if not ident in locallyBoundInHyps + locallyBoundInTerm + punctuation]
        locallyBoundIdentifiers = [ident for ident in identifiers
                                   if not ident in globallyBoundIdentifiers + punctuation]
        for var in locallyBoundInTerm:
            assert var in locallyBoundIdentifiers, \
                "{}, {}".format(globallyBoundIdentifiers, locallyBoundInTerm)
            locallyBoundIdentifiers.remove(var)
        return [math.log1p(float(len(locallyBoundIdentifiers))) ,
                # math.log1p(float(len(globallyBoundIdentifiers))),
                float(len(globallyBoundIdentifiers)) /
                float(len(globallyBoundIdentifiers) + len(locallyBoundIdentifiers))]
    def feature_size(self) -> int:
        return 2

class NumHypotheses(VecFeature):
    def __call__(self, context : TacticContext) -> List[float]:
        return [math.log1p(float(len(context.hypotheses)))]
    def feature_size(self) -> int:
        return 1

class HasFalseToken(VecFeature):
    def __call__(self, context : TacticContext) -> List[float]:
        goalHasFalse = re.match("\bFalse\b", context.goal)
        hypsHaveFalse = False
        for hyp in context.hypotheses:
            if re.match("\bFalse\b", hyp):
                hypsHaveFalse = True
                break
        return [float(bool(goalHasFalse)), float(bool(hypsHaveFalse))]
    def feature_size(self) -> int:
        return 2

vec_feature_constructors = [
    # HasFalseToken,
    # NumHypotheses,
    # NumUnboundIdentifiersInGoal,
    # NumEqualitiesInHyps,
    # NumEvarsInGoal,
    BestHypScore,
    # ConstFeature,
]

word_feature_constructors = [
    PrevTactic,
    TopLevelTokenInGoal,
    TopLevelTokenInBestHyp,
    # ConstFeatureW,
]

feature_constructors = vec_feature_constructors
