from models.tactic_predictor import TacticContext
from tokenizer import get_symbols
import serapi_instance

import typing
from typing import List
from abc import ABCMeta, abstractmethod
from collections import Counter
import re
import argparse
import math

class Feature(metaclass=ABCMeta):
    def __init__(self, init_dataset : List[TacticContext],
                 args : argparse.Namespace) -> None:
        pass
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

class TopLevelTokenInGoal(WordFeature):
    def __init__(self, init_dataset : List[TacticContext],
                 args : argparse.Namespace) -> None:
        headTokenCounts : typing.Counter[str] = Counter()
        for prev_tactics, hyps, goal in init_dataset:
            headToken = get_symbols(goal)[0]
            headTokenCounts[headToken] += 1
        self.headKeywords = [word for word, count in
                             headTokenCounts.most_common(args.num_head_keywords)]
        if args.print_keywords:
            print("Head keywords are {}".format(self.headKeywords))
    def __call__(self, context : TacticContext) -> int:
        headToken = get_symbols(context.goal)[0]
        if headToken in self.headKeywords:
            return self.headKeywords.index(headToken) + 1
        else:
            return 0
    def vocab_size(self) -> int:
        return len(self.headKeywords) + 1

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
        return [# math.log1p(
            float(len(locallyBoundIdentifiers))# )
    ,
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
    HasFalseToken,
    NumHypotheses,
    NumUnboundIdentifiersInGoal,
    NumEqualitiesInHyps,
    NumEvarsInGoal,
    # ConstFeature,
]

word_feature_constructors = [
    TopLevelTokenInGoal,
    # ConstFeatureW,
]

feature_constructors = vec_feature_constructors
