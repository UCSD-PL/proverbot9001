from models.tactic_predictor import TacticContext
from tokenizer import get_symbols

import typing
from typing import List
from abc import ABCMeta, abstractmethod
from collections import Counter
import re
import argparse

class Feature(metaclass=ABCMeta):
    def __init__(self, init_dataset : List[TacticContext],
                 args : argparse.Namespace) -> None:
        pass
    @abstractmethod
    def __call__(self, context : TacticContext) -> List[float]:
        pass
    @abstractmethod
    def feature_size(self) -> int:
        pass

class ConstFeature(Feature):
    def __call__(self, context : TacticContext) -> List[float]:
        return [0.5]
    def feature_size(self):
        return 1
class NumEvarsInGoal(Feature):
    def __call__(self, context : TacticContext) -> List[float]:
        return [float(len(re.findall("\s\?\w", context.goal)))]
    def feature_size(self):
        return 1

class NumEqualitiesInHyps(Feature):
    def __call__(self, context : TacticContext) -> List[float]:
        return [float(sum([re.match("\w+ : eq ", hypothesis) != None
                           for hypothesis in context.hypotheses]))]
    def feature_size(self):
        return 1

class TopLevelTokenInGoal(Feature):
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
    def __call__(self, context : TacticContext) -> List[float]:
        onehotHeads = [0.] * len(self.headKeywords)
        headToken = get_symbols(context.goal)[0]
        # print("head token of {} is {}".format(context.goal, headToken))
        if headToken in self.headKeywords:
            onehotHeads[self.headKeywords.index(headToken)] = 1.0
        return onehotHeads
    def feature_size(self):
        return len(self.headKeywords)

def numUnboundIdentifiersInGoal(context : TacticContext) -> List[float]:
    pass

feature_constructors = [NumEqualitiesInHyps, NumEvarsInGoal,
                        TopLevelTokenInGoal]
