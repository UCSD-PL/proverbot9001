#!/usr/bin/env python3.7
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

from typing import List, Tuple
from difflib import SequenceMatcher

from models.tactic_predictor import (TacticPredictor, Prediction)
from coq_serapy.contexts import TacticContext
import coq_serapy as coq_serapy
import tokenizer

class ApplyLongestPredictor(TacticPredictor):
    def __init__(self) -> None:
        pass

    def predictKTactics(self, in_data : TacticContext, k : int) -> List[Prediction]:
        if len(in_data.hypotheses) == 0:
            return [Prediction("eauto", 0)]
        k = min(k, len(in_data.hypotheses))
        best_hyps = sorted(in_data.hypotheses, key=len, reverse=True)[:k]
        return [Prediction("apply " + coq_serapy.get_first_var_in_hyp(hyp) + ".",
                           .5 ** idx) for idx, hyp in enumerate(best_hyps)]
    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) \
        -> Tuple[List[Prediction], float]:
        return self.predictKTactics(in_data, k), 0.0

    def predictKTacticsWithLoss_batch(self, in_datas : List[TacticContext],
                                      k : int, corrects : List[str]) \
        -> Tuple[List[List[Prediction]], float] :
        predictions = [self.predictKTactics(in_data, k) for in_data in in_datas]
        return predictions, 0.0
    def getOptions(self) -> List[Tuple[str, str]]:
        return []
    def _description(self) -> str:
        return "A predictor that tries to apply the longest hypothesis to the goal"

class ApplyStringSimilarPredictor(TacticPredictor):
    def __init__(self) -> None:
        pass

    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) \
        -> Tuple[List[Prediction], float]:
        return self.predictKTactics(in_data, k), 0.0

    def predictKTacticsWithLoss_batch(self, in_datas : List[TacticContext],
                                      k : int, corrects : List[str]) \
        -> Tuple[List[List[Prediction]], float] :
        predictions = [self.predictKTactics(in_data, k) for in_data in in_datas]
        return predictions, 0.0
    def getOptions(self) -> List[Tuple[str, str]]:
        return []

    def predictKTactics(self, in_data : TacticContext, k : int) -> List[Prediction]:
        if len(in_data.hypotheses) == 0:
            return [Prediction("eauto", 0)]
        k = min(k, len(in_data.hypotheses))
        best_hyps = \
            sorted(in_data.hypotheses,
                   reverse=True,
                   key=lambda hyp:
                   SequenceMatcher(None, coq_serapy.get_hyp_type(hyp),
                                   in_data.goal).ratio()
            )[:k]
        return [Prediction("apply " + coq_serapy.get_first_var_in_hyp(hyp) + ".",
                           .5 ** idx) for idx, hyp in enumerate(best_hyps)]

    def _description(self) -> str:
        return "A predictor that tries to apply the most similar hypothesis to the goal"

class ApplyNormalizedSimilarPredictor(TacticPredictor):
    def __init__(self) -> None:
        pass

    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) \
        -> Tuple[List[Prediction], float]:
        return self.predictKTactics(in_data, k), 0.0

    def predictKTacticsWithLoss_batch(self, in_datas : List[TacticContext],
                                      k : int, corrects : List[str]) \
        -> Tuple[List[List[Prediction]], float] :
        predictions = [self.predictKTactics(in_data, k) for in_data in in_datas]
        return predictions, 0.0
    def getOptions(self) -> List[Tuple[str, str]]:
        return []

    def predictKTactics(self, in_data : TacticContext, k : int) -> List[Prediction]:
        if len(in_data.hypotheses) == 0:
            return [Prediction("eauto", 0)]
        k = min(k, len(in_data.hypotheses))
        best_hyps = \
            sorted(in_data.hypotheses,
                   reverse=True,
                   key=lambda hyp:
                   SequenceMatcher(None, coq_serapy.get_hyp_type(hyp),
                                   in_data.goal).ratio() * len(hyp)
            )[:k]
        return [Prediction("apply " + coq_serapy.get_first_var_in_hyp(hyp) + ".",
                           .5 ** idx) for idx, hyp in enumerate(best_hyps)]

    def _description(self) -> str:
        return "A predictor that tries to apply the most similar hypothesis to the goal"

class ApplyWordSimlarPredictor(TacticPredictor):
    def __init__(self) -> None:
        pass
    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) \
        -> Tuple[List[Prediction], float]:
        return self.predictKTactics(in_data, k), 0.0

    def predictKTacticsWithLoss_batch(self, in_datas : List[TacticContext],
                                      k : int, corrects : List[str]) \
        -> Tuple[List[List[Prediction]], float] :
        predictions = [self.predictKTactics(in_data, k) for in_data in in_datas]
        return predictions, 0.0
    def getOptions(self) -> List[Tuple[str, str]]:
        return []
    def predictKTactics(self, in_data : TacticContext, k : int) -> List[Prediction]:
        if len(in_data.hypotheses) == 0:
            return [Prediction("eauto", 0)]
        k = min(k, len(in_data.hypotheses))
        best_hyps = \
            sorted(in_data.hypotheses,
                   reverse=True,
                   key=lambda hyp:
                   SequenceMatcher(None,
                                   tokenizer.get_symbols(
                                       coq_serapy.get_hyp_type(hyp)),
                                   in_data.goal).ratio() * len(hyp)
            )[:k]
        return [Prediction("apply " + coq_serapy.get_first_var_in_hyp(hyp) + ".",
                           .5 ** idx) for idx, hyp in enumerate(best_hyps)]
    def _description(self) -> str:
        return "A predictor that tries to apply the most similar hypothesis to the goal"
