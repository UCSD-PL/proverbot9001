#!/usr/bin/env python3.7

from typing import List, Tuple
from difflib import SequenceMatcher

from models.tactic_predictor import (TacticContext, TacticPredictor, Prediction)
import serapi_instance

class ApplyLongestPredictor(TacticPredictor):
    def __init__(self) -> None:
        pass

    def predictKTactics(self, in_data : TacticContext, k : int) -> List[Prediction]:
        if len(in_data.hypotheses) == 0:
            return [Prediction("eauto", 0)]
        k = min(k, len(in_data.hypotheses))
        best_hyps = sorted(in_data.hypotheses, key=len, reverse=True)[:k]
        return [Prediction("apply " + serapi_instance.get_first_var_in_hyp(hyp) + ".",
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
                   SequenceMatcher(None, serapi_instance.get_hyp_type(hyp),
                                   in_data.goal).ratio()
            )[:k]
        return [Prediction("apply " + serapi_instance.get_first_var_in_hyp(hyp) + ".",
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
                   SequenceMatcher(None, serapi_instance.get_hyp_type(hyp),
                                   in_data.goal).ratio() * len(hyp)
            )[:k]
        return [Prediction("apply " + serapi_instance.get_first_var_in_hyp(hyp) + ".",
                           .5 ** idx) for idx, hyp in enumerate(best_hyps)]

    def _description(self) -> str:
        return "A predictor that tries to apply the most similar hypothesis to the goal"
