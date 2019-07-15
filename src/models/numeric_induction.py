from typing import List, Tuple
from models.tactic_predictor import (TacticContext, TacticPredictor, Prediction)

class NumericInductionPredictor(TacticPredictor):
    def __init__(self) -> None:
        pass

    def predictKTactics(self, in_data : TacticContext, k : int) -> List[Prediction]:
        return [Prediction("induction {}.".format(idx), .5 ** idx) for idx in range(1, k+1)]
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
        return "A predictor that tries to run induction 1, induction 2, etc."
