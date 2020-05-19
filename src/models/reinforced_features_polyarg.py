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
from typing import List, Tuple, Any, Dict, Optional
import argparse
import torch
from pathlib_revised import Path2

from models.tactic_predictor import TacticPredictor, Prediction
from models.components import NeuralPredictorState
from models import features_polyarg_predictor, features_q_estimator
from format import TacticContext

PickleableFPAMetadata = Any


class ReinforcedFeaturesPolyargPredictor(TacticPredictor):

    def __init__(self) -> None:
        self._fpa: Optional[
            features_polyarg_predictor.FeaturesPolyargPredictor
        ] = None
        self._estimator: Optional[
            features_q_estimator.FeaturesQEstimator
        ] = None

    def load_saved_state(self,
                         args: argparse.Namespace,
                         unparsed_args: List[str],
                         metadata: Tuple[
                             Any,
                             features_q_estimator.FeaturesQMetadata],
                         state: Tuple[NeuralPredictorState,
                                      Dict[str, Any]]) -> None:
        fpa_meta, q_meta = metadata
        fpa_state, q_state = state
        self._fpa = features_polyarg_predictor.FeaturesPolyargPredictor()
        self._fpa.load_saved_state(args, unparsed_args, fpa_meta,
                                   fpa_state)
        # The argument values here don't matter if we're not training,
        # so we set them to zero (which will hopefully break if we try
        # to train)
        self._estimator = features_q_estimator.FeaturesQEstimator(0, 0, 0)
        self._estimator.load_saved_state(args, unparsed_args, q_meta, q_state)
        pass

    def getOptions(self) -> List[Tuple[str, str]]:
        return []

    def predictKTactics(self, in_data: TacticContext, k: int) \
            -> List[Prediction]:
        assert self._fpa
        assert self._estimator
        inner_predictions = self._fpa.predictKTactics(in_data, 16)
        q_choices = zip(self._estimator(
            [(in_data, prediction.prediction)
             for prediction in inner_predictions]),
                        inner_predictions)
        ordered_actions = [p[1] for p in
                           sorted(q_choices,
                                  key=lambda q: q[0],
                                  reverse=True)]
        return ordered_actions[:k]

    def predictKTacticsWithLoss(self, in_data: TacticContext, k: int,
                                correct: str) -> \
        Tuple[List[Prediction], float]: pass

    def predictKTacticsWithLoss_batch(self,
                                      in_data: List[TacticContext],
                                      k: int, correct: List[str]) -> \
            Tuple[List[List[Prediction]], float]:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct weights for reinforced features polyarg")
    parser.add_argument("fpa_weights")
    parser.add_argument("q_weights")
    parser.add_argument("out_weights", type=Path2)
    args = parser.parse_args()

    fpa_name, fpa_saved = torch.load(args.fpa_weights)
    assert fpa_name == "polyarg", "Weights aren't  for an FPA predictor!"
    fpa_args, fpa_up_args, fpa_meta, fpa_state = \
        fpa_saved

    q_name, *q_saved = torch.load(args.q_weights)
    assert q_name == "features evaluator"
    q_args, q_up_args, q_meta, q_state = q_saved

    with args.out_weights.open('w') as f:
        torch.save(("refpa", (fpa_args, fpa_up_args, (fpa_meta, q_meta),
                             (fpa_state, q_state))),
                   f)

    pass


if __name__ == "__main__":
    main()
