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

import argparse

from typing import List, Tuple, Any, Dict, Optional

from models.tactic_predictor import TacticPredictor, Prediction
from models.components import NeuralPredictorState
from models import (features_polyarg_predictor, q_estimator,
                    features_q_estimator, polyarg_q_estimator)
from coq_serapy.contexts import TacticContext

# from util import eprint

PickleableFPAMetadata = Any


class ReinforcedFeaturesPolyargPredictor(TacticPredictor):

    def __init__(self) -> None:
        self._fpa: Optional[
            features_polyarg_predictor.FeaturesPolyargPredictor
        ] = None
        self._estimator: Optional[
            q_estimator.QEstimator
        ] = None

    def load_saved_state(self,
                         args: argparse.Namespace,
                         unparsed_args: List[str],
                         metadata: Tuple[
                             Any,
                             str,
                             Any],
                         state: Tuple[NeuralPredictorState,
                                      Dict[str, Any]]) -> None:
        fpa_meta, q_name, q_meta = metadata
        fpa_state, q_state = state
        self._fpa = features_polyarg_predictor.FeaturesPolyargPredictor()
        self._fpa.load_saved_state(args, unparsed_args, fpa_meta,
                                   fpa_state)
        # The argument values here don't matter if we're not training,
        # so we set them to zero (which will hopefully break if we try
        # to train)
        if q_name == "features evaluator":
            self._estimator = features_q_estimator.FeaturesQEstimator(0, 0, 0)
        elif q_name == "polyarg evaluator":
            self._estimator = polyarg_q_estimator.PolyargQEstimator(0, 0, 0,
                                                                    self._fpa)
        else:
            assert False
        self._estimator.load_saved_state(args, unparsed_args, q_meta, q_state)

        self.unparsed_args = unparsed_args
        pass

    def getOptions(self) -> List[Tuple[str, str]]:
        return [("predictor", "refpa")]

    def predictKTactics(self, in_data: TacticContext, k: int) \
            -> List[Prediction]:
        assert self._fpa
        assert self._estimator
        # eprint(f"In goal: {in_data.goal}")
        inner_predictions = self._fpa.predictKTactics(in_data, 16)
        # q_choices = list(zip([p.certainty for p in inner_predictions],
        #                      inner_predictions))
        q_choices = list(zip(self._estimator(
            [(in_data, prediction.prediction, prediction.certainty)
             for prediction in inner_predictions]),
                        inner_predictions))
        # eprint("Scored predictions:")
        # for score, prediction in q_choices:
        #     eprint(f"{prediction.prediction}: {score}")
        # assert False
        ordered_actions = [Prediction(p[1].prediction, p[0]) for p in
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
    def share_memory(self) -> None:
        self._fpa.share_memory()
        self._estimator.share_memory()
    def to_device(self, device) -> None:
        self._fpa.to_device(device)
        self._estimator.to_device(device)
