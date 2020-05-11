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

import torch
from models import id_evaluator
from models import features_dnn_evaluator
from models import goal_enc_evaluator
from models.state_evaluator import StateEvaluator, TrainableEvaluator
from pathlib_revised import Path2

from typing import Dict, Type

loadable_evaluators : Dict[str, Type[TrainableEvaluator]] = {
    'features-dnn' : features_dnn_evaluator.FeaturesDNNEvaluator,
    'eval-goal-enc' : goal_enc_evaluator.GoalEncEvaluator,
}

static_evaluators = {
    'id' : id_evaluator.IdEvaluator
}

trainable_modules = {
    'eval-features-dnn' : features_dnn_evaluator.main,
    'eval-goal-enc' : goal_enc_evaluator.main
}


def loadEvaluatorByName(evaluator_type : str) -> StateEvaluator:
    # Silencing the type checker on this line because the "real" type
    # of the evaluators dictionary is "string to classes constructors
    # that derive from EvaluatorPredictor, but are not state
    # evaluator". But I don't know how to specify that.
    return static_evaluators[evaluator_type]() # type: ignore

def loadEvaluatorByFile(filename : Path2) -> StateEvaluator:
    evaluator_type, saved_state = torch.load(str(filename), map_location='cpu')
    # Silencing the type checker on this line because the "real" type
    # of the predictors dictionary is "string to classes constructors
    # that derive from TacticPredictor, but are not tactic
    # predictor". But I don't know how to specify that.
    evaluator = loadable_evaluators[evaluator_type]() # type: ignore
    evaluator.load_saved_state(*saved_state)
    return evaluator
