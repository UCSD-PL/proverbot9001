#!/usr/bin/env python3

from typing import Dict, List, Union
from models.tactic_predictor import TacticPredictor

from models import encdecrnn_predictor
from models import try_common_predictor

predictors = {
    'encdecrnn' : encdecrnn_predictor.EncDecRNNPredictor,
    'trycommon' : try_common_predictor.TryCommonPredictor,
}

def loadPredictor(options : Dict[str, Union[int, str]], predictor_type) -> TacticPredictor:
    return predictors[predictor_type](options)
