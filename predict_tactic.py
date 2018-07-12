#!/usr/bin/env python3

from typing import Dict, List, Union
from tactic_predictor import TacticPredictor

import encdecrnn_predictor

predictors = {
    'encdecrnn' : encdecrnn_predictor.EncDecRNNPredictor,
}

def loadPredictor(options : Dict[str, Union[int, str]], predictor_type) -> TacticPredictor:
    return predictors[predictor_type](options)
