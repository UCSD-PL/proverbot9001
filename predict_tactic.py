#!/usr/bin/env python3

from typing import Dict, List, Union
import encdecrnn_predictor
from tactic_predictor import TacticPredictor

predictors = {
    'encdecrnn' : encdecrnn_predictor.EncDecRNNPredictor
}

def loadPredictor(options : Dict[str, Union[int, str]], predictor_type="encdecrnn") -> TacticPredictor:
    return predictors[predictor_type](options)
