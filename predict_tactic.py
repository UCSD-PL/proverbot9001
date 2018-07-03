#!/usr/bin/env python3

from typing import Dict, List
import encdecrnn_predictor

predictors = {
    'encdecrnn' : encdecrnn_predictor.EncDecRNNPredictor
}

def loadPredictor(options : Dict[str, int], predictor_type="encdecrnn") -> TacticPredictor:
    return predictors[predictor_type](options)
