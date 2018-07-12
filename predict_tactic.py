#!/usr/bin/env python3

from typing import Dict, List, Union
from models.tactic_predictor import TacticPredictor

from models import encdecrnn_predictor
from models import try_common_predictor
from models import wordbagclass_predictor
from models import encclass_predictor


predictors = {
    'encdecrnn' : encdecrnn_predictor.EncDecRNNPredictor,
    'encclass' : encclass_predictor.EncClassPredictor
    'trycommon' : try_common_predictor.TryCommonPredictor,
    'wordbagclass' : wordbagclass_predictor.WordBagClassifyPredictor,
}

def loadPredictor(options : Dict[str, Union[int, str]], predictor_type) -> TacticPredictor:
    return predictors[predictor_type](options)
