#!/usr/bin/env python3

from typing import Dict, List, Union
from models.tactic_predictor import TacticPredictor

from models import encdecrnn_predictor
from models import try_common_predictor
from models import wordbagclass_predictor
from models import encclass_predictor
from models import dnnclass_predictor
from models import k_nearest_predictor

predictors = {
    'encdecrnn' : encdecrnn_predictor.EncDecRNNPredictor,
    'encclass' : encclass_predictor.EncClassPredictor,
    'dnnclass' : dnnclass_predictor.DNNClassPredictor,
    'trycommon' : try_common_predictor.TryCommonPredictor,
    'wordbagclass' : wordbagclass_predictor.WordBagClassifyPredictor,
    'k-nearest' : k_nearest_predictor.KNNPredictor,
}

def loadPredictor(options : Dict[str, Union[int, str]], predictor_type) -> TacticPredictor:
    return predictors[predictor_type](options)
