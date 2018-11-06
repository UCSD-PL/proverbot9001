#!/usr/bin/env python3

from typing import Dict, List, Union
from models.tactic_predictor import TacticPredictor

from models import encdecrnn_predictor
from models import try_common_predictor
from models import wordbagclass_predictor
from models import ngramclass_predictor
from models import encclass_predictor
from models import dnnclass_predictor
from models import k_nearest_predictor
from models import autoclass_predictor

predictors = {
    'encdecrnn' : encdecrnn_predictor.EncDecRNNPredictor,
    'encclass' : encclass_predictor.EncClassPredictor,
    'dnnclass' : dnnclass_predictor.DNNClassPredictor,
    'trycommon' : try_common_predictor.TryCommonPredictor,
    'wordbagclass' : wordbagclass_predictor.WordBagClassifyPredictor,
    'ngramclass' : ngramclass_predictor.NGramClassifyPredictor,
    'k-nearest' : k_nearest_predictor.KNNPredictor,
    'autoclass' : autoclass_predictor.AutoClassPredictor,
}

def loadPredictor(options : Dict[str, Union[int, str]], predictor_type) -> TacticPredictor:
    # Silencing the type checker on this line because the "real" type
    # of the predictors dictionary is "string to classes constructors
    # that derive from TacticPredictor, but are not tactic
    # predictor". But I don't know how to specify that.
    return predictors[predictor_type](options) # type: ignore
