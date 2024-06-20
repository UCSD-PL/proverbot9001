#!/usr/bin/env python3.7
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
from typing import Dict, List, Union, Callable, Optional
import functools
from models.tactic_predictor import TacticPredictor, TrainablePredictor
from models.components import DNNClassifierModel

import importlib

loadable_predictors = {
    'encdec' : ("encdecrnn_predictor", "EncDecRNNPredictor"),
    'encclass' : ("encclass_predictor", "EncClassPredictor"),
    'dnnclass' : ("dnnclass_predictor", "DNNClassPredictor"),
    'trycommon' : ("try_common_predictor", "TryCommonPredictor"),
    'wordbagclass' : ("wordbagclass_predictor", "WordBagClassifyPredictor"),
    'ngramclass' : ("ngramclass_predictor", "NGramClassifyPredictor"),
    'k-nearest' : ("k_nearest_predictor", "KNNPredictor"),
    'autoclass' : ("autoclass_predictor", "AutoClassPredictor"),
    'wordbagsvm' : ("wordbagsvm_classifier", "WordBagSVMClassifier"),
    'ngramsvm' : ("ngramsvm_classifier", "NGramSVMClassifier"),
    'pec' : ("pec_predictor", "PECPredictor"),
    'features' : ("features_predictor", "FeaturesPredictor"),
    'featuressvm' : ("featuressvm_predictor", "FeaturesSVMPredictor"),
    'encfeatures' : ("encfeatures_predictor", "EncFeaturesPredictor"),
    'apply' : ("apply_predictor", "ApplyPredictor"),
    'hypfeatures' : ("hypfeatures_predictor", "HypFeaturesPredictor"),
    'copyarg' : ("copyarg_predictor", "CopyArgPredictor"),
    'polyarg' : ("features_polyarg_predictor", "FeaturesPolyargPredictor"),
    'refpa': ("reinforced_features_polyarg", "ReinforcedFeaturesPolyargPredictor"),
}

static_predictors = {
    'apply_longest' : ("apply_baselines", "ApplyLongestPredictor"),
    'apply_similar' : ("apply_baselines", "ApplyStringSimilarPredictor"),
    'apply_similar2' : ("apply_baselines", "ApplyNormalizedSimilarPredictor"),
    'apply_wordsim' : ("apply_baselines", "ApplyWordSimlarPredictor"),
    'numeric_induction' : ("numeric_induction", "NumericInductionPredictor"),
}

trainable_modules = {
    "encdec" : ("encdecrnn_predictor", "main"),
    "encclass" : ("encclass_predictor", "main"),
    "dnnclass" : ("dnnclass_predictor", "main"),
    "trycommon" : ("try_common_predictor", "train"),
    "wordbagclass" : ("wordbagclass_predictor", "main"),
    "ngramclass" : ("ngramclass_predictor", "main"),
    "k-nearest" : ("k_nearest_predictor", "main"),
    "autoclass" : ("autoclass_predictor", "main"),
    "wordbagsvm" : ("wordbagsvm_classifier", "main"),
    "ngramsvm" : ("ngramsvm_classifier", "main"),
    "pec" : ("pec_predictor", "main"),
    "features" : ("features_predictor", "main"),
    "featuressvm" : ("featuressvm_predictor", "main"),
    "encfeatures" : ("encfeatures_predictor", "main"),
    "relevance" : ("apply_predictor", "train_relevance"),
    "hypstem" : ("hypstem_predictor", "main"),
    "hypfeatures" : ("hypfeatures_predictor", "main"),
    "copyarg" : ("copyarg_predictor", "main"),
    "polyarg" : ("features_polyarg_predictor", "main"),
}

def loadTrainablePredictor(predictor_type: str) -> Callable[[List[str]], None]:
    module_name, method_name = trainable_modules[predictor_type]
    method = vars(importlib.import_module("models." + module_name))[method_name]
    return method

def loadPredictorByName(predictor_type : str) -> TacticPredictor:
    module_name, class_name = static_predictors[predictor_type]
    predictor_class = vars(importlib.import_module("models." + module_name))[class_name]
    return predictor_class() # type: ignore

def loadPredictorByFile(filename : str, device: Optional[str] = None) -> TrainablePredictor:
    predictor_type, saved_state = torch.load(str(filename), map_location='cpu')
    module_name, class_name = loadable_predictors[predictor_type]
    predictor_class = vars(importlib.import_module("models." + module_name))[class_name]
    predictor = predictor_class(device)
    predictor.load_saved_state(*saved_state)
    return predictor
