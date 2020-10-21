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
from typing import Dict, List, Union, Callable
import functools
from models.tactic_predictor import TacticPredictor, TrainablePredictor
from models.components import DNNClassifierModel

from models import encdecrnn_predictor
from models import try_common_predictor
from models import wordbagclass_predictor
from models import ngramclass_predictor
from models import encclass_predictor
from models import dnnclass_predictor
from models import k_nearest_predictor
from models import autoclass_predictor
from models import wordbagsvm_classifier
from models import ngramsvm_classifier
from models import pec_predictor
from models import features_predictor
from models import encfeatures_predictor
from models import featuressvm_predictor
from models import apply_predictor
from models import apply_baselines
from models import hypstem_predictor
from models import hypfeatures_predictor
from models import copyarg_predictor
from models import numeric_induction
from models import features_polyarg_predictor
from models import reinforced_features_polyarg

loadable_predictors = {
    'encdec' : encdecrnn_predictor.EncDecRNNPredictor,
    'encclass' : encclass_predictor.EncClassPredictor,
    'dnnclass' : dnnclass_predictor.DNNClassPredictor,
    'trycommon' : try_common_predictor.TryCommonPredictor,
    'wordbagclass' : wordbagclass_predictor.WordBagClassifyPredictor,
    'ngramclass' : ngramclass_predictor.NGramClassifyPredictor,
    'k-nearest' : k_nearest_predictor.KNNPredictor,
    'autoclass' : autoclass_predictor.AutoClassPredictor,
    'wordbagsvm' : wordbagsvm_classifier.WordBagSVMClassifier,
    'ngramsvm' : ngramsvm_classifier.NGramSVMClassifier,
    'pec' : pec_predictor.PECPredictor,
    'features' : features_predictor.FeaturesPredictor,
    'featuressvm' : featuressvm_predictor.FeaturesSVMPredictor,
    'encfeatures' : encfeatures_predictor.EncFeaturesPredictor,
    'apply' : apply_predictor.ApplyPredictor,
    "hypstem" : functools.partial(hypstem_predictor.HypStemPredictor,
                                  DNNClassifierModel),
    "hypfeatures" : hypfeatures_predictor.HypFeaturesPredictor,
    "copyarg" : copyarg_predictor.CopyArgPredictor,
    "polyarg" : features_polyarg_predictor.FeaturesPolyargPredictor,
    "refpa": reinforced_features_polyarg.ReinforcedFeaturesPolyargPredictor,
}

static_predictors = {
    'apply_longest' : apply_baselines.ApplyLongestPredictor,
    'apply_similar' : apply_baselines.ApplyStringSimilarPredictor,
    'apply_similar2' : apply_baselines.ApplyNormalizedSimilarPredictor,
    'apply_wordsim' : apply_baselines.ApplyWordSimlarPredictor,
    'numeric_induction' : numeric_induction.NumericInductionPredictor,
}

trainable_modules : Dict[str, Callable[[List[str]], None]] = {
    "encdec" : encdecrnn_predictor.main,
    "encclass" : encclass_predictor.main,
    "dnnclass" : dnnclass_predictor.main,
    "trycommon" : try_common_predictor.train,
    "wordbagclass" : wordbagclass_predictor.main,
    "ngramclass" : ngramclass_predictor.main,
    "k-nearest" : k_nearest_predictor.main,
    "autoclass" : autoclass_predictor.main,
    "wordbagsvm" : wordbagsvm_classifier.main,
    "ngramsvm" : ngramsvm_classifier.main,
    "pec" : pec_predictor.main,
    "features" : features_predictor.main,
    "featuressvm" : featuressvm_predictor.main,
    "encfeatures" : encfeatures_predictor.main,
    "relevance" : apply_predictor.train_relevance,
    "hypstem" : hypstem_predictor.main,
    "hypfeatures" : hypfeatures_predictor.main,
    "copyarg" : copyarg_predictor.main,
    "polyarg" : features_polyarg_predictor.main,
}

def loadPredictorByName(predictor_type : str) -> TacticPredictor:
    # Silencing the type checker on this line because the "real" type
    # of the predictors dictionary is "string to classes constructors
    # that derive from TacticPredictor, but are not tactic
    # predictor". But I don't know how to specify that.
    return static_predictors[predictor_type]() # type: ignore

def loadPredictorByFile(filename : str) -> TrainablePredictor:
    predictor_type, saved_state = torch.load(filename, map_location='cpu')
    # Silencing the type checker on this line because the "real" type
    # of the predictors dictionary is "string to classes constructors
    # that derive from TacticPredictor, but are not tactic
    # predictor". But I don't know how to specify that.
    predictor = loadable_predictors[predictor_type]() # type: ignore
    predictor.load_saved_state(*saved_state)
    return predictor
