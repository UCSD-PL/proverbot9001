import numpy as np
import torch
import json


from data import (ListDataset, RawDataset,
                  EOS_token)
from util import (eprint, maybe_cuda, LongTensor, FloatTensor,
                  ByteTensor, print_time, unwrap)
import util
import math
from coq_serapy.contexts import TacticContext
from models.components import (WordFeaturesEncoder, Embedding,
                               DNNClassifier, EncoderDNN, EncoderRNN,
                               add_nn_args)
from models.tactic_predictor import (TacticPredictor, TrainablePredictor,
                                     NeuralPredictorState, Prediction,
                                     optimize_checkpoints, add_tokenizer_args)
import dataloader
from dataloader import (features_polyarg_tensors,
                        features_polyarg_tensors_with_meta,
                        sample_fpa,
                        sample_fpa_batch,
                        decode_fpa_result,
                        encode_fpa_stem,
                        encode_fpa_arg,
                        decode_fpa_stem,
                        # decode_fpa_arg,
                        # features_vocab_sizes,
                        get_num_tokens,
                        get_num_indices,
                        get_word_feature_vocab_sizes,
                        get_vec_features_size,
                        DataloaderArgs,
                        get_fpa_words)

import coq_serapy as serapi_instance

import argparse
import sys
from argparse import Namespace
from typing import (List, Tuple, NamedTuple, Optional, Sequence, Dict,
                    cast, Union, Set, Type, Any, Iterable)

from enum import Enum, auto


class PromptingPredictor(TacticPredictor):
    def __init__(self):
        self.training_args: Optional[argparse.Namespace] = None
        self.unparsed_args : List[str] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        '''
            todo: initialize model params
        '''

    def getOptions(self) -> List[Tuple[str, str]]:
        return [("predictor", "prompting")]
    def load_saved_state(self) -> None:
        '''
            todo: set model params, load model     
        '''
    def build_prompt(self, context: TacticContext) -> str:
        '''
            todo: build a prompt based on the context 
        '''
        lemma_statement = context.prev_tactics[0]
        goal = context.goal 
        prompt = ""
        lemmas = "\n".join(context.relevant_lemmas[-3:])
        hyps = "\n".join(context.hypotheses)
        prevs = "\n".join([l.strip() for l in context.prev_tactics])
        return prompt

    def predictKTactics(self, context: TacticContext, k: int) -> List[Prediction]:
        prompt_text = self.build_prompt(context)
        predictions = ["intros"]*k # dummy predictions, todo: fetch real predictions from OpenAI model
        
        return [Prediction(s, 1.0) for s in predictions]


    def getOptions(self) -> List[Tuple[str, str]]: pass

    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) -> \
        Tuple[List[Prediction], float]: pass
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, correct : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]: pass
