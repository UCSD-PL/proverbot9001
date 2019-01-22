import argparse
import time
import math
import pickle
import sys
import threading

from typing import Dict, Any, List, Tuple, Iterable, cast, Union
from argparse import Namespace

# Using sklearn for the actual learning
from sklearn import svm

# Using Torch to get nllloss
import torch
from torch import nn
from torch.autograd import Variable

from models.tactic_predictor import TokenizingPredictor, Prediction, TacticContext, TokenizerEmbeddingState
from tokenizer import tokenizers
from data import get_text_data, getNGramTokenbagVector, encode_ngram_classify_data, \
    encode_ngram_classify_input, TokenizedDataset, Dataset, NGram, NGramSample, \
    NGramDataset
from util import *
from serapi_instance import get_stem

class NGramSVMClassifier(TokenizingPredictor[NGramDataset, svm.SVC]):
    def load_saved_state(self,
                         args : Namespace,
                         metadata : TokenizerEmbeddingState,
                         state : svm.SVC) -> None:
        self._tokenizer = metadata.tokenizer
        self._embedding = metadata.embedding
        self.training_args = args
        self.context_filter = args.context_filter
        self._model = state
        pass
    def getOptions(self) -> List[Tuple[str, str]]:
        return list(vars(self.training_args).items())

    def __init__(self) -> None:
        self._criterion = nn.NLLLoss()
        self._lock = threading.Lock()

    def _encode_tokenized_data(self, data : TokenizedDataset, arg_values : Namespace,
                               term_vocab_size : int, tactic_vocab_size : int) \
        -> NGramDataset:
        return NGramDataset([NGramSample(getNGramTokenbagVector(arg_values.num_grams,
                                                                term_vocab_size,
                                                                goal),
                                         tactic) for prev_tactic, goal, tactic in data])

    def predictDistribution(self, in_data : TacticContext) \
        -> torch.FloatTensor:
        feature_vector = \
            encode_ngram_classify_input(in_data.goal,
                                        self.training_args.num_grams,
                                        self._tokenizer)
        distribution = self._model.predict_log_proba([feature_vector])[0]
        return distribution

    def predictKTactics(self, in_data : TacticContext, k : int) \
        -> List[Prediction]:
        with self._lock:
            distribution = self.predictDistribution(in_data)
            indices, probabilities = list_topk(list(distribution), k)
        return [Prediction(self._embedding.decode_token(idx) + ".",
                           math.exp(certainty))
                for certainty, idx in zip(probabilities, indices)]

    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int,
                                correct : str) -> Tuple[List[Prediction], float]:
        with self._lock:
            distribution = self.predictDistribution(in_data)
            correct_stem = get_stem(correct)
            if self._embedding.has_token(correct_stem):
                loss = self._criterion(torch.FloatTensor(distribution).view(1, -1), Variable(torch.LongTensor([self._embedding.encode_token(correct_stem)]))).item()
            else:
                loss = float("+inf")
            indices, probabilities = list_topk(list(distribution), k)
            predictions = [Prediction(self._embedding.decode_token(idx) + ".",
                                      math.exp(certainty))
                           for certainty, idx in zip(probabilities, indices)]
        return predictions, loss
    def predictKTacticsWithLoss_batch(self,
                                      in_datas : List[TacticContext],
                                      k : int, corrects : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:

        results = [self.predictKTacticsWithLoss(in_data, k, correct)
                   for in_data, correct in zip(in_datas, corrects)]

        results2 : Tuple[List[List[Prediction]], List[float]] = tuple(zip(*results)) # type: ignore
        prediction_lists, losses = results2
        return prediction_lists, sum(losses)/len(losses)
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) \
                           -> None:
        super().add_args_to_parser(parser, default_values)
        parser.add_argument("--num-grams", "-n", dest="num_grams", type=int, default=1)
        parser.add_argument("--kernel", choices=svm_kernels, type=str,
                            default=svm_kernels[0])
    def _optimize_model_to_disc(self,
                                encoded_data : NGramDataset,
                                encdec_state : TokenizerEmbeddingState,
                                arg_values : Namespace) \
        -> None:
        curtime = time.time()
        print("Training SVM...", end="")
        sys.stdout.flush()
        model = svm.SVC(gamma='scale', kernel=arg_values.kernel, probability=True)
        inputs, outputs = zip(*encoded_data.data)
        model.fit(inputs, outputs)
        print(" {:.2f}s".format(time.time() - curtime))
        loss = model.score(inputs, outputs)
        print("Training loss: {}".format(loss))
        with open(arg_values.save_file, 'wb') as f:
            torch.save((arg_values, encdec_state, model), f)

    def _description(self) -> str:
        return "A simple predictor which tries the k most common tactic stems."

svm_kernels = [
    "rbf",
    "linear",
]

def main(args_list : List[str]) -> None:
    predictor = NGramSVMClassifier()
    predictor.train(args_list)
