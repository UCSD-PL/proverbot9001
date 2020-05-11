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

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.components import (EncoderRNN, WordFeaturesEncoder, Embedding, add_nn_args)
from features import (WordFeature, VecFeature,
                      word_feature_constructors, vec_feature_constructors)
from models.tactic_predictor import (TrainablePredictor,
                                     NeuralPredictorState, Prediction,
                                     optimize_checkpoints,
                                     save_checkpoints, tokenize_goals,
                                     embed_data, add_tokenizer_args)
from tokenizer import Tokenizer, limitNumTokens, get_symbols
from data import (ListDataset, normalizeSentenceLength, RawDataset,
                  EmbeddedSample)
from util import *
from format import ScrapedTactic, TacticContext, strip_scraped_output
import serapi_instance

import threading
import multiprocessing
import functools
import sys
from typing import List, NamedTuple, cast, Sequence, Dict
import argparse
from argparse import Namespace
from difflib import SequenceMatcher

class HypFeaturesSample(NamedTuple):
    word_features : List[int]
    vec_features : List[float]
    goal : List[int]
    best_hypothesis : List[int]
    next_tactic : int
class HypFeaturesDataset(ListDataset[HypFeaturesSample]):
    pass

class HypFeaturesStemClassifier(nn.Module):
    def __init__(self, vec_features_size : int,
                 word_feature_vocab_sizes : List[int],
                 term_token_vocab_size : int,
                 hidden_size : int, num_layers : int,
                 tactic_vocab_size : int) -> None:
        super().__init__()
        self._goal_encoder = EncoderRNN(term_token_vocab_size, hidden_size, hidden_size)
        self._hyp_encoder = EncoderRNN(term_token_vocab_size, hidden_size, hidden_size)
        self._word_features_encoder = WordFeaturesEncoder(word_feature_vocab_sizes,
                                                          hidden_size, num_layers-1,
                                                          hidden_size)
        self._layer = nn.Linear(hidden_size * 3 + vec_features_size, hidden_size)
        self._out_layer = nn.Linear(hidden_size, tactic_vocab_size)
        self._softmax = maybe_cuda(nn.LogSoftmax(dim=1))
        pass

    def forward(self,
                goal_batch : torch.LongTensor,
                hyp_batch : torch.LongTensor,
                vec_features_batch : torch.FloatTensor,
                word_features_batch : torch.LongTensor) -> torch.FloatTensor:
        goal_data = self._goal_encoder(goal_batch)
        hyp_data = self._hyp_encoder(hyp_batch)
        word_features_data = self._word_features_encoder(word_features_batch)
        catted_data = torch.cat((goal_data, hyp_data, word_features_data,
                                 maybe_cuda(vec_features_batch)),
                                dim=1)
        full_data = self._layer(F.relu(catted_data))
        full_data = self._out_layer(F.relu(full_data))
        result = self._softmax(full_data)
        return result

class HypFeaturesPredictor(TrainablePredictor[HypFeaturesDataset,
                                              Tuple[Tokenizer, Embedding,
                                                    List[WordFeature], List[VecFeature]],
                                              NeuralPredictorState]):
    def __init__(self) -> None:
        self._word_feature_functions: Optional[List[WordFeature]] = None
        self._vec_feature_functions: Optional[List[VecFeature]] = None
        self._criterion = maybe_cuda(nn.NLLLoss())
        self._lock = threading.Lock()
        self._tokenizer : Optional[Tokenizer] = None
        self._embedding : Optional[Embedding] = None
    def _get_word_features(self, context : TacticContext) -> List[int]:
        assert self._word_feature_functions
        return [feature(context) for feature in self._word_feature_functions]
    def _get_vec_features(self, context : TacticContext) -> List[float]:
        assert self._vec_feature_functions
        return [feature_val for feature in self._vec_feature_functions
                for feature_val in feature(context)]
    def _predictDistributions(self, in_datas : List[TacticContext]) -> torch.FloatTensor:
        assert self._tokenizer
        assert self._embedding
        assert self.training_args
        goals_batch = [normalizeSentenceLength(self._tokenizer.toTokenList(goal),
                                               self.training_args.max_length)
                       for _, _, _, goal in in_datas]
        hyps = [get_closest_hyp(hyps, goal, self.training_args.max_length)
                for _, _, hyps, goal in in_datas]
        hyp_types = [serapi_instance.get_hyp_type(hyp) for hyp in hyps]
        hyps_batch = [normalizeSentenceLength(
            self._tokenizer.toTokenList(hyp_type),
                      self.training_args.max_length)
                      for hyp_type in hyp_types]
        word_features_batch = [self._get_word_features(in_data) for in_data in in_datas]
        vec_features_batch = [self._get_vec_features(in_data) for in_data in in_datas]
        stem_distribution = self._model(LongTensor(goals_batch),
                                        LongTensor(hyps_batch),
                                        FloatTensor(vec_features_batch),
                                        LongTensor(word_features_batch))
        return stem_distribution
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) -> None:
        super().add_args_to_parser(parser, {"learning-rate": 0.4,
                                            **default_values})
        add_nn_args(parser, default_values)
        add_tokenizer_args(parser, default_values)
        parser.add_argument("--max-length", dest="max_length", type=int,
                            default=default_values.get("max-length", 10))
        parser.add_argument("--num-head-keywords", dest="num_head_keywords", type=int,
                            default=default_values.get("num-head-keywords", 100))
        parser.add_argument("--num-tactic-keywords", dest="num_tactic_keywords", type=int,
                            default=default_values.get("num-tactic-keywords", 50))

    def _encode_data(self, data : RawDataset, arg_values : Namespace) \
        -> Tuple[HypFeaturesDataset, Tuple[Tokenizer, Embedding,
                                           List[WordFeature], List[VecFeature]]]:
        start = time.time()
        print("Stripping...", end="")
        sys.stdout.flush()
        stripped_data = [strip_scraped_output(dat) for dat in data]
        print("{:.2f}s".format(time.time() - start))
        self._word_feature_functions = [feature_constructor(stripped_data, arg_values) for # type: ignore
                                        feature_constructor in
                                        word_feature_constructors]
        self._vec_feature_functions = [feature_constructor(stripped_data, arg_values) for # type: ignore
                                       feature_constructor in vec_feature_constructors]
        embedding, embedded_data = embed_data(data)
        tokenizer, tokenized_goals = tokenize_goals(embedded_data, arg_values)
        with multiprocessing.Pool(arg_values.num_threads) as pool:
            start = time.time()
            print("Getting closest hyps...", end="")
            sys.stdout.flush()
            tokenized_hyps = list(pool.imap(functools.partial(get_closest_hyp_type,
                                                              tokenizer,
                                                              arg_values.max_length),
                                            data))
            print("{:.2f}s".format(time.time() - start))
            start = time.time()
            print("Creating dataset...", end="")
            sys.stdout.flush()
            result_data = HypFeaturesDataset(list(pool.imap(
                functools.partial(mkHFSample,
                                  arg_values.max_length,
                                  self._word_feature_functions,
                                  self._vec_feature_functions),
                zip(embedded_data, tokenized_goals,
                    tokenized_hyps))))
            print("{:.2f}s".format(time.time() - start))
        return result_data, (tokenizer, embedding,
                             self._word_feature_functions,
                             self._vec_feature_functions)

    def _optimize_model_to_disc(self,
                                encoded_data : HypFeaturesDataset,
                                metadata : Tuple[Tokenizer, Embedding,
                                                 List[WordFeature], List[VecFeature]],
                                arg_values : Namespace) \
        -> None:
        tokenizer, embedding, word_features, vec_features = metadata
        save_checkpoints("hypfeatures",
                         metadata, arg_values,
                         self._optimize_checkpoints(encoded_data, arg_values,
                                                    tokenizer, embedding))
    def _optimize_checkpoints(self, encoded_data : HypFeaturesDataset,
                              arg_values : Namespace,
                              tokenizer : Tokenizer,
                              embedding : Embedding) \
        -> Iterable[NeuralPredictorState]:
        return optimize_checkpoints(self._data_tensors(encoded_data, arg_values),
                                    arg_values,
                                    self._get_model(arg_values, embedding.num_tokens(),
                                                    tokenizer.numTokens()),
                                    lambda batch_tensors, model:
                                    self._getBatchPredictionLoss(batch_tensors, model))

    def load_saved_state(self,
                         args : Namespace,
                         unparsed_args : List[str],
                         metadata : Tuple[Tokenizer, Embedding,
                                          List[WordFeature], List[VecFeature]],
                         state : NeuralPredictorState) -> None:
        self._tokenizer, self._embedding, \
            self._word_feature_functions, self._vec_feature_functions= \
                metadata
        self._model = maybe_cuda(self._get_model(args,
                                                 self._embedding.num_tokens(),
                                                 self._tokenizer.numTokens()))
        self._model.load_state_dict(state.weights)
        self.training_loss = state.loss
        self.num_epochs = state.epoch
        self.training_args = args
        self.unparsed_args = unparsed_args
    def _data_tensors(self, encoded_data : HypFeaturesDataset,
                      arg_values : Namespace) \
        -> List[torch.Tensor]:
        word_features, vec_features, goals, hyps, tactics = zip(*encoded_data)
        return [torch.LongTensor(word_features),
                torch.FloatTensor(vec_features),
                torch.LongTensor(goals),
                torch.LongTensor(hyps),
                torch.LongTensor(tactics)]

    def _get_model(self, arg_values : Namespace,
                   tactic_vocab_size : int,
                   goal_vocab_size : int) \
        -> HypFeaturesStemClassifier:
        assert self._word_feature_functions
        assert self._vec_feature_functions
        feature_vec_size = sum([feature.feature_size()
                                for feature in self._vec_feature_functions])
        word_feature_vocab_sizes = [feature.vocab_size()
                                    for feature in self._word_feature_functions]
        return HypFeaturesStemClassifier(feature_vec_size, word_feature_vocab_sizes,
                                         goal_vocab_size,
                                         arg_values.hidden_size,
                                         arg_values.num_layers,
                                         tactic_vocab_size)

    def _getBatchPredictionLoss(self, data_batch : Sequence[torch.Tensor],
                                model : HypFeaturesStemClassifier) \
        -> torch.FloatTensor:
        word_features_batch, vec_features_batch, goals_batch, hyps_batch, outputs_batch = \
            cast(Tuple[torch.LongTensor, torch.FloatTensor,
                       torch.LongTensor, torch.LongTensor,
                       torch.LongTensor],
                 data_batch)
        predictionDistribution = model(goals_batch, hyps_batch,
                                       vec_features_batch, word_features_batch)
        output_var = maybe_cuda(Variable(outputs_batch))
        return self._criterion(predictionDistribution, output_var)
    def add_arg(self, tactic_stem : str, goal : str, hyps : List[str], max_length : int):
        if serapi_instance.tacticTakesHypArgs(tactic_stem):
            return tactic_stem + " " + \
                serapi_instance.get_first_var_in_hyp(get_closest_hyp(hyps, goal,
                                                                     max_length)) \
                + "."
        else:
            return tactic_stem + "."
    def add_args(self, stem_predictions : List[Prediction],
                 goal : str, hyps : List[str], max_length : int):
        possibilities : List[Prediction] = []
        for stem, stem_score in stem_predictions:
            if serapi_instance.tacticTakesHypArgs(stem):
                for hyp, hyp_score in get_closest_hyps(hyps, goal,
                                                       len(stem_predictions),
                                                       max_length):
                    possibilities.append(
                        Prediction(stem + " " +
                                   serapi_instance.get_first_var_in_hyp(hyp) + ".",
                                   hyp_score * stem_score))
            else:
                possibilities.append(Prediction(stem + ".", stem_score * 0.5))
        return list(sorted(possibilities, key=lambda pred: pred.certainty,
                           reverse=True)[:len(stem_predictions)])

    def predictKTactics(self, in_data : TacticContext, k : int) \
        -> List[Prediction]:
        assert self.training_args
        assert self._embedding
        with self._lock:
            prediction_distribution = self._predictDistributions([in_data])[0]
        if k > self._embedding.num_tokens():
            k = self._embedding.num_tokens()
        if len(in_data.hypotheses) == 0:
            certainties, idxs = topk_with_filter(
                prediction_distribution.view(-1), k,
                lambda certainty, idx:
                not serapi_instance.tacticTakesHypArgs(
                    cast(Embedding, self._embedding).decode_token(idx)))
        else:
            certainties, idxs = prediction_distribution.view(-1).topk(k)
        results = self.add_args([Prediction(self._embedding.decode_token(stem_idx.item()),
                                            math.exp(certainty.item()))
                                 for certainty, stem_idx in zip(certainties, idxs)],
                                in_data.goal, in_data.hypotheses,
                                self.training_args.max_length)
        return results

    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) -> \
        Tuple[List[Prediction], float]:
        assert self.training_args
        assert self._embedding
        with self._lock:
            prediction_distribution = self._predictDistributions([in_data])[0]
        if k > self._embedding.num_tokens():
            k = self._embedding.num_tokens()
        correct_stem = serapi_instance.get_stem(correct)
        if self._embedding.has_token(correct_stem):
            output_var = maybe_cuda(Variable(
                LongTensor([self._embedding.encode_token(correct_stem)])))
            loss = self._criterion(prediction_distribution.view(1, -1), output_var).item()
        else:
            loss = 0
        if len(in_data.hypotheses) == 0:
            certainties, idxs = topk_with_filter(
                prediction_distribution.view(-1), k,
                lambda certainty, idx:
                not serapi_instance.tacticTakesHypArgs(
                    cast(Embedding, self._embedding).decode_token(idx)))
        else:
            certainties, idxs = prediction_distribution.view(-1).topk(k)
        results = [Prediction(self.add_arg(self._embedding.decode_token(stem_idx.item()),
                                           in_data.goal, in_data.hypotheses,
                                           self.training_args.max_length),
                              math.exp(certainty.item()))
                   for certainty, stem_idx in zip(certainties, idxs)]
        return results, loss
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, corrects : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        assert self._embedding
        assert self.training_args
        with self._lock:
            prediction_distributions = self._predictDistributions(in_data)
        correct_stems = [serapi_instance.get_stem(correct) for correct in corrects]
        output_var = maybe_cuda(Variable(
            LongTensor([self._embedding.encode_token(correct_stem)
                        if self._embedding.has_token(correct_stem)
                        else 0
                        for correct_stem in correct_stems])))
        loss = self._criterion(prediction_distributions, output_var).item()
        if k > self._embedding.num_tokens():
            k = self._embedding.num_tokens()
        certainties_and_idxs_list = \
            [single_distribution.view(-1).topk(k) if len(context.hypotheses) > 0 else
             topk_with_filter(single_distribution.view(-1), k,
                              lambda certainty, idx:
                              not serapi_instance.tacticTakesHypArgs(
                                  cast(Embedding, self._embedding).decode_token(idx)))
             for single_distribution, context in
             zip(prediction_distributions, in_data)]
        results = [[Prediction(self.add_arg(self._embedding.decode_token(stem_idx.item()),
                                            in_datum.goal, in_datum.hypotheses,
                                            self.training_args.max_length),
                               math.exp(certainty.item()))
                    for certainty, stem_idx in zip(*certainties_and_idxs)]
                   for certainties_and_idxs, in_datum in
                   zip(certainties_and_idxs_list, in_data)]
        return results, loss
    def getOptions(self) -> List[Tuple[str, str]]:
        assert self.training_args
        return list(vars(self.training_args).items()) + \
            [("training loss", self.training_loss),
             ("# epochs", self.num_epochs),
             ("predictor", "hypfeatures")]

    def _description(self) -> str:
        return "A predictor using an RNN on the tokenized goal and "\
            "hand-engineered features."

def score_hyp_type(goal : str, hyp_type : str, max_length : int):
    ratio = SequenceMatcher(None, goal, hyp_type).ratio()
    score = ratio * (len(get_symbols(hyp_type)) / max_length)
    return score
def get_closest_hyp(hyps : List[str], goal : str, max_length : int):
    if len(hyps) == 0:
        return ":"

    result = max(hyps, key=lambda hyp:
                 score_hyp_type(limitNumTokens(goal, max_length),
                                limitNumTokens(serapi_instance.get_hyp_type(hyp), max_length),
                                max_length))
    return result
def get_closest_hyp_type(tokenizer : Tokenizer, max_length : int, context : TacticContext):
    return tokenizer.toTokenList(serapi_instance.get_hyp_type(
        get_closest_hyp(context.hypotheses, context.goal, max_length)))
def get_closest_hyps(hyps : List[str], goal : str, num_hyps : int, max_length : int)\
                        -> List[Tuple[str, float]]:
    if len(hyps) == 0:
        return [Prediction(":", 0)] * num_hyps
    else:
        return list(sorted([(hyp, score_hyp_type(limitNumTokens(goal, max_length),
                                                 limitNumTokens(serapi_instance.get_hyp_type(hyp), max_length),
                                                 max_length))
                            for hyp in hyps],
                           reverse=True,
                           key=lambda hyp_and_score: hyp_and_score[0]))
def mkHFSample(max_length : int,
               word_feature_functions : List[WordFeature],
               vec_feature_functions : List[VecFeature],
               zipped : Tuple[EmbeddedSample, List[int], List[int]]) \
    -> HypFeaturesSample:
    context, goal, best_hyp = zipped
    (relevant_lemmas, prev_tactic_list, hypotheses, goal_str, tactic) = context
    tac_context = TacticContext(relevant_lemmas, prev_tactic_list, hypotheses, goal_str)
    return HypFeaturesSample([feature(tac_context)
                              for feature in word_feature_functions],
                             [feature_val for feature in vec_feature_functions
                              for feature_val in feature(tac_context)],
                             normalizeSentenceLength(goal, max_length),
                             normalizeSentenceLength(best_hyp, max_length),
                             tactic)
def main(arg_list : List[str]) -> None:
    predictor = HypFeaturesPredictor()
    predictor.train(arg_list)
