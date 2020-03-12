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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from features import (WordFeature, VecFeature, Feature,
                      word_feature_constructors, vec_feature_constructors)
import tokenizer
from tokenizer import Tokenizer
from data import (ListDataset, normalizeSentenceLength, RawDataset,
                  EOS_token)
from util import *
import time
import random
from format import ScrapedTactic, TacticContext, strip_scraped_output
import serapi_instance
from models.components import (WordFeaturesEncoder, Embedding, SimpleEmbedding,
                               DNNClassifier, EncoderDNN, EncoderRNN,
                               add_nn_args)
from models.tactic_predictor import (TrainablePredictor,
                                     NeuralPredictorState, Prediction,
                                     optimize_checkpoints,
                                     save_checkpoints, tokenize_goals,
                                     embed_data, add_tokenizer_args)

import threading
import multiprocessing
import argparse
import sys
import functools
from itertools import islice
from argparse import Namespace
from typing import (List, Tuple, NamedTuple, Optional, Sequence, Dict,
                    cast, Union, Set, Type)

from enum import Enum, auto
class ArgType(Enum):
    HYP_ID = auto()
    GOAL_TOKEN = auto()
    NO_ARG = auto()

class HypIdArg(NamedTuple):
    hyp_idx : int
class GoalTokenArg(NamedTuple):
    token_idx : int

TacticArg = Optional[Union[HypIdArg, GoalTokenArg]]

class FeaturesPolyArgSample(NamedTuple):
    tokenized_hyp_types : List[List[int]]
    hyp_features : List[List[float]]
    tokenized_goal : List[int]
    word_features : List[int]
    vec_features : List[float]
    tactic_stem : int
    arg_type : ArgType
    arg : TacticArg

class FeaturesPolyArgDataset(ListDataset[FeaturesPolyArgSample]):
    pass

class GoalTokenArgModel(nn.Module):
    def __init__(self, stem_vocab_size : int,
                 input_vocab_size : int, input_length : int,
                 hidden_size : int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self._stem_embedding = maybe_cuda(nn.Embedding(stem_vocab_size, hidden_size))
        self._token_embedding = maybe_cuda(nn.Embedding(input_vocab_size, hidden_size))
        self._gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self._likelyhood_layer = maybe_cuda(EncoderDNN(hidden_size, hidden_size, 1, 2))
        self._softmax = maybe_cuda(nn.LogSoftmax(dim=1))
    def forward(self, stem_batch : torch.LongTensor, goal_batch : torch.LongTensor) \
        -> torch.FloatTensor:
        goal_var = maybe_cuda(Variable(goal_batch))
        stem_var = maybe_cuda(Variable(stem_batch))
        batch_size = goal_batch.size()[0]
        assert  stem_batch.size()[0] == batch_size
        initial_hidden = self._stem_embedding(stem_var)\
                             .view(1, batch_size, self.hidden_size)
        hidden = initial_hidden
        copy_likelyhoods : List[torch.FloatTensor] = []
        for i in range(goal_batch.size()[1]):
            token_batch = self._token_embedding(goal_var[:,i])\
                .view(1, batch_size, self.hidden_size)
            token_batch = F.relu(token_batch)
            token_out, hidden = self._gru(token_batch, hidden)
            copy_likelyhood = self._likelyhood_layer(F.relu(token_out))
            copy_likelyhoods.append(copy_likelyhood[0])
        end_token_embedded = self._token_embedding(LongTensor([EOS_token])
                                                   .expand(batch_size))\
                                                   .view(1, batch_size, self.hidden_size)
        final_out, final_hidden = self._gru(F.relu(end_token_embedded), hidden)
        final_likelyhood = self._likelyhood_layer(F.relu(final_out))
        copy_likelyhoods.insert(0, final_likelyhood[0])
        catted = torch.cat(copy_likelyhoods, dim=1)
        return catted

class HypArgModel(nn.Module):
    def __init__(self, goal_data_size : int,
                 stem_vocab_size : int,
                 token_vocab_size : int,
                 hyp_features_size : int,
                 hidden_size : int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self._stem_embedding = maybe_cuda(nn.Embedding(stem_vocab_size, hidden_size))
        self._token_embedding = maybe_cuda(nn.Embedding(token_vocab_size, hidden_size))
        self._in_hidden = maybe_cuda(EncoderDNN(hidden_size + goal_data_size, hidden_size, hidden_size, 1))
        self._hyp_gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self._likelyhood_decoder = maybe_cuda(EncoderDNN(hidden_size + hyp_features_size, hidden_size, 1, 2))
    def forward(self, stems_batch : torch.LongTensor,
                goals_encoded_batch : torch.FloatTensor, hyps_batch : torch.LongTensor,
                hypfeatures_batch : torch.FloatTensor):
        stems_var = maybe_cuda(Variable(stems_batch))
        hyps_var = maybe_cuda(Variable(hyps_batch))
        hypfeatures_var = maybe_cuda(Variable(hypfeatures_batch))
        batch_size = stems_batch.size()[0]
        assert goals_encoded_batch.size()[0] == batch_size
        assert hyps_batch.size()[0] == batch_size, \
            "batch_size: {}; hyps_batch_size()[0]: {}"\
            .format(batch_size, hyps_batch.size()[0])
        assert hypfeatures_batch.size()[0] == batch_size
        stem_encoded = self._stem_embedding(stems_var)\
                           .view(batch_size, self.hidden_size)
        initial_hidden = self._in_hidden(torch.cat(
            (stem_encoded, goals_encoded_batch), dim=1))\
                             .view(1, batch_size, self.hidden_size)
        hidden = initial_hidden
        for i in range(hyps_batch.size()[1]):
            token_batch = self._token_embedding(hyps_var[:,i])\
                .view(1, batch_size, self.hidden_size)
            token_batch = F.relu(token_batch)
            token_out, hidden = self._hyp_gru(token_batch, hidden)
        hyp_likelyhoods = self._likelyhood_decoder(
            torch.cat((token_out.view(batch_size, self.hidden_size), hypfeatures_var),
                      dim=1))
        return hyp_likelyhoods

class FeaturesClassifier(nn.Module):
    def __init__(self,
                 word_features : List[WordFeature],
                 vec_features : List[VecFeature],
                 hidden_size : int,
                 num_layers : int,
                 stem_vocab_size : int)\
        -> None:
        super().__init__()
        feature_vec_size = sum([feature.feature_size()
                                for feature in vec_features])
        word_features_vocab_sizes = [features.vocab_size()
                                     for features in word_features]
        self._word_features_encoder = maybe_cuda(
            WordFeaturesEncoder(word_features_vocab_sizes,
                                hidden_size, 1, hidden_size))
        self._features_classifier = maybe_cuda(
            DNNClassifier(hidden_size + feature_vec_size,
                          hidden_size, stem_vocab_size, num_layers))
        self._softmax = maybe_cuda(nn.LogSoftmax(dim=1))
        pass
    def forward(self,
                word_features_batch : torch.LongTensor,
                vec_features_batch : torch.FloatTensor) -> torch.FloatTensor:
        encoded_word_features = self._word_features_encoder(
            maybe_cuda(word_features_batch))
        stem_distribution = self._softmax(self._features_classifier(
            torch.cat((encoded_word_features, maybe_cuda(vec_features_batch)), dim=1)))
        return stem_distribution

class FeaturesPolyArgModel(nn.Module):
    def __init__(self,
                 stem_classifier : FeaturesClassifier,
                 goal_args_model : GoalTokenArgModel,
                 goal_encoder : EncoderRNN,
                 hyp_model : HypArgModel) -> None:
        super().__init__()
        self.stem_classifier = maybe_cuda(stem_classifier)
        self.goal_args_model = maybe_cuda(goal_args_model)
        self.goal_encoder = maybe_cuda(goal_encoder)
        self.hyp_model = maybe_cuda(hyp_model)

from difflib import SequenceMatcher
class FeaturesPolyargPredictor(
        TrainablePredictor[FeaturesPolyArgDataset,
                           Tuple[Tokenizer, Embedding,
                                 List[WordFeature], List[VecFeature]],
                           NeuralPredictorState]):
    def __init__(self) -> None:
        self._criterion = maybe_cuda(nn.NLLLoss())
        self._lock = threading.Lock()
        self.training_args : Optional[argparse.Namespace] = None
        self.training_loss : Optional[float] = None
        self.num_epochs : Optional[int] = None
        self._word_feature_functions: Optional[List[WordFeature]] = None
        self._vec_feature_functions: Optional[List[VecFeature]] = None
        self._softmax = maybe_cuda(nn.LogSoftmax(dim=1))
        self._softmax2 = maybe_cuda(nn.LogSoftmax(dim=2))
        self._tokenizer : Optional[Tokenizer] = None
        self._embedding : Optional[Embedding] = None
        self._model : Optional[FeaturesPolyArgModel] = None
    def predictKTactics(self, context : TacticContext, k : int) -> List[Prediction]:
        assert self._tokenizer
        assert self._embedding
        assert self.training_args
        assert self._model
        beam_width=min(self.training_args.max_beam_width, k ** 2)
        if self.training_args.lemma_args:
            all_hyps = context.hypotheses + context.relevant_lemmas
        else:
            all_hyps = context.hypotheses
        num_hyps = len(all_hyps)

        num_stem_poss = self._embedding.num_tokens()
        stem_width = min(self.training_args.max_beam_width, num_stem_poss, k ** 2)

        with self._lock:

            word_features, vec_features = self.encodeFeatureVecs([context])
            stem_distribution = self._model.stem_classifier(word_features, vec_features)
            stem_certainties, stem_idxs = stem_distribution.topk(stem_width)

            goals_batch = LongTensor([self.encodeStrTerm(context.goal)])
            goal_arg_values = self._model.goal_args_model(
                stem_idxs.view(1 * stem_width),
                goals_batch.view(1, 1, self.training_args.max_length)\
                .expand(-1, stem_width, -1).contiguous()\
                .view(1 * stem_width,
                      self.training_args.max_length))\
                      .view(1, stem_width, self.training_args.max_length + 1)
            goal_symbols = tokenizer.get_symbols(context.goal)
            for i in range(len(goal_symbols) + 1, goal_arg_values.size()[2]):
                goal_arg_values[:, :, i] = -float("Inf")
            assert goal_arg_values.size() == torch.Size([1, stem_width,
                                                         self.training_args.max_length + 1]),\
                "goal_arg_values.size(): {}; stem_width: {}".format(goal_arg_values.size(),
                                                                    stem_width)

            num_probs = 1 + len(all_hyps) + self.training_args.max_length
            if len(all_hyps) > 0:
                encoded_goals = self._model.goal_encoder(goals_batch)\
                                           .view(1, 1, self.training_args.hidden_size)

                hyps_batch = LongTensor([[self.encodeStrTerm(hyp)
                                          for hyp in all_hyps]])
                assert hyps_batch.size() == torch.Size([1, num_hyps,
                                                        self.training_args.max_length])
                hyps_batch_expanded = hyps_batch.expand(stem_width, -1, -1)\
                                                .contiguous()\
                                                .view(stem_width * num_hyps,
                                                      self.training_args.max_length)

                hypfeatures_batch = encodeHypsFeatureVecs(self.training_args,
                                                          context.goal,
                                                          all_hyps)
                assert hypfeatures_batch.size() == torch.Size([num_hyps, hypFeaturesSize()])
                hypfeatures_batch_expanded = \
                    hypfeatures_batch.view(1, num_hyps, hypFeaturesSize())\
                                     .expand(stem_width, -1, -1)\
                                     .contiguous()\
                                     .view(stem_width * num_hyps, hypFeaturesSize())
                hyp_arg_values = self.runHypModel(stem_idxs,
                                                  encoded_goals, hyps_batch,
                                                  hypfeatures_batch)
                assert hyp_arg_values.size() == \
                    torch.Size([1, stem_width, len(all_hyps)])
                total_values = torch.cat((goal_arg_values, hyp_arg_values), dim=2)
            else:
                total_values = goal_arg_values
            all_prob_batches = self._softmax((total_values +
                                              stem_certainties.view(1, stem_width, 1)
                                              .expand(-1, -1, num_probs))
                                             .contiguous()
                                             .view(1, stem_width * num_probs))\
                                   .view(stem_width * num_probs)

            final_probs, final_idxs = all_prob_batches.topk(k)
            assert not torch.isnan(final_probs).any()
            assert final_probs.size() == torch.Size([k])
            row_length = self.training_args.max_length + len(all_hyps) + 1
            stem_keys = final_idxs / row_length
            assert stem_keys.size() == torch.Size([k])
            assert stem_idxs.size() == torch.Size([1, stem_width]), stem_idxs.size()
            prediction_stem_idxs = stem_idxs.view(stem_width).index_select(0, stem_keys)
            assert prediction_stem_idxs.size() == torch.Size([k]), \
                prediction_stem_idxs.size()
            arg_idxs = final_idxs % row_length
            assert arg_idxs.size() == torch.Size([k])
            return [Prediction(self.decodePrediction(context.goal,
                                                     all_hyps,
                                                     stem_idx.item(),
                                                     arg_idx.item()),
                               math.exp(prob))
                    for stem_idx, arg_idx, prob in
                    islice(zip(prediction_stem_idxs, arg_idxs, final_probs), k)]
    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) -> \
        Tuple[List[Prediction], float]:
        return self.predictKTactics(in_data, k), 0
    def predictKTacticsWithLoss_batch(self,
                                      in_datas : List[TacticContext],
                                      k : int, corrects : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        subresults = [self.predictKTacticsWithLoss(in_data, k, correct)
                      for in_data, correct in
                      zip(in_datas, corrects)]
        loss = sum([subresult[1] for subresult in subresults])
        predictions = [subresult[0] for subresult in subresults]
        return predictions, loss

    def predictTactic(self, context : TacticContext) -> Prediction:
        assert self.training_args
        assert self._model

        word_features, vec_features = self.encodeFeatureVecs([context])
        goals_batch = LongTensor([self.encodeStrTerm(context.goal)])
        stem_distribution = self._model.stem_classifier(word_features, vec_features)
        stem_certainties, stem_idxs = stem_distribution.topk(1)
        goal_arg_values = self._model.goal_args_model(stem_idxs, goals_batch)
        goal_symbols = tokenizer.get_symbols(context.goal)
        for i in range(len(goal_symbols) + 1, goal_arg_values.size()[1]):
            goal_arg_values[0, i] = -float("Inf")
        encoded_goals = self._model.goal_encoder(goals_batch)
        if self.training_args.lemma_args:
            all_hyps = context.hypotheses + context.relevant_lemmas
        else:
            all_hyps = context.hypotheses
        if len(all_hyps) > 0:
            hyps_batch = LongTensor([[self.encodeStrTerm(hyp)
                                      for hyp in all_hyps]])
            hypfeatures_batch = encodeHypsFeatureVecs(self.training_args,
                                                      context.goal,
                                                      all_hyps)
            hyp_arg_values = self.runHypModel(stem_idxs, encoded_goals, hyps_batch,
                                              hypfeatures_batch)\
                                 .view(1, len(all_hyps))
            total_arg_values = torch.cat((goal_arg_values, hyp_arg_values),
                                         dim=1)
        else:
            total_arg_values = goal_arg_values

        total_arg_distribution = self._softmax(total_arg_values)
        total_arg_certainties, total_arg_idxs = total_arg_distribution.topk(1)
        probability = math.exp(stem_certainties[0] +
                               total_arg_certainties[0])
        return Prediction(self.decodePrediction(context.goal,
                                                all_hyps,
                                                stem_idxs[0].item(),
                                                total_arg_idxs[0].item()),
                          probability)
    def encodeFeatureVecs(self, contexts : List[TacticContext])\
        -> Tuple[torch.LongTensor, torch.FloatTensor]:
        assert self._word_feature_functions
        assert self._vec_feature_functions
        if self.training_args.features:
            word_features = LongTensor([[feature(c) for feature in
                                         self._word_feature_functions]
                                        for c in contexts])
            vec_features = FloatTensor([[feature_val for feature in
                                         self._vec_feature_functions
                                         for feature_val in feature(c)]
                                        for c in contexts])
        else:
            word_features = LongTensor([[0 for feature in
                                         self._word_feature_functions]
                                        for c in contexts])
            vec_features = FloatTensor ([[0 for feature in
                                          self._vec_feature_functions
                                          for feature_val in feature(c)]
                                         for c in contexts])
        return word_features, vec_features
    def encodeStrTerm(self, term : str) -> List[int]:
        assert self._tokenizer
        assert self.training_args
        return normalizeSentenceLength(
            self._tokenizer.toTokenList(term),
            self.training_args.max_length)
    def runHypModel(self, stem_idxs : torch.LongTensor, encoded_goals : torch.FloatTensor,
                    hyps_batch : torch.LongTensor, hypfeatures_batch : torch.FloatTensor):
        assert self._model
        assert self.training_args
        batch_size = encoded_goals.size()[0]
        assert batch_size == 1
        num_hyps = hyps_batch.size()[1]
        beam_width = stem_idxs.size()[1]
        features_size = hypfeatures_batch.size()[1]
        hyp_arg_values = \
            self._model.hyp_model(stem_idxs.view(batch_size, beam_width, 1)
                                  .expand(-1, -1, num_hyps).contiguous()
                                  .view(batch_size * beam_width * num_hyps),
                                  encoded_goals.view(batch_size, 1,
                                                     self.training_args.hidden_size)
                                  .expand(-1, beam_width * num_hyps, -1)
                                  .contiguous()
                                  .view(batch_size * beam_width * num_hyps,
                                        self.training_args.hidden_size),
                                  hyps_batch.view(batch_size, 1, num_hyps,
                                                  self.training_args.max_length)
                                  .expand(-1, beam_width, -1, -1).contiguous()
                                  .view(batch_size * beam_width * num_hyps,
                                        self.training_args.max_length),
                                  hypfeatures_batch
                                  .view(batch_size, 1, num_hyps, features_size)
                                  .expand(-1, beam_width, -1, -1).contiguous()
                                  .view(batch_size * beam_width * num_hyps,
                                        features_size))\
                                  .view(batch_size, beam_width, num_hyps)
        return hyp_arg_values
    def decodePrediction(self, goal : str, hyps : List[str], stem_idx : int,
                         total_arg_idx : int):
        assert self._embedding
        assert self.training_args
        stem = self._embedding.decode_token(stem_idx)
        max_goal_symbols = self.training_args.max_length
        if total_arg_idx == 0:
            return stem + "."
        elif total_arg_idx - 1 < max_goal_symbols:
            return stem + " " + tokenizer.get_symbols(goal)[total_arg_idx - 1] + "."
        else:
            return stem + " " + serapi_instance.get_var_term_in_hyp(
                hyps[total_arg_idx - 1 - max_goal_symbols]) + "."

    def getOptions(self) -> List[Tuple[str, str]]:
        assert self.training_args
        assert self.training_loss
        assert self.num_epochs
        return list(vars(self.training_args).items()) + \
            [("training loss", self.training_loss),
             ("# epochs", self.num_epochs),
             ("predictor", "polyarg")]
    def _description(self) -> str:
        return "A predictor combining the goal token args and hypothesis args models."
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) -> None:
        new_defaults = {"batch-size":128, "learning-rate":0.4, "epoch-step":3,
                        **default_values}
        super().add_args_to_parser(parser, new_defaults)
        add_nn_args(parser, new_defaults)
        add_tokenizer_args(parser, new_defaults)
        feature_set : Set[str] = set()
        all_constructors : List[Type[Feature]] = vec_feature_constructors + word_feature_constructors # type: ignore
        for feature_constructor in all_constructors:
            new_args = feature_constructor\
                .add_feature_arguments(parser, feature_set, default_values)
            feature_set = feature_set.union(new_args)
        parser.add_argument("--max-length", dest="max_length", type=int,
                            default=default_values.get("max-length", 30))
        parser.add_argument("--max-beam-width", dest="max_beam_width", type=int,
                            default=default_values.get("max-beam-width", 10))
        parser.add_argument("--no-lemma-args", dest="lemma_args", action='store_false')
        parser.add_argument("--no-hyp-features", dest="hyp_features", action="store_false")
        parser.add_argument("--no-features", dest="features", action="store_false")
        parser.add_argument("--no-hyp-rnn", dest="hyp_rnn", action="store_false")
        parser.add_argument("--no-goal-rnn", dest="goal_rnn", action="store_false")
        parser.add_argument("--replace-rnns-with-dnns", action="store_true")

    def _encode_data(self, data : RawDataset, arg_values : Namespace) \
        -> Tuple[FeaturesPolyArgDataset, Tuple[Tokenizer, Embedding,
                                               List[WordFeature], List[VecFeature]]]:
        embedding : Embedding
        tokenizer : Tokenizer
        if arg_values.start_from:
            predictor_name, (loaded_args, metadata, predictor_state) = \
                torch.load(arg_values.start_from)
            assert predictor_name == "polyarg"
            tokenizer, embedding, wfeats, vfeats = metadata
            _, embedded_data = embed_data(data, embedding)
            _, tokenized_goals = tokenize_goals(embedded_data,
                                                arg_values, tokenizer)
            self._word_feature_functions = wfeats
            self._vec_feature_functions = vfeats
        else:
            stripped_data = [strip_scraped_output(dat)
                             for dat in data]
            self._word_feature_functions = \
                [feature_constructor(stripped_data, arg_values)  # type: ignore
                 for feature_constructor in
                 word_feature_constructors]
            self._vec_feature_functions = \
                [feature_constructor(stripped_data, arg_values) for # type: ignore
                 feature_constructor in vec_feature_constructors]
            embedding, embedded_data = embed_data(data)
            tokenizer, tokenized_goals = tokenize_goals(embedded_data,
                                                        arg_values)
        torch.multiprocessing.set_sharing_strategy('file_system')
        with multiprocessing.Pool(arg_values.num_threads) as pool:
            start = time.time()
            print("Creating dataset...", end="")
            sys.stdout.flush()
            result_data = FeaturesPolyArgDataset(list(pool.imap(
                functools.partial(mkFPASample, embedding,
                                  tokenizer,
                                  arg_values,
                                  self._word_feature_functions,
                                  self._vec_feature_functions),
                zip(data, tokenized_goals))))
            print("{:.2f}s".format(time.time() - start))
        assert self._word_feature_functions
        assert self._vec_feature_functions
        return result_data, (tokenizer, embedding, self._word_feature_functions,
                             self._vec_feature_functions)
    def _optimize_model_to_disc(self,
                                encoded_data : FeaturesPolyArgDataset,
                                metadata : Tuple[Tokenizer, Embedding,
                                                 List[WordFeature], List[VecFeature]],
                                arg_values : Namespace) \
        -> None:
        tokenizer, embedding, word_features, vec_features = metadata
        save_checkpoints("polyarg",
                         metadata, arg_values,
                         self._optimize_checkpoints(encoded_data, arg_values,
                                                    tokenizer, embedding))
    def _optimize_checkpoints(self, encoded_data : FeaturesPolyArgDataset,
                              arg_values : Namespace,
                              tokenizer : Tokenizer,
                              embedding : Embedding) \
        -> Iterable[NeuralPredictorState]:
        return optimize_checkpoints(self._data_tensors(encoded_data, arg_values),
                                    arg_values,
                                    self._get_model(arg_values, embedding.num_tokens(),
                                                    tokenizer.numTokens()),
                                    lambda batch_tensors, model:
                                    self._getBatchPredictionLoss(arg_values,
                                                                 batch_tensors,
                                                                 model))
    def load_saved_state(self,
                         args : Namespace,
                         metadata : Tuple[Tokenizer, Embedding,
                                          List[WordFeature], List[VecFeature]],
                         state : NeuralPredictorState) -> None:
        self._tokenizer, self._embedding, \
            self._word_feature_functions, self._vec_feature_functions= \
                metadata
        model = maybe_cuda(self._get_model(args,
                                           self._embedding.num_tokens(),
                                           self._tokenizer.numTokens()))
        model.load_state_dict(state.weights)
        self._model = model
        self.training_loss = state.loss
        self.num_epochs = state.epoch
        self.training_args = args
    def _data_tensors(self, encoded_data : FeaturesPolyArgDataset,
                      arg_values : Namespace) \
        -> List[torch.Tensor]:
        tokenized_hyp_types, hyp_features, tokenized_goals, \
            word_features, vec_features, tactic_stems, \
            arg_types, args = zip(*sorted(encoded_data,
                                          key=lambda s: len(s.tokenized_hyp_types),
                                          reverse=True))
        padded_hyps = pad_sequence([torch.LongTensor(tokenized_hyps_list)
                                    for tokenized_hyps_list
                                    in tokenized_hyp_types],
                                   batch_first=True)
        padded_hyp_features = pad_sequence([torch.FloatTensor(hyp_features_list)
                                            for hyp_features_list
                                            in hyp_features],
                                           batch_first=True)
        for arg, arg_type, hlist in zip(args, arg_types, tokenized_hyp_types):
            if arg_type == ArgType.GOAL_TOKEN:
                assert arg.token_idx < arg_values.max_length
            elif arg_type == ArgType.HYP_ID:
                assert arg.hyp_idx < len(hlist)
        result = [padded_hyps,
                  padded_hyp_features,
                  torch.LongTensor([len(tokenized_hyp_type_list)
                                    for tokenized_hyp_type_list
                                    in tokenized_hyp_types]),
                  torch.LongTensor(tokenized_goals),
                  torch.LongTensor(word_features),
                  torch.FloatTensor(vec_features),
                  torch.LongTensor(tactic_stems),
                  torch.LongTensor([
                      0 if arg_type == ArgType.NO_ARG else
                      (arg.token_idx + 1) if arg_type == ArgType.GOAL_TOKEN else
                      (arg.hyp_idx + arg_values.max_length + 1)
                      for arg_type, arg in zip(arg_types, args)])]
        return result
    def _get_model(self, arg_values : Namespace,
                   stem_vocab_size : int,
                   goal_vocab_size : int) \
        -> FeaturesPolyArgModel:
        assert self._word_feature_functions
        assert self._vec_feature_functions
        feature_vec_size = sum([feature.feature_size()
                                for feature in self._vec_feature_functions])
        word_feature_vocab_sizes = [feature.vocab_size()
                                    for feature in self._word_feature_functions]
        return FeaturesPolyArgModel(
            FeaturesClassifier(self._word_feature_functions,
                               self._vec_feature_functions,
                               arg_values.hidden_size,
                               arg_values.num_layers,
                               stem_vocab_size),
            GoalTokenArgModel(stem_vocab_size, goal_vocab_size, arg_values.max_length,
                              arg_values.hidden_size),
            EncoderRNN(goal_vocab_size, arg_values.hidden_size, arg_values.hidden_size),
            HypArgModel(arg_values.hidden_size, stem_vocab_size, goal_vocab_size,
                        hypFeaturesSize(), arg_values.hidden_size))
    def _getBatchPredictionLoss(self, arg_values : Namespace,
                                data_batch : Sequence[torch.Tensor],
                                model : FeaturesPolyArgModel) -> torch.FloatTensor:
        tokenized_hyp_types_batch, hyp_features_batch, num_hyps_batch, \
            tokenized_goals_batch, \
            word_features_batch, vec_features_batch, \
            stem_idxs_batch, arg_total_idxs_batch = \
                cast(Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor,
                           torch.LongTensor,
                           torch.LongTensor, torch.FloatTensor,
                           torch.LongTensor, torch.LongTensor],
                     data_batch)
        batch_size = tokenized_goals_batch.size()[0]
        goal_size = tokenized_goals_batch.size()[1]
        stemDistributions = model.stem_classifier(word_features_batch, vec_features_batch)
        num_stem_poss = stemDistributions.size()[1]
        stem_width = min(arg_values.max_beam_width, num_stem_poss)
        stem_var = maybe_cuda(Variable(stem_idxs_batch))
        predictedProbs, predictedStemIdxs = stemDistributions.topk(stem_width)
        mergedStemIdxs = []
        for stem_idx, predictedStemIdxList in zip(stem_idxs_batch, predictedStemIdxs):
            if stem_idx.item() in predictedStemIdxList:
                mergedStemIdxs.append(predictedStemIdxList)
            else:
                mergedStemIdxs.append(
                    torch.cat((stem_idx.view(1).cuda(),
                               predictedStemIdxList[:stem_width-1])))
        mergedStemIdxsT = torch.stack(mergedStemIdxs)
        correctPredictionIdxs = torch.LongTensor([list(idxList).index(stem_idx) for
                                                  idxList, stem_idx
                                                  in zip(mergedStemIdxs, stem_var)])
        if arg_values.hyp_rnn:
            tokenized_hyps_var = maybe_cuda(Variable(tokenized_hyp_types_batch))
        else:
            tokenized_hyps_var = maybe_cuda(Variable(torch.zeros_like(tokrnized_hyp_types_batch)))

        if arg_values.hyp_features:
            hyp_features_var = maybe_cuda(Variable(hyp_features_batch))
        else:
            hyp_features_var = maybe_cuda(Variable(torch.zeros_like(hyp_features_batch)))

        goal_arg_values = model.goal_args_model(
            mergedStemIdxsT.view(batch_size * stem_width),
            tokenized_goals_batch.view(batch_size, 1, goal_size).expand(-1, stem_width, -1)
            .contiguous().view(batch_size * stem_width, goal_size))\
            .view(batch_size, stem_width, goal_size + 1)
        encoded_goals = model.goal_encoder(tokenized_goals_batch)

        hyp_lists_length = tokenized_hyp_types_batch.size()[1]
        hyp_length = tokenized_hyp_types_batch.size()[2]
        hyp_features_size = hyp_features_batch.size()[2]
        encoded_goal_size = encoded_goals.size()[1]

        encoded_goals_expanded = \
            encoded_goals.view(batch_size, 1, 1, encoded_goal_size)\
            .expand(-1, stem_width, hyp_lists_length, -1).contiguous()\
            .view(batch_size * stem_width * hyp_lists_length, encoded_goal_size)
        if not arg_values.goal_rnn:
            encoded_goals_expanded = torch.zeros_like(encoded_goals_expanded)
        stems_expanded = \
            mergedStemIdxsT.view(batch_size, stem_width, 1)\
            .expand(-1, -1, hyp_lists_length).contiguous()\
            .view(batch_size * stem_width * hyp_lists_length)
        hyp_arg_values_concatted = \
            model.hyp_model(stems_expanded,
                            encoded_goals_expanded,
                            tokenized_hyps_var
                            .view(batch_size, 1, hyp_lists_length, hyp_length)
                            .expand(-1, stem_width, -1, -1).contiguous()
                            .view(batch_size * stem_width * hyp_lists_length,
                                  hyp_length),
                            hyp_features_var
                            .view(batch_size, 1, hyp_lists_length, hyp_features_size)
                            .expand(-1, stem_width, -1, -1).contiguous()
                            .view(batch_size * stem_width * hyp_lists_length,
                                  hyp_features_size))
        assert hyp_arg_values_concatted.size() == torch.Size([batch_size * stem_width * hyp_lists_length, 1]), hyp_arg_values_concatted.size()
        hyp_arg_values = hyp_arg_values_concatted.view(batch_size, stem_width,
                                                       hyp_lists_length)
        total_arg_values = torch.cat((goal_arg_values, hyp_arg_values),
                                     dim=2)
        num_probs = hyp_lists_length + goal_size + 1
        total_arg_distribution = \
            self._softmax(total_arg_values.view(batch_size, stem_width * num_probs))
        total_arg_var = maybe_cuda(Variable(arg_total_idxs_batch +
                                            (correctPredictionIdxs * num_probs)))\
                                            .view(batch_size)
        loss = FloatTensor([0.])
        loss += self._criterion(stemDistributions, stem_var)
        loss += self._criterion(total_arg_distribution, total_arg_var)
        return loss

def mkFPASample(embedding : Embedding,
                mytokenizer : Tokenizer,
                training_args : argparse.Namespace,
                word_feature_functions : List[WordFeature],
                vec_feature_functions : List[VecFeature],
                zipped : Tuple[ScrapedTactic, List[int]]) \
                -> FeaturesPolyArgSample:
    inter, tokenized_goal = zipped
    relevant_lemmas, prev_tactics, hypotheses, goal_str, tactic = inter
    context = strip_scraped_output(inter)
    if training_args.features:
        word_features = [feature(context) for feature in word_feature_functions]
        vec_features = [feature_val for feature in vec_feature_functions
                        for feature_val in feature(context)]
    else:
        word_features = [0 for feature in word_feature_functions]
        vec_features = [0 for feature in vec_feature_functions
                        for feature_val in feature(context)]
    tactic_stem, tactic_argstr = serapi_instance.split_tactic(tactic)
    stem_idx = embedding.encode_token(tactic_stem)
    argstr_tokens = tactic_argstr.strip(".").split()
    assert len(argstr_tokens) < 2, \
        "Tactic {} doesn't fit our argument model! Too many tokens" .format(tactic)
    arg : TacticArg
    if training_args.lemma_args:
        all_hyps = hypotheses + relevant_lemmas
    else:
        all_hyps = hypotheses
    if len(argstr_tokens) == 0:
        arg_type = ArgType.NO_ARG
        arg = None
        if len(all_hyps) > training_args.max_premises:
            selected_hyps = random.sample(all_hyps, training_args.max_premises)
        else:
            selected_hyps = all_hyps
    else:
        goal_symbols = tokenizer.get_symbols(goal_str)[:training_args.max_length]
        arg_token = argstr_tokens[0]
        if arg_token in goal_symbols:
            arg_type = ArgType.GOAL_TOKEN
            arg_idx = goal_symbols.index(arg_token)
            assert arg_idx < training_args.max_length, "Tactic {} doesn't fit our argument model! "\
                "Token {} is not a hyp var or goal token.\n"\
                "Hyps: {}\n"\
                "Goal: {}".format(tactic, arg_token, all_hyps, goal_str)
            arg = GoalTokenArg(goal_symbols.index(arg_token))
            if len(all_hyps) > training_args.max_premises:
                selected_hyps = random.sample(all_hyps, training_args.max_premises)
            else:
                selected_hyps = all_hyps
        else:
            indexed_hyp_vars = serapi_instance.get_indexed_vars_in_hyps(all_hyps)
            hyp_vars = [hyp_var for hyp_var, idx in indexed_hyp_vars]
            assert arg_token in hyp_vars, "Tactic {} doesn't fit our argument model! "\
                "Token {} is not a hyp var or goal token.\n"\
                "Hyps: {}\n"\
                "Goal: {}".format(tactic, arg_token, all_hyps, goal_str)
            arg_type = ArgType.HYP_ID
            correct_hyp_idx = dict(indexed_hyp_vars)[arg_token]
            if len(all_hyps) > training_args.max_premises:
                selected_hyps = random.sample(all_hyps[:correct_hyp_idx]+
                                              all_hyps[correct_hyp_idx+1:],
                                              training_args.max_premises-1)
                new_hyp_idx = random.randrange(training_args.max_premises)
                selected_hyps.insert(new_hyp_idx,
                                     all_hyps[correct_hyp_idx])
                arg = HypIdArg(new_hyp_idx)
            else:
                selected_hyps = all_hyps
                arg = HypIdArg(correct_hyp_idx)
    if len(selected_hyps) == 0:
        selected_hyps = [":"]
    tokenized_hyp_types = [normalizeSentenceLength(
        mytokenizer.toTokenList(serapi_instance.get_hyp_type(hyp)),
        training_args.max_length)
                           for hyp in selected_hyps]
    hypfeatures = encodeHypsFeatureVecs(training_args, goal_str, selected_hyps)
    return FeaturesPolyArgSample(
        tokenized_hyp_types,
        hypfeatures,
        normalizeSentenceLength(tokenized_goal, training_args.max_length),
        word_features,
        vec_features,
        stem_idx,
        arg_type,
        arg)

def hypFeaturesSize() -> int:
    return 2
def encodeHypsFeatureVecs(args : argparse.Namespace,
                          goal : str, hyps : List[str]) -> torch.FloatTensor:
    def features(hyp : str):
        similarity_ratio = SequenceMatcher(None, goal,
                                           serapi_instance.get_hyp_type(hyp)).ratio()
        is_equals_on_goal_token = 0.0
        equals_match = re.match("eq\s+(.*)", hyp)
        if equals_match:
            left_side, right_side = \
                split_by_char_outside_matching(
                    "\(", "\)", "\s*",
                    equals_match.group(1))
            if left_side in goal:
                is_equals_on_goal_token = -1.0
            elif right_side in goal:
                is_equals_on_goal_token = 1.0

        if args.features:
            return [similarity_ratio,
                    is_equals_on_goal_token]
        else:
            return [0.0, 0.0]

    return torch.FloatTensor([features(hyp) for hyp in hyps])

def main(arg_list : List[str]) -> None:
    predictor = FeaturesPolyargPredictor()
    predictor.train(arg_list)
