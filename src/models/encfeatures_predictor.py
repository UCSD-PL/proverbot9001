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
from models.tactic_predictor import \
    (NeuralPredictorState, TrainablePredictor, Prediction,
     save_checkpoints, optimize_checkpoints, predictKTactics,
     predictKTacticsWithLoss, predictKTacticsWithLoss_batch,
     add_tokenizer_args, embed_data, tokenize_goals)

from models.components import (Embedding, SimpleEmbedding, add_nn_args)
from data import (Sentence, ListDataset, RawDataset,
                  normalizeSentenceLength)
from serapi_instance import get_stem
from util import *
from format import TacticContext, strip_scraped_output
from tokenizer import Tokenizer
from features import (vec_feature_constructors,
                      word_feature_constructors, VecFeature,
                      WordFeature, Feature)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from typing import (List, Any, Tuple, NamedTuple, Dict, Sequence,
                    cast, Optional)
from dataclasses import dataclass
import threading
import argparse
from argparse import Namespace

class EncFeaturesSample(NamedTuple):
    vec_features : List[float]
    word_features : List[int]
    goal : Sentence
    next_tactic : int

class EncFeaturesDataset(ListDataset[EncFeaturesSample]):
    pass

class EncFeaturesClassifier(nn.Module):
    def __init__(self,
                 vec_features_size : int,
                 word_feature_vocab_sizes : List[int],
                 goal_vocab_size : int,
                 hidden_size : int,
                 word_embedding_size : int,
                 tactic_vocab_size : int,
                 num_encoder_layers : int# ,
                 # num_decoder_layers : int
    ) -> None:
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.word_embedding_size = word_embedding_size
        self.num_word_features = len(word_feature_vocab_sizes)
        # self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self._features_in_layer = maybe_cuda(nn.Linear(
            vec_features_size +
            self.num_word_features * word_embedding_size,
            hidden_size))
        for i, vocab_size in enumerate(word_feature_vocab_sizes):
            self.add_module("_word_embedding{}".format(i),
                            maybe_cuda(nn.Embedding(vocab_size, word_embedding_size)))
        for i in range(num_encoder_layers- 1):
            self.add_module("_features_encoder_layer{}".format(i),
                            maybe_cuda(nn.Linear(hidden_size, hidden_size)))
        self._goal_token_embedding = \
            maybe_cuda(nn.Embedding(goal_vocab_size, hidden_size))
        self._goal_gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self._decoder_in_layer = maybe_cuda(nn.Linear(hidden_size * 2, hidden_size))
        # for i in range(num_decoder_layers - 1):
        #     self.add_module("_decoder_layer{}".format(i),
        #                     maybe_cuda(nn.Linear(hidden_size, hidden_size)))
        self._decoder_out_layer = maybe_cuda(nn.Linear(hidden_size, tactic_vocab_size))
        self._softmax = maybe_cuda(nn.LogSoftmax(dim=1))

    def forward(self, vec_features_batch : torch.FloatTensor,
                word_features_batch : torch.LongTensor,
                goals_batch : torch.LongTensor) \
                -> torch.FloatTensor:

        batch_size = vec_features_batch.size()[0]
        assert goals_batch.size()[0] == batch_size
        goals_var = maybe_cuda(Variable(goals_batch))
        hidden = maybe_cuda(Variable(torch.zeros(1, batch_size, self.hidden_size)))
        for i in range(goals_batch.size()[1]):
            token_batch = self._goal_token_embedding(goals_var[:,i])\
                              .view(1, batch_size, self.hidden_size)
            token_batch = F.relu(token_batch)
            token_out, hidden = self._goal_gru(token_batch, hidden)
        goal_data = token_out[0]

        vec_features_var = maybe_cuda(Variable(vec_features_batch))
        word_embedded_features = []
        for i in range(self.num_word_features):
            word_feature_var = maybe_cuda(Variable(word_features_batch[:,i]))
            embedded = getattr(self, "_word_embedding{}".format(i))(word_feature_var)\
                .view(batch_size, self.word_embedding_size)
            word_embedded_features.append(embedded)
        word_embedded_features_vec = \
            torch.cat(word_embedded_features, dim=1)

        features_data = self._features_in_layer(
            torch.cat((vec_features_var, word_embedded_features_vec), dim=1))
        for i in range(self.num_encoder_layers - 1):
            features_data = F.relu(features_data)
            features_data = \
                getattr(self, "_features_encoder_layer{}".format(i))(features_data)

        full_data = self._decoder_in_layer(F.relu(torch.cat((goal_data, features_data), dim=1)))
        # for i in range(self.num_decoder_layers - 1):
        #     full_data = F.relu(full_data)
        #     full_data = \
        #         getattr(self, "_decoder_layer{}".format(i))(full_data)
        full_data = F.relu(full_data)

        result = self._softmax(self._decoder_out_layer(full_data)).view(batch_size, -1)
        return result

class EncFeaturesPredictor(TrainablePredictor[EncFeaturesDataset,
                                              Tuple[Tokenizer, Embedding,
                                                    List[VecFeature], List[WordFeature]],
                                              NeuralPredictorState]):
    def __init__(self) \
        -> None:
        self._vec_feature_functions : Optional[List[VecFeature]] = None
        self._word_feature_functions : Optional[List[WordFeature]] = None
        self._criterion = maybe_cuda(nn.NLLLoss())
        self._lock = threading.Lock()

    def _predictDistributions(self, in_datas : List[TacticContext]) -> torch.FloatTensor:
        assert self.training_args
        vec_features_batch = [self._get_vec_features(in_data) for in_data in in_datas]
        word_features_batch = [self._get_word_features(in_data) for in_data in in_datas]
        goals_batch = [normalizeSentenceLength(self._tokenizer.toTokenList(goal),
                                               self.training_args.max_length)
                       for _, _, _, goal in in_datas]
        return self._model(torch.FloatTensor(vec_features_batch),
                           torch.LongTensor(word_features_batch),
                           torch.LongTensor(goals_batch))
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) -> None:
        super().add_args_to_parser(parser, default_values)
        add_nn_args(parser, default_values)
        add_tokenizer_args(parser, default_values)
        parser.add_argument("--max-length", dest="max_length", type=int,
                            default=default_values.get("max-length", 100))
        parser.add_argument("--num-encoder-layers", dest="num_encoder_layers", type=int,
                            default=default_values.get("num-encoder-layers", 3))
        # parser.add_argument("--num-decoder-layers", dest="num_decoder_layers", type=int,
        #                     default=default_values.get("num-decoder-layers", 2))
        parser.add_argument("--num-head-keywords", dest="num_head_keywords", type=int,
                            default=default_values.get("num-head-keywords", 100))
        parser.add_argument("--num-tactic-keywords", dest="num_tactic_keywords", type=int,
                            default=default_values.get("num-tactic-keywords", 50))
        parser.add_argument("--word-embedding-size", dest="word_embedding_size", type=int,
                            default=default_values.get("word_embedding_size", 10))
    def _get_vec_features(self, context : TacticContext) -> List[float]:
        assert self._vec_feature_functions
        return [feature_val for feature in self._vec_feature_functions
                for feature_val in feature(context)]
    def _get_word_features(self, context : TacticContext) -> List[int]:
        assert self._word_feature_functions
        return [feature(context) for feature in self._word_feature_functions]

    def _encode_data(self, data : RawDataset, arg_values : Namespace) \
        -> Tuple[EncFeaturesDataset, Tuple[Tokenizer, Embedding,
                                           List[VecFeature], List[WordFeature]]]:
        stripped_data = [strip_scraped_output(dat) for dat in data]
        self._vec_feature_functions = [feature_constructor(stripped_data, arg_values) for # type: ignore
                                       feature_constructor in vec_feature_constructors]
        self._word_feature_functions = [feature_constructor(stripped_data, arg_values) for # type: ignore
                                       feature_constructor in word_feature_constructors]
        embedding, embedded_data = embed_data(data)
        tokenizer, tokenized_goals = tokenize_goals(embedded_data, arg_values)
        result_data = EncFeaturesDataset([EncFeaturesSample(
            self._get_vec_features(TacticContext([], prev_tactics, hypotheses, goal)),
            self._get_word_features(TacticContext([], prev_tactics, hypotheses, goal)),
            normalizeSentenceLength(tokenized_goal, arg_values.max_length),
            tactic)
                                           for (relevant_lemmas, prev_tactics, hypotheses, goal, tactic),
                                           tokenized_goal in
                                           zip(embedded_data, tokenized_goals)])
        return result_data, (tokenizer, embedding,
                             self._vec_feature_functions, self._word_feature_functions)
    def _optimize_model_to_disc(self,
                                encoded_data : EncFeaturesDataset,
                                metadata : Tuple[Tokenizer, Embedding,
                                                 List[VecFeature], List[WordFeature]],
                                arg_values : Namespace) \
        -> None:
        tokenizer, embedding, vec_features, word_features = metadata
        save_checkpoints("encfeatures", metadata, arg_values,
                         self._optimize_checkpoints(encoded_data, arg_values,
                                                    tokenizer, embedding))
    def _optimize_checkpoints(self, encoded_data : EncFeaturesDataset,
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
                                          List[VecFeature], List[WordFeature]],
                         state : NeuralPredictorState) -> None:
        self._tokenizer, self._embedding, \
            self._vec_feature_functions, self._word_feature_functions = \
                metadata
        self._model = maybe_cuda(self._get_model(args,
                                                 self._embedding.num_tokens(),
                                                 self._tokenizer.numTokens()))
        self._model.load_state_dict(state.weights)
        self.training_loss = state.loss
        self.num_epochs = state.epoch
        self.training_args = args
        self.unparsed_args = unparsed_args
    def _data_tensors(self, encoded_data : EncFeaturesDataset,
                      arg_values : Namespace) \
        -> List[torch.Tensor]:
        vec_features, word_features, goals, tactics = zip(*encoded_data)
        return [torch.FloatTensor(vec_features),
                torch.LongTensor(word_features),
                torch.LongTensor(goals),
                torch.LongTensor(tactics)]
    def _get_model(self, arg_values : Namespace,
                   tactic_vocab_size : int,
                   goal_vocab_size : int) \
        -> EncFeaturesClassifier:
        assert self._vec_feature_functions
        assert self._word_feature_functions
        feature_vec_size = sum([feature.feature_size()
                                for feature in self._vec_feature_functions])
        word_feature_vocab_sizes = [feature.vocab_size()
                                    for feature in self._word_feature_functions]
        return EncFeaturesClassifier(feature_vec_size,
                                     word_feature_vocab_sizes,
                                     goal_vocab_size,
                                     arg_values.hidden_size,
                                     arg_values.word_embedding_size,
                                     tactic_vocab_size,
                                     arg_values.num_encoder_layers# ,
                                     # arg_values.num_decoder_layers
        )

    def _getBatchPredictionLoss(self, data_batch : Sequence[torch.Tensor],
                                model : EncFeaturesClassifier) \
        -> torch.FloatTensor:
        vec_features_batch, word_features_batch, goals_batch, output_batch = \
            cast(Tuple[torch.FloatTensor, torch.LongTensor,
                       torch.LongTensor, torch.LongTensor],
                 data_batch)
        predictionDistribution = model(vec_features_batch, word_features_batch,
                                       goals_batch)
        output_var = maybe_cuda(Variable(output_batch))
        return self._criterion(predictionDistribution, output_var)
    def predictKTactics(self, in_data : TacticContext, k : int) \
        -> List[Prediction]:
        with self._lock:
            return predictKTactics(self._predictDistributions([in_data])[0],
                                   self._embedding, k)
    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) -> \
        Tuple[List[Prediction], float]:
        with self._lock:
            return predictKTacticsWithLoss(self._predictDistributions([in_data])[0],
                                           self._embedding, k,
                                           correct, self._criterion)
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, correct : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        with self._lock:
            return predictKTacticsWithLoss_batch(self._predictDistributions(in_data),
                                                 self._embedding, k,
                                                 correct, self._criterion)
    def getOptions(self) -> List[Tuple[str, str]]:
        return list(vars(self.training_args).items()) + \
            [("training loss", self.training_loss),
             ("# epochs", self.num_epochs)]

    def _description(self) -> str:
        return "A predictor using an RNN on the tokenized goal and "\
            "hand-engineered features."

def main(arg_list : List[str]) -> None:
    predictor = EncFeaturesPredictor()
    predictor.train(arg_list)
