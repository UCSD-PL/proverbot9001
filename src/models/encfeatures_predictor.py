from models.tactic_predictor import (NeuralPredictorState,
                                     TrainablePredictor,
                                     TacticContext, Prediction,
                                     save_checkpoints,
                                     optimize_checkpoints,
                                     predictKTactics,
                                     predictKTacticsWithLoss,
                                     predictKTacticsWithLoss_batch,
                                     add_nn_args, add_tokenizer_args,
                                     embed_data, tokenize_goals)

from models.components import (Embedding, SimpleEmbedding)
from data import (Sentence, ListDataset, RawDataset,
                  normalizeSentenceLength)
from serapi_instance import get_stem
from util import *
from tokenizer import Tokenizer

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from typing import (List, Any, Tuple, NamedTuple, Dict, Sequence,
                    cast)
from dataclasses import dataclass
import threading
import argparse
from argparse import Namespace
from features import feature_functions

class EncFeaturesSample(NamedTuple):
    features : List[float]
    goal : Sentence
    next_tactic : int

class EncFeaturesDataset(ListDataset[EncFeaturesSample]):
    pass

class EncFeaturesClassifier(nn.Module):
    def __init__(self,
                 num_features : int,
                 goal_vocab_size : int,
                 hidden_size : int,
                 tactic_vocab_size : int,
                 num_encoder_layers : int,
                 num_decoder_layers : int) -> None:
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self._features_in_layer = maybe_cuda(nn.Linear(num_features, hidden_size))
        self._features_encoder_layers = [maybe_cuda(nn.Linear(hidden_size, hidden_size))
                                        for _ in range(num_encoder_layers - 1)]
        self._goal_token_embedding = \
            maybe_cuda(nn.Embedding(goal_vocab_size, hidden_size))
        self._goal_gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self._decoder_in_layer = maybe_cuda(nn.Linear(hidden_size * 2, hidden_size))
        self._decoder_layers = [maybe_cuda(nn.Linear(hidden_size, hidden_size))
                                for _ in range(num_decoder_layers - 1)]
        self._decoder_out_layer = maybe_cuda(nn.Linear(hidden_size, tactic_vocab_size))
        self._softmax = maybe_cuda(nn.LogSoftmax(dim=1))

    def forward(self, features_batch : torch.FloatTensor, goals_batch : torch.LongTensor) \
        -> torch.FloatTensor:

        batch_size = features_batch.size()[0]
        assert goals_batch.size()[0] == batch_size
        goals_var = maybe_cuda(Variable(goals_batch))
        hidden = maybe_cuda(Variable(torch.zeros(1, batch_size, self.hidden_size)))
        for i in range(goals_batch.size()[1]):
            token_batch = self._goal_token_embedding(goals_var[:,i])\
                              .view(1, batch_size, self.hidden_size)
            token_batch = F.relu(token_batch)
            token_out, hidden = self._goal_gru(token_batch, hidden)
        # return self._softmax(self._decoder_out_layer(token_out[0])).view(batch_size, -1)
        goal_data = token_out[0]

        features_var = maybe_cuda(Variable(features_batch))
        features_data = self._features_in_layer(features_var)
        for i in range(self.num_encoder_layers - 1):
            features_data = F.relu(features_data)
            features_data = self._features_encoder_layers[i](features_data)

        full_data = self._decoder_in_layer(torch.cat((goal_data, features_data), dim=1))
        for i in range(self.num_decoder_layers - 1):
            full_data = F.relu(full_data)
            full_data = self._decoder_layers[i](full_data)

        result = self._softmax(self._decoder_out_layer(full_data))
        return result

class EncFeaturesPredictor(TrainablePredictor[EncFeaturesDataset,
                                              Tuple[Tokenizer, Embedding],
                                              NeuralPredictorState]):
    def __init__(self) \
        -> None:
        self._feature_functions = feature_functions
        self._criterion = maybe_cuda(nn.NLLLoss())
        self._lock = threading.Lock()

    def _predictDistributions(self, in_datas : List[TacticContext]) -> torch.FloatTensor:
        features_batch = [[feature_val
                          for feature in self._feature_functions
                          for feature_val in feature(in_data)]
                         for in_data in in_datas]
        goals_batch = [normalizeSentenceLength(self._tokenizer.toTokenList(goal),
                                               self.training_args.max_length)
                       for _, _, goal in in_datas]
        return self._model(torch.FloatTensor(features_batch),
                           torch.LongTensor(goals_batch))
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) -> None:
        super().add_args_to_parser(parser, default_values)
        add_nn_args(parser, default_values)
        add_tokenizer_args(parser, default_values)
        parser.add_argument("--max-length", dest="max_length", type=int,
                            default=default_values.get("max-length", 100))
        parser.add_argument("--hidden-size", dest="hidden_size", type=int,
                            default=default_values.get("hidden-size", 128))
        parser.add_argument("--num-encoder-layers", dest="num_encoder_layers", type=int,
                            default=default_values.get("num-encoder-layers", 2))
        parser.add_argument("--num-decoder-layers", dest="num_decoder_layers", type=int,
                            default=default_values.get("num-decoder-layers", 2))

    def _encode_data(self, data : RawDataset, arg_values : Namespace) \
        -> Tuple[EncFeaturesDataset, Tuple[Tokenizer, Embedding]]:
        preprocessed_data = self._preprocess_data(data, arg_values)
        embedding, embedded_data = embed_data(RawDataset(list(preprocessed_data)))
        tokenizer, tokenized_goals = tokenize_goals(embedded_data, arg_values)
        result_data = EncFeaturesDataset([EncFeaturesSample(
            [feature_val for feature in self._feature_functions
             for feature_val in feature(TacticContext(prev_tactics, hypotheses, goal))],
            normalizeSentenceLength(tokenized_goal, arg_values.max_length),
            tactic)
                                           for (prev_tactics, hypotheses, goal, tactic),
                                           tokenized_goal in
                                           zip(embedded_data, tokenized_goals)])
        return result_data, (tokenizer, embedding)
    def _optimize_model_to_disc(self,
                                encoded_data : EncFeaturesDataset,
                                metadata : Tuple[Tokenizer, Embedding],
                                arg_values : Namespace) \
        -> None:
        tokenizer, embedding = metadata
        save_checkpoints(metadata, arg_values,
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
                         metadata : Tuple[Tokenizer, Embedding],
                         state : NeuralPredictorState) -> None:
        self._tokenizer, self._embedding = metadata
        self._model = maybe_cuda(self._get_model(args,
                                                 self._embedding.num_tokens(),
                                                 self._tokenizer.numTokens()))
        self._model.load_state_dict(state.weights)
        self.training_loss = state.loss
        self.num_epochs = state.epoch
        self.training_args = args
    def _data_tensors(self, encoded_data : EncFeaturesDataset,
                      arg_values : Namespace) \
        -> List[torch.Tensor]:
        features, goals, tactics = zip(*encoded_data)
        return [torch.FloatTensor(features),
                torch.LongTensor(goals),
                torch.LongTensor(tactics)]
    def _get_model(self, arg_values : Namespace,
                   tactic_vocab_size : int,
                   goal_vocab_size : int) \
        -> EncFeaturesClassifier:
        return EncFeaturesClassifier(len(self._feature_functions),
                                     goal_vocab_size,
                                     arg_values.hidden_size,
                                     tactic_vocab_size,
                                     arg_values.num_encoder_layers,
                                     arg_values.num_decoder_layers)

    def _getBatchPredictionLoss(self, data_batch : Sequence[torch.Tensor],
                                model : EncFeaturesClassifier) \
        -> torch.FloatTensor:
        features_batch, goals_batch, output_batch = \
            cast(Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor], data_batch)
        predictionDistribution = model(features_batch, goals_batch)
        output_var = maybe_cuda(Variable(output_batch))
        return self._criterion(predictionDistribution, output_var)
    def predictKTactics(self, in_data : TacticContext, k : int) \
        -> List[Prediction]:
        return predictKTactics(self._predictDistributions([in_data])[0],
                               self._embedding, k)
    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) -> \
        Tuple[List[Prediction], float]:
        return predictKTacticsWithLoss(self._predictDistributions([in_data])[0],
                                       self._embedding, k,
                                       correct, self._criterion)
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, correct : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
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
