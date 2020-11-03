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
from torch.nn.utils.rnn import pad_sequence

from features import (WordFeature, VecFeature, Feature,
                      word_feature_constructors, vec_feature_constructors)
from tokenizer import Tokenizer
from data import (ListDataset, RawDataset,
                  EOS_token)
from util import (eprint, maybe_cuda, LongTensor, FloatTensor,
                  ByteTensor, print_time)
import math
from format import TacticContext
from models.components import (WordFeaturesEncoder, Embedding,
                               DNNClassifier, EncoderDNN, EncoderRNN,
                               add_nn_args)
from models.tactic_predictor import (TrainablePredictor,
                                     NeuralPredictorState, Prediction,
                                     optimize_checkpoints, add_tokenizer_args)
import dataloader
from dataloader import (features_polyarg_tensors,
                        features_polyarg_tensors_with_meta,
                        sample_fpa,
                        sample_fpa_batch,
                        decode_fpa_result,
                        # decode_fpa_stem,
                        # decode_fpa_arg,
                        # features_vocab_sizes,
                        get_num_tokens,
                        get_num_indices,
                        get_word_feature_vocab_sizes,
                        get_vec_features_size,
                        DataloaderArgs,
                        get_fpa_words)

import argparse
import sys
from itertools import islice
from argparse import Namespace
from typing import (List, Tuple, NamedTuple, Optional, Sequence, Dict,
                    cast, Union, Set, Type, Any, Iterable)

from enum import Enum, auto


class ArgType(Enum):
    HYP_ID = auto()
    GOAL_TOKEN = auto()
    NO_ARG = auto()


class HypIdArg(NamedTuple):
    hyp_idx: int


class GoalTokenArg(NamedTuple):
    token_idx: int


TacticArg = Optional[Union[HypIdArg, GoalTokenArg]]


class FeaturesPolyArgSample(NamedTuple):
    num_hyps: int
    tokenized_hyp_types: List[List[int]]
    hyp_features: List[List[float]]
    tokenized_goal: List[int]
    word_features: List[int]
    vec_features: List[float]
    tactic_stem: int
    arg_type: ArgType
    arg: TacticArg


class FeaturesPolyArgDataset(ListDataset[FeaturesPolyArgSample]):
    pass


FeaturesPolyargState = Tuple[Any, NeuralPredictorState]


class GoalTokenArgModel(nn.Module):
    def __init__(self, stem_vocab_size: int,
                 input_vocab_size: int, input_length: int,
                 hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self._stem_embedding = maybe_cuda(
            nn.Embedding(stem_vocab_size, hidden_size))
        self._token_embedding = maybe_cuda(
            nn.Embedding(input_vocab_size, hidden_size))
        self._gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self._likelyhood_layer = maybe_cuda(
            EncoderDNN(hidden_size, hidden_size, 1, 2))
        self._softmax = maybe_cuda(nn.LogSoftmax(dim=1))
    def forward(self, stem_batch: torch.LongTensor, goal_batch: torch.LongTensor) \
            -> torch.FloatTensor:
        goal_var = maybe_cuda(Variable(goal_batch))
        stem_var = maybe_cuda(Variable(stem_batch))
        batch_size = goal_batch.size()[0]
        assert stem_batch.size()[0] == batch_size
        initial_hidden = self._stem_embedding(stem_var)\
                             .view(1, batch_size, self.hidden_size)
        hidden = initial_hidden
        copy_likelyhoods: List[torch.FloatTensor] = []
        for i in range(goal_batch.size()[1]):
            try:
                token_batch = self._token_embedding(goal_var[:, i])\
                                  .view(1, batch_size, self.hidden_size)
                token_batch2 = F.relu(token_batch)
                token_out, hidden = self._gru(token_batch2, hidden)
                copy_likelyhood = self._likelyhood_layer(F.relu(token_out))
                copy_likelyhoods.append(copy_likelyhood[0])
            except RuntimeError:
                eprint("Tokenized goal:")
                for j in range(goal_batch.size()[0]):
                    eprint(goal_batch[j, i].item(), end=" ")
                    assert goal_batch[j, i] < 123
                eprint()
                eprint(f"goal_var: {goal_var}")
                eprint("Token batch")
                eprint(token_batch)
                raise
        end_token_embedded = self._token_embedding(LongTensor([EOS_token])
                                                   .expand(batch_size))\
            .view(1, batch_size, self.hidden_size)
        final_out, final_hidden = self._gru(F.relu(end_token_embedded), hidden)
        final_likelyhood = self._likelyhood_layer(F.relu(final_out))
        copy_likelyhoods.insert(0, final_likelyhood[0])
        catted = torch.cat(copy_likelyhoods, dim=1)
        return catted


class HypArgModel(nn.Module):
    def __init__(self, goal_data_size: int,
                 stem_vocab_size: int,
                 token_vocab_size: int,
                 hyp_features_size: int,
                 hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self._stem_embedding = maybe_cuda(
            nn.Embedding(stem_vocab_size, hidden_size))
        self._token_embedding = maybe_cuda(
            nn.Embedding(token_vocab_size, hidden_size))
        self._in_hidden = maybe_cuda(EncoderDNN(
            hidden_size + goal_data_size, hidden_size, hidden_size, 1))
        self._hyp_gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self._likelyhood_decoder = maybe_cuda(EncoderDNN(
            hidden_size + hyp_features_size, hidden_size, 1, 2))

    def forward(self, stems_batch: torch.LongTensor,
                goals_encoded_batch: torch.FloatTensor, hyps_batch: torch.LongTensor,
                hypfeatures_batch: torch.FloatTensor):
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
            token_batch = self._token_embedding(hyps_var[:, i])\
                .view(1, batch_size, self.hidden_size)
            token_batch = F.relu(token_batch)
            token_out, hidden = self._hyp_gru(token_batch, hidden)
        hyp_likelyhoods = self._likelyhood_decoder(
            torch.cat((token_out.view(batch_size, self.hidden_size), hypfeatures_var),
                      dim=1))
        return hyp_likelyhoods


class FeaturesClassifier(nn.Module):
    def __init__(self,
                 wordf_sizes: List[int],
                 vecf_size: int,
                 hidden_size: int,
                 num_layers: int,
                 stem_vocab_size: int)\
            -> None:
        super().__init__()
        self._word_features_encoder = maybe_cuda(
            WordFeaturesEncoder(wordf_sizes,
                                hidden_size, 1, hidden_size))
        self._features_classifier = maybe_cuda(
            DNNClassifier(hidden_size + vecf_size,
                          hidden_size, stem_vocab_size, num_layers))
        self._softmax = maybe_cuda(nn.LogSoftmax(dim=1))
        pass

    def forward(self,
                word_features_batch: torch.LongTensor,
                vec_features_batch: torch.FloatTensor) -> torch.FloatTensor:
        encoded_word_features = self._word_features_encoder(
            maybe_cuda(word_features_batch))
        stem_distribution = self._softmax(self._features_classifier(
            torch.cat((encoded_word_features, maybe_cuda(vec_features_batch)), dim=1)))
        return stem_distribution


class FeaturesPolyArgModel(nn.Module):
    def __init__(self,
                 stem_classifier: FeaturesClassifier,
                 goal_args_model: GoalTokenArgModel,
                 goal_encoder: EncoderRNN,
                 hyp_model: HypArgModel) -> None:
        super().__init__()
        self.stem_classifier = maybe_cuda(stem_classifier)
        self.goal_args_model = maybe_cuda(goal_args_model)
        self.goal_encoder = maybe_cuda(goal_encoder)
        self.hyp_model = maybe_cuda(hyp_model)


class FeaturesPolyargPredictor(
        TrainablePredictor[FeaturesPolyArgDataset,
                           Tuple[Tokenizer, Embedding,
                                 List[WordFeature], List[VecFeature]],
                           NeuralPredictorState]):
    def __init__(self) -> None:
        self._criterion = maybe_cuda(nn.NLLLoss())
        self.training_args: Optional[argparse.Namespace] = None
        self.training_loss: Optional[float] = None
        self.num_epochs: Optional[int] = None
        # self._word_feature_functions: Optional[List[WordFeature]] = None
        # self._vec_feature_functions: Optional[List[VecFeature]] = None
        self._softmax = maybe_cuda(nn.LogSoftmax(dim=1))
        self._softmax2 = maybe_cuda(nn.LogSoftmax(dim=2))
        # self._tokenizer : Optional[Tokenizer] = None
        # self._embedding : Optional[Embedding] = None
        self._model: Optional[FeaturesPolyArgModel] = None

    def train(self, args: List[str]) -> None:
        argparser = argparse.ArgumentParser(self._description())
        self.add_args_to_parser(argparser)
        arg_values = argparser.parse_args(args)
        save_states = self._optimize_model(arg_values)

        for metadata, state in save_states:
            with open(arg_values.save_file, 'wb') as f:
                torch.save((self.shortname(),
                            (arg_values, sys.argv, metadata, state)), f)

    def predictKTactics_batch(self, contexts: List[TacticContext], k: int
                              ) -> List[List[Prediction]]:
        all_predictions_batch = self.getAllPredictionIdxs_batch(contexts)

        def generate():
            for context, prediction_idxs in zip(
                    contexts, all_predictions_batch):
                predictions = self.decodeNonDuplicatePredictions(
                    context, prediction_idxs, k)

                yield predictions

        predictions = list(generate())

        for context, pred_list in zip(contexts, predictions):
            for batch_pred, single_pred in zip(
                    pred_list, self.predictKTactics(context, k)):
                assert batch_pred.prediction == single_pred.prediction, \
                    (batch_pred, single_pred)
        return predictions

    def getAllPredictionIdxs(self, context: TacticContext
                             ) -> List[Tuple[float, int, int]]:
        assert self.training_args
        assert self._model

        num_stem_poss = get_num_tokens(self._metadata)
        stem_width = min(self.training_args.max_beam_width, num_stem_poss)

        tokenized_premises, hyp_features, \
            nhyps_batch, tokenized_goal, \
            goal_mask, \
            word_features, vec_features = \
            sample_fpa(extract_dataloader_args(self.training_args),
                       self._metadata,
                       context.relevant_lemmas,
                       context.prev_tactics,
                       context.hypotheses,
                       context.goal)

        stem_certainties, stem_idxs = self.predict_stems(
            stem_width, word_features, vec_features)

        goal_arg_values = self.goal_token_scores(
            stem_idxs, tokenized_goal, goal_mask)

        if len(tokenized_premises[0]) > 0:
            hyp_arg_values = self.hyp_name_scores(
                stem_idxs[0], tokenized_goal[0],
                tokenized_premises[0], hyp_features[0])

            total_scores = torch.cat((goal_arg_values, hyp_arg_values), dim=2)
        else:
            total_scores = goal_arg_values

        final_probs, predicted_stem_idxs, predicted_arg_idxs = \
            self.predict_args(total_scores, stem_certainties, stem_idxs)

        result = list(zip(list(final_probs), list(predicted_stem_idxs),
                          list(predicted_arg_idxs)))
        return result

    def getAllPredictionIdxs_batch(self, contexts: List[TacticContext]
                                   ) -> List[List[Tuple[float, int, int]]]:
        assert self.training_args
        assert self._model

        num_stem_poss = get_num_tokens(self._metadata)
        stem_width = min(self.training_args.max_beam_width, num_stem_poss)

        tokenized_premises_batch, premise_features_batch, \
            nhyps_batch, tokenized_goal_batch, \
            goal_mask, \
            word_features, vec_features = \
            sample_fpa_batch(extract_dataloader_args(self.training_args),
                             self._metadata,
                             [context_py2r(context)
                              for context in contexts])

        stem_certainties_batch, stem_idxs_batch = self.predict_stems(
            stem_width, word_features, vec_features)

        goal_arg_values_batch = self.goal_token_scores(
            stem_idxs_batch, tokenized_goal_batch, goal_mask)

        idxs_batch = []

        for (stem_certainties, stem_idxs,
             goal_arg_values, tokenized_goal,
             tokenized_premises,
             premise_features) in \
            zip(stem_certainties_batch, stem_idxs_batch,
                goal_arg_values_batch, tokenized_goal_batch,
                tokenized_premises_batch, premise_features_batch):
            premise_arg_values = self.hyp_name_scores(
                stem_idxs,
                tokenized_goal,
                tokenized_premises,
                premise_features)
            total_scores = torch.cat((goal_arg_values.unsqueeze(0),
                                      premise_arg_values),
                                     dim=2)

            probs, stems, args = self.predict_args(
                total_scores, stem_certainties, stem_idxs)
            idxs_batch.append(list(zip(list(probs), list(stems), list(args))))

        return idxs_batch

    def decodeNonDuplicatePredictions(
            self, context: TacticContext,
            all_idxs: List[Tuple[float, int, int]],
            k: int) -> List[Prediction]:
        assert self.training_args
        num_stem_poss = get_num_tokens(self._metadata)
        stem_width = min(self.training_args.max_beam_width, num_stem_poss)

        if self.training_args.lemma_args:
            all_hyps = context.hypotheses + context.relevant_lemmas
        else:
            all_hyps = context.hypotheses

        prediction_strs: List[str] = []
        prediction_probs: List[float] = []
        next_i = 0
        num_valid_probs = (1 + len(all_hyps) +
                           len(get_fpa_words(context.goal))) * stem_width
        while len(prediction_strs) < k and next_i < num_valid_probs:
            next_pred_str = decode_fpa_result(
                extract_dataloader_args(self.training_args),
                self._metadata,
                all_hyps, context.goal,
                all_idxs[next_i][1],
                all_idxs[next_i][2])
            # next_pred_str = ""
            if next_pred_str not in prediction_strs:
                prediction_strs.append(next_pred_str)
                prediction_probs.append(math.exp(all_idxs[next_i][0]))
            next_i += 1

        predictions = [Prediction(s, prob) for s, prob in
                       zip(prediction_strs, prediction_probs)]

        return predictions

    def predictKTactics(self, context: TacticContext, k: int
                        ) -> List[Prediction]:
        assert self.training_args
        assert self._model

        all_predictions = self.getAllPredictionIdxs(context)

        predictions = self.decodeNonDuplicatePredictions(
            context, all_predictions, k)

        return predictions

    def predict_stems(self, k: int,
                      word_features: List[List[int]],
                      vec_features: List[List[float]]
                      ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        assert self._model
        assert len(word_features) == len(vec_features)
        batch_size = len(word_features)
        stem_distribution = self._model.stem_classifier(
            LongTensor(word_features), FloatTensor(vec_features))
        stem_probs, stem_idxs = stem_distribution.topk(k)
        assert stem_probs.size() == torch.Size([batch_size, k])
        assert stem_idxs.size() == torch.Size([batch_size, k])
        return stem_probs, stem_idxs

    def goal_token_scores(self, stem_idxs: torch.LongTensor,
                          tokenized_goals: List[List[int]],
                          goal_masks: List[List[bool]],
                          ) -> torch.FloatTensor:
        assert self._model
        assert self.training_args
        batch_size = stem_idxs.size()[0]
        stem_width = stem_idxs.size()[1]
        goal_len = self.training_args.max_length
        # The goal probabilities include the "no argument" probability
        num_goal_probs = goal_len + 1
        unmasked_probabilities = self._model.goal_args_model(
            stem_idxs.view(batch_size * stem_width),
            LongTensor(tokenized_goals).view(
                batch_size, 1, goal_len)
            .expand(-1, stem_width, -1).contiguous()
            .view(batch_size * stem_width, goal_len))\
            .view(batch_size, stem_width, num_goal_probs)

        masked_probabilities = torch.where(
            ByteTensor(goal_masks)
            .view(batch_size, 1, num_goal_probs)
            .expand(-1, stem_width, -1),
            unmasked_probabilities,
            torch.full_like(unmasked_probabilities, -float("Inf")))

        assert masked_probabilities.size() == torch.Size(
            [batch_size, stem_width, num_goal_probs])
        return masked_probabilities

    def hyp_name_scores(self,
                        stem_idxs: torch.LongTensor,
                        tokenized_goal: List[int],
                        tokenized_premises: List[List[int]],
                        premise_features: List[List[float]]
                        ) -> torch.FloatTensor:
        assert self._model
        assert len(stem_idxs.size()) == 1
        stem_width = stem_idxs.size()[0]
        num_hyps = len(tokenized_premises)
        encoded_goals = self._model.goal_encoder(LongTensor([tokenized_goal]))
        hyp_arg_values = self.runHypModel(stem_idxs.unsqueeze(0),
                                          encoded_goals,
                                          LongTensor([tokenized_premises]),
                                          FloatTensor([premise_features]))
        assert hyp_arg_values.size() == torch.Size([1, stem_width, num_hyps])
        return hyp_arg_values

    def predict_args(self,
                     total_scores: torch.FloatTensor,
                     stem_certainties: torch.FloatTensor,
                     stem_idxs: torch.LongTensor
                     ) -> Tuple[torch.FloatTensor, torch.LongTensor,
                                torch.LongTensor]:
        batch_size = total_scores.size()[0]
        assert batch_size == 1
        stem_width = total_scores.size()[1]
        num_probs_per_stem = total_scores.size()[2]
        all_prob_batches = self._softmax(
            (total_scores +
             stem_certainties.view(batch_size, stem_width, 1)
             .expand(-1, -1, num_probs_per_stem))
            .contiguous()
            .view(batch_size, stem_width * num_probs_per_stem))
        prediction_probs, arg_idxs = all_prob_batches.sort(descending=True)
        assert prediction_probs.size() == torch.Size(
            [batch_size, stem_width * num_probs_per_stem])
        assert arg_idxs.size() == torch.Size(
            [batch_size, stem_width * num_probs_per_stem])
        predicted_stem_keys = arg_idxs // num_probs_per_stem
        predicted_stem_idxs = stem_idxs.view(stem_width)\
                                       .index_select(
                                           0, predicted_stem_keys.squeeze(
                                               dim=0))
        predicted_arg_idxs = arg_idxs % num_probs_per_stem
        return prediction_probs[0], predicted_stem_idxs, predicted_arg_idxs[0]

    def predictKTacticsWithLoss(self, in_data: TacticContext, k: int, correct: str) -> \
            Tuple[List[Prediction], float]:
        return self.predictKTactics(in_data, k), 0

    def predictKTacticsWithLoss_batch(self,
                                      in_datas: List[TacticContext],
                                      k: int, corrects: List[str]) -> \
            Tuple[List[List[Prediction]], float]:
        return self.predictKTactics_batch(in_datas, k), 0

    def runHypModel(self, stem_idxs: torch.LongTensor, encoded_goals: torch.FloatTensor,
                    hyps_batch: torch.LongTensor, hypfeatures_batch: torch.FloatTensor):
        assert self._model
        assert self.training_args
        batch_size = encoded_goals.size()[0]
        assert batch_size == 1
        num_hyps = hyps_batch.size()[1]
        beam_width = stem_idxs.size()[1]
        if hypfeatures_batch.size()[1] == 0:
            return maybe_cuda(torch.zeros(batch_size, beam_width, 0))
        features_size = hypfeatures_batch.size()[2]
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

    def shortname(self) -> str:
        return "polyarg"

    def add_args_to_parser(self, parser: argparse.ArgumentParser,
                           default_values: Dict[str, Any] = {}) -> None:
        new_defaults = {"batch-size": 128, "learning-rate": 0.4, "epoch-step": 3,
                        **default_values}
        super().add_args_to_parser(parser, new_defaults)
        add_nn_args(parser, new_defaults)
        add_tokenizer_args(parser, new_defaults)
        feature_set: Set[str] = set()
        all_constructors: List[Type[Feature]] = vec_feature_constructors + \
            word_feature_constructors  # type: ignore
        for feature_constructor in all_constructors:
            new_args = feature_constructor\
                .add_feature_arguments(parser, feature_set, default_values)
            feature_set = feature_set.union(new_args)
        parser.add_argument("--max-length", dest="max_length", type=int,
                            default=default_values.get("max-length", 30))
        parser.add_argument("--max-string-distance", type=int,
                            default=default_values.get("max-string-distance", 50))
        parser.add_argument("--max-beam-width", dest="max_beam_width", type=int,
                            default=default_values.get("max-beam-width", 10))
        parser.add_argument("--no-lemma-args",
                            dest="lemma_args", action='store_false')
        parser.add_argument("--no-hyp-features",
                            dest="hyp_features", action="store_false")
        parser.add_argument(
            "--no-features", dest="features", action="store_false")
        parser.add_argument("--no-hyp-rnn", dest="hyp_rnn",
                            action="store_false")
        parser.add_argument(
            "--no-goal-rnn", dest="goal_rnn", action="store_false")
        parser.add_argument("--replace-rnns-with-dnns", action="store_true")
        parser.add_argument("--print-tensors", action="store_true")
        parser.add_argument("--load-text-tokens", default=None)
        parser.add_argument("--load-tensors", default=None)

        parser.add_argument("--save-embedding", type=str, default=None)
        parser.add_argument("--save-features-state", type=str, default=None)
        parser.add_argument("--load-embedding", type=str, default=None)
        parser.add_argument("--load-features-state", type=str, default=None)

    def _encode_data(self, data: RawDataset, arg_values: Namespace) \
        -> Tuple[FeaturesPolyArgDataset, Tuple[Tokenizer, Embedding,
                                               List[WordFeature], List[VecFeature]]]:
        pass

    def _optimize_model(self, arg_values: Namespace) -> Iterable[FeaturesPolyargState]:
        with print_time("Loading data", guard=arg_values.verbose):
            if arg_values.start_from:
                _, (old_arg_values, unparsed_args,
                    metadata, state) = torch.load(arg_values.start_from)
                _, data_lists, \
                    (word_features_size, vec_features_size) = \
                    features_polyarg_tensors_with_meta(
                        extract_dataloader_args(arg_values),
                        str(arg_values.scrape_file),
                        metadata)
            else:
                metadata, data_lists, \
                    (word_features_size, vec_features_size) = \
                    features_polyarg_tensors(
                        extract_dataloader_args(arg_values),
                        str(arg_values.scrape_file))
        with print_time("Converting data to tensors", guard=arg_values.verbose):
            unpadded_tokenized_hyp_types, \
                unpadded_hyp_features, \
                num_hyps, \
                tokenized_goals, \
                goal_masks, \
                word_features, \
                vec_features, \
                tactic_stem_indices, \
                arg_indices = data_lists

            tensors = [pad_sequence([torch.LongTensor(tokenized_hyps_list)
                                     for tokenized_hyps_list
                                     in unpadded_tokenized_hyp_types],
                                    batch_first=True),
                       pad_sequence([torch.FloatTensor(hyp_features_vec)
                                     for hyp_features_vec
                                     in unpadded_hyp_features],
                                    batch_first=True),
                       torch.LongTensor(num_hyps),
                       torch.LongTensor(tokenized_goals),
                       torch.ByteTensor(goal_masks),
                       torch.LongTensor(word_features),
                       torch.FloatTensor(vec_features),
                       torch.LongTensor(tactic_stem_indices),
                       torch.LongTensor(arg_indices)]
            with open("tensors.pickle", 'wb') as f:
                torch.save(tensors, f)
            eprint(tensors, guard=arg_values.print_tensors)

        with print_time("Building the model", guard=arg_values.verbose):

            if arg_values.start_from:
                self.load_saved_state(arg_values, unparsed_args,
                                      metadata, state)
                model = self._model
                epoch_start = self.num_epochs
            else:
                model = self._get_model(arg_values,
                                        word_features_size,
                                        vec_features_size,
                                        get_num_indices(metadata),
                                        get_num_tokens(metadata))
                epoch_start = 1

        assert model
        assert epoch_start
        return ((metadata, state) for state in optimize_checkpoints(tensors, arg_values, model,
                                                                    lambda batch_tensors, model:
                                                                    self._getBatchPredictionLoss(arg_values,
                                                                                                 batch_tensors,
                                                                                                 model), epoch_start))

    def load_saved_state(self,
                         args: Namespace,
                         unparsed_args: List[str],
                         metadata: Any,
                         state: NeuralPredictorState) -> None:
        model = maybe_cuda(self._get_model(args,
                                           get_word_feature_vocab_sizes(
                                               metadata),
                                           get_vec_features_size(metadata),
                                           get_num_indices(metadata),
                                           get_num_tokens(metadata)))
        model.load_state_dict(state.weights)
        self._model = model
        self.training_loss = state.loss
        self.num_epochs = state.epoch
        self.training_args = args
        self.unparsed_args = unparsed_args
        self._metadata = metadata

    def _get_model(self, arg_values: Namespace,
                   wordf_sizes: List[int],
                   vecf_size: int,
                   stem_vocab_size: int,
                   goal_vocab_size: int) \
            -> FeaturesPolyArgModel:
        return FeaturesPolyArgModel(
            FeaturesClassifier(wordf_sizes, vecf_size,
                               arg_values.hidden_size,
                               arg_values.num_layers,
                               stem_vocab_size),
            GoalTokenArgModel(stem_vocab_size, goal_vocab_size, arg_values.max_length,
                              arg_values.hidden_size),
            EncoderRNN(goal_vocab_size, arg_values.hidden_size,
                       arg_values.hidden_size),
            HypArgModel(arg_values.hidden_size, stem_vocab_size, goal_vocab_size,
                        hypFeaturesSize(), arg_values.hidden_size))

    def _getBatchPredictionLoss(self, arg_values: Namespace,
                                batch: Sequence[torch.Tensor],
                                model: FeaturesPolyArgModel) -> torch.FloatTensor:
        tokenized_hyp_types_batch, hyp_features_batch, num_hyps_batch, \
            tokenized_goals_batch, goal_masks_batch, \
            word_features_batch, vec_features_batch, \
            stem_idxs_batch, arg_total_idxs_batch = \
            cast(Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor,
                       torch.LongTensor, torch.ByteTensor,
                       torch.LongTensor, torch.FloatTensor,
                       torch.LongTensor, torch.LongTensor],
                 data_batch)
        batch_size = tokenized_goals_batch.size()[0]
        goal_size = tokenized_goals_batch.size()[1]
        stemDistributions = model.stem_classifier(
            word_features_batch, vec_features_batch)
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
                    torch.cat((maybe_cuda(stem_idx.view(1)),
                               predictedStemIdxList[:stem_width-1])))
        mergedStemIdxsT = torch.stack(mergedStemIdxs)
        correctPredictionIdxs = torch.LongTensor([list(idxList).index(stem_idx) for
                                                  idxList, stem_idx
                                                  in zip(mergedStemIdxs, stem_var)])
        if arg_values.hyp_rnn:
            tokenized_hyps_var = maybe_cuda(
                Variable(tokenized_hyp_types_batch))
        else:
            tokenized_hyps_var = maybe_cuda(
                Variable(torch.zeros_like(tokenized_hyp_types_batch)))

        if arg_values.hyp_features:
            hyp_features_var = maybe_cuda(Variable(hyp_features_batch))
        else:
            hyp_features_var = maybe_cuda(
                Variable(torch.zeros_like(hyp_features_batch)))

        goal_arg_values = model.goal_args_model(
            mergedStemIdxsT.view(batch_size * stem_width),
            tokenized_goals_batch.view(
                batch_size, 1, goal_size).expand(-1, stem_width, -1)
            .contiguous().view(batch_size * stem_width, goal_size))\
            .view(batch_size, stem_width, goal_size + 1)
        goal_arg_values = torch.where(
            maybe_cuda(goal_masks_batch.view(
                batch_size, 1, arg_values.max_length + 1))
            .expand(-1, stem_width, -1),
            goal_arg_values,
            maybe_cuda(torch.full_like(goal_arg_values, -float("Inf"))))
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
        assert hyp_arg_values_concatted.size() == torch.Size(
            [batch_size * stem_width * hyp_lists_length, 1]), hyp_arg_values_concatted.size()
        hyp_arg_values = hyp_arg_values_concatted.view(batch_size, stem_width,
                                                       hyp_lists_length)
        total_arg_values = torch.cat((goal_arg_values, hyp_arg_values),
                                     dim=2)
        num_probs = hyp_lists_length + goal_size + 1
        total_arg_distribution = \
            self._softmax(total_arg_values.view(
                batch_size, stem_width * num_probs))
        total_arg_var = maybe_cuda(Variable(arg_total_idxs_batch +
                                            (correctPredictionIdxs * num_probs)))\
            .view(batch_size)
        loss = FloatTensor([0.])
        loss += self._criterion(stemDistributions, stem_var)
        loss += self._criterion(total_arg_distribution, total_arg_var)
        return loss


def hypFeaturesSize() -> int:
    return 2


def extract_dataloader_args(args: argparse.Namespace) -> DataloaderArgs:
    dargs = DataloaderArgs()
    # dargs.max_distance = args.max_distance
    dargs.max_length = args.max_length
    dargs.num_keywords = args.num_keywords
    dargs.max_string_distance = args.max_string_distance
    dargs.max_premises = args.max_premises
    dargs.num_relevance_samples = args.num_relevance_samples
    assert args.load_tokens, \
        "Must have a keywords file for the rust dataloader"
    dargs.keywords_file = args.load_tokens
    dargs.context_filter = args.context_filter
    dargs.save_embedding = args.save_embedding
    dargs.save_features_state = args.save_features_state
    dargs.load_embedding = args.load_embedding
    dargs.load_features_state = args.load_features_state
    return dargs


def context_py2r(py_context: TacticContext) -> dataloader.TacticContext:
    return dataloader.TacticContext(
        py_context.relevant_lemmas, py_context.prev_tactics,
        dataloader.Obligation(py_context.hypotheses, py_context.goal))


def main(arg_list: List[str]) -> None:
    predictor = FeaturesPolyargPredictor()
    predictor.train(arg_list)
