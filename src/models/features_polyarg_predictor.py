import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from features import (WordFeature, VecFeature,
                      word_feature_constructors, vec_feature_constructors)
import tokenizer
from tokenizer import Tokenizer
from data import (ListDataset, normalizeSentenceLength, RawDataset,
                  EmbeddedSample, EOS_token)
from util import *
from format import ScrapedTactic
import serapi_instance
from models.components import (WordFeaturesEncoder, Embedding,
                               DNNClassifier, EncoderDNN, EncoderRNN,
                               add_nn_args)
from models.tactic_predictor import (TrainablePredictor,
                                     NeuralPredictorState,
                                     TacticContext, Prediction,
                                     optimize_checkpoints,
                                     save_checkpoints, tokenize_goals,
                                     embed_data, add_tokenizer_args,
                                     strip_scraped_output)

import threading
import multiprocessing
import argparse
import sys
import functools
from itertools import islice
from argparse import Namespace
from typing import (List, Tuple, NamedTuple, Optional, Sequence, Dict,
                    cast, Union)

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
        self._tokenizer : Optional[Tokenizer] = None
        self._embedding : Optional[Embedding] = None
        self._model : Optional[FeaturesPolyArgModel] = None
    def predictKTactics(self, in_data : TacticContext, k : int) -> List[Prediction]:
        return [self.predictTactic(in_data)] * k
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
        assert self._word_feature_functions
        assert self._vec_feature_functions
        assert self._tokenizer
        assert self._embedding
        assert self.training_args
        assert self._model

        word_features = LongTensor([[feature(context) for feature in
                                     self._word_feature_functions]])
        vec_features = FloatTensor([[feature_val for feature in
                                    self._vec_feature_functions
                                    for feature_val in feature(context)]])
        stem_distribution = self._model.stem_classifier(word_features, vec_features)
        stem_certainties, stem_idxs = stem_distribution.topk(1)
        goals_batch = torch.LongTensor([normalizeSentenceLength(
            self._tokenizer.toTokenList(context.goal),
            self.training_args.max_length)])
        hyps_batch = torch.LongTensor([normalizeSentenceLength(
            self._tokenizer.toTokenList(hyp),
            self.training_args.max_length)
                                       for hyp in context.hypotheses])
        hypfeatures_batch = torch.FloatTensor([
            [SequenceMatcher(None, context.goal,
                             serapi_instance.get_hyp_type(hyp)).ratio(),
             len(hyp)]
            for hyp in context.hypotheses])
        goal_arg_values = self._model.goal_args_model(stem_idxs, goals_batch)
        goal_symbols = tokenizer.get_symbols(context.goal)
        # print("Goal values are {}, zeroing...".format(goal_arg_values))
        for i in range(len(goal_symbols) + 1, goal_arg_values.size()[1]):
            goal_arg_values[0, i] = -float("Inf")
        # print("Zeroed")
        encoded_goals = self._model.goal_encoder(goals_batch)
        if len(context.hypotheses) > 0:
            hyp_arg_values = \
                self._model.hyp_model(stem_idxs.view(1, 1)\
                                      .expand(-1, len(context.hypotheses))\
                                      .contiguous()\
                                      .view(len(context.hypotheses)),
                                      encoded_goals.view(1, 1,
                                                         self.training_args.hidden_size)\
                                      .expand(-1, len(context.hypotheses), -1)\
                                      .contiguous()\
                                      .view(len(context.hypotheses),
                                            self.training_args.hidden_size),
                                      hyps_batch, hypfeatures_batch)\
                                      .view(1, len(context.hypotheses))
            total_arg_values = torch.cat((goal_arg_values, hyp_arg_values),
                                         dim=1)
        else:
            total_arg_values = goal_arg_values

        total_arg_distribution = self._softmax(total_arg_values)
        total_arg_certainties, total_arg_idxs = total_arg_distribution.topk(1)
        total_arg_idx = total_arg_idxs[0]
        stem = self._embedding.decode_token(stem_idxs[0].item())
        probability = math.exp(stem_certainties[0] +
                               total_arg_certainties[0])
        if total_arg_idx == 0:
            return Prediction(stem + ".", probability)

        elif total_arg_idx < goal_arg_values.size()[1]:
            return Prediction(stem + " " +
                              tokenizer.get_symbols(context.goal)[total_arg_idx - 1] + ".",
                              probability)
        else:
            return Prediction(stem + " " + \
                              serapi_instance.get_var_term_in_hyp(
                                  context.hypotheses[total_arg_idx
                                                     - goal_arg_values.size()[1]]) + \
                              ".", probability)
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
        super().add_args_to_parser(parser, {"learning-rate": 0.4,
                                            **default_values})
        add_nn_args(parser, default_values)
        add_tokenizer_args(parser, default_values)
        parser.add_argument("--max-length", dest="max_length", type=int,
                            default=default_values.get("max-length", 30))
        parser.add_argument("--num-head-keywords", dest="num_head_keywords", type=int,
                            default=default_values.get("num-head-keywords", 100))
        parser.add_argument("--num-tactic-keywords", dest="num_tactic_keywords", type=int,
                            default=default_values.get("num-tactic-keywords", 50))
    def _preprocess_data(self, data : RawDataset, arg_values : Namespace) \
        -> Iterable[ScrapedTactic]:
        data_iter = super()._preprocess_data(data, arg_values)
        yield from map(serapi_instance.normalizeNumericArgs, data_iter)

    def _encode_data(self, data : RawDataset, arg_values : Namespace) \
        -> Tuple[FeaturesPolyArgDataset, Tuple[Tokenizer, Embedding,
                                               List[WordFeature], List[VecFeature]]]:
        preprocessed_data = list(self._preprocess_data(data, arg_values))
        stripped_data = [strip_scraped_output(dat) for dat in preprocessed_data]
        self._word_feature_functions  = [feature_constructor(stripped_data, arg_values) for # type: ignore
                                       feature_constructor in
                                        word_feature_constructors]
        self._vec_feature_functions = [feature_constructor(stripped_data, arg_values) for # type: ignore
                                       feature_constructor in vec_feature_constructors]
        embedding, embedded_data = embed_data(RawDataset(preprocessed_data))
        tokenizer, tokenized_goals = tokenize_goals(embedded_data, arg_values)
        with multiprocessing.Pool(arg_values.num_threads) as pool:
            start = time.time()
            print("Creating dataset...", end="")
            result_data = FeaturesPolyArgDataset(list(pool.imap(
                functools.partial(mkFPASample, embedding,
                                  tokenizer,
                                  arg_values.max_length,
                                  self._word_feature_functions,
                                  self._vec_feature_functions),
                zip(preprocessed_data, tokenized_goals))))
            print("{:.2f}s".format(time.time() - start))
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
                                    self._getBatchPredictionLoss(batch_tensors, model))
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
                        2, arg_values.hidden_size))
    def _getBatchPredictionLoss(self, data_batch : Sequence[torch.Tensor],
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
        stemDistributions = model.stem_classifier(word_features_batch, vec_features_batch)
        stem_var = maybe_cuda(Variable(stem_idxs_batch))
        tokenized_hyps_var = maybe_cuda(Variable(tokenized_hyp_types_batch))
        hyp_features_var = maybe_cuda(Variable(hyp_features_batch))
        goal_arg_values = model.goal_args_model(stem_idxs_batch,
                                                tokenized_goals_batch)
        encoded_goals = model.goal_encoder(tokenized_goals_batch)

        hyp_lists_length = tokenized_hyp_types_batch.size()[1]
        hyp_length = tokenized_hyp_types_batch.size()[2]
        hyp_features_size = hyp_features_batch.size()[2]
        encoded_goal_size = encoded_goals.size()[1]

        encoded_goals_expanded = \
            encoded_goals.view(batch_size, 1, encoded_goal_size)\
            .expand(-1, hyp_lists_length, -1).contiguous()\
            .view(batch_size * hyp_lists_length, encoded_goal_size)
        stems_expanded = \
            stem_var.view(batch_size, 1)\
            .expand(-1, hyp_lists_length).contiguous()\
            .view(batch_size * hyp_lists_length)
        hyp_arg_values_concatted = \
            model.hyp_model(stems_expanded,
                            encoded_goals_expanded,
                            tokenized_hyps_var
                            .view(batch_size * hyp_lists_length, hyp_length),
                            hyp_features_var
                            .view(batch_size * hyp_lists_length, hyp_features_size))
        hyp_arg_values = pad_sequence([concatted[:num_hyps]
                                       for concatted, num_hyps in
                                       zip(hyp_arg_values_concatted
                                           .view(batch_size, hyp_lists_length),
                                           num_hyps_batch)],
                                      padding_value=float('-Inf'),
                                      batch_first=True)
        total_arg_values = torch.cat((goal_arg_values, hyp_arg_values),
                                     dim=1)
        total_arg_distribution = self._softmax(total_arg_values)
        total_arg_var = maybe_cuda(Variable(arg_total_idxs_batch)).view(batch_size)
        loss = FloatTensor([0.])
        # print(stemDistributions.size())
        # print(stem_var.size())
        loss += self._criterion(stemDistributions, stem_var)
        # print("Num hyps: {}".format(num_hyps_batch))
        # print("Hyp lists length: {}".format(hyp_lists_length))
        # print("Goal max length: {}".format(goal_arg_values.size()[1]))
        # print(total_arg_distribution.size())
        # print(total_arg_var)
        loss += self._criterion(total_arg_distribution, total_arg_var)
        return loss

def mkFPASample(embedding : Embedding,
                mytokenizer : Tokenizer,
                max_length : int,
                word_feature_functions : List[WordFeature],
                vec_feature_functions : List[VecFeature],
                zipped : Tuple[ScrapedTactic, List[int]]) \
                -> FeaturesPolyArgSample:
    inter, tokenized_goal = zipped
    prev_tactics, hypotheses, goal_str, tactic = inter
    context = strip_scraped_output(inter)
    word_features = [feature(context) for feature in word_feature_functions]
    vec_features = [feature_val for feature in vec_feature_functions
                    for feature_val in feature(context)]
    tokenized_hyp_types = [normalizeSentenceLength(
        mytokenizer.toTokenList(serapi_instance.get_hyp_type(hyp)),
        max_length)
                           for hyp in hypotheses]
    hypfeatures = [[SequenceMatcher(None, goal_str,
                                    serapi_instance.get_hyp_type(hyp)).ratio(),
                    len(hyp)] for hyp in hypotheses]
    tactic_stem, tactic_argstr = serapi_instance.split_tactic(tactic)
    stem_idx = embedding.encode_token(tactic_stem)
    argstr_tokens = tactic_argstr.strip(".").split()
    assert len(argstr_tokens) < 2, \
        "Tactic {} doesn't fit our argument model! Too many tokens" .format(tactic)
    arg : TacticArg
    if len(argstr_tokens) == 0:
        arg_type = ArgType.NO_ARG
        arg = None
    else:
        goal_symbols = tokenizer.get_symbols(goal_str)
        arg_token = argstr_tokens[0]
        if arg_token in goal_symbols:
            arg_type = ArgType.GOAL_TOKEN
            arg_idx = goal_symbols.index(arg_token)
            assert arg_idx < max_length
            arg = GoalTokenArg(goal_symbols.index(arg_token))
        else:
            hyp_vars = [serapi_instance.get_var_term_in_hyp(hyp)
                        for hyp in hypotheses]
            assert arg_token in hyp_vars, "Tactic {} doesn't fit our argument model! "\
                "Token {} is not a hyp var or goal token.\n"\
                "Hyps: {}\n"\
                "Goal: {}".format(tactic, arg_token, hypotheses, goal_str)
            arg_type = ArgType.HYP_ID
            arg = HypIdArg(hyp_vars.index(arg_token))
    return FeaturesPolyArgSample(
        tokenized_hyp_types,
        hypfeatures,
        normalizeSentenceLength(tokenized_goal, max_length),
        word_features,
        vec_features,
        stem_idx,
        arg_type,
        arg)

def main(arg_list : List[str]) -> None:
    predictor = FeaturesPolyargPredictor()
    predictor.train(arg_list)
