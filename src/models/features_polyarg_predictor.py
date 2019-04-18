import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
                 hyp_length : int,
                 num_hyps : int,
                 hidden_size : int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self._stem_embedding = maybe_cuda(nn.Embedding(stem_vocab_size, hidden_size))
        self._token_embedding = maybe_cuda(nn.Embedding(token_vocab_size, hidden_size))
        self._in_hidden = maybe_cuda(EncoderDNN(hidden_size + goal_data_size, hidden_size, hidden_size, 1))
        self._hyp_gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self._likelyhood_decoder = maybe_cuda(EncoderDNN(hidden_size, hidden_size, 1, 2))
    def forward(self, stems_batch : torch.LongTensor,
                goals_encoded_batch : torch.FloatTensor, hyps_batch : torch.LongTensor,
                hypfeatures_batch : torch.FloatTensor):
        stems_var = maybe_cuda(Variable(stems_batch))
        hyps_var = maybe_cuda(Variable(hyps_batch))
        hypfeatures_var = maybe_cuda(Variable(hypfeatures_batch))
        batch_size = stems_batch.size()[0]
        stem_encoded = self._stem_embedding(stems_var)\
                           .view(1, batch_size, self.hidden_size)
        initial_hidden = self._in_hidden(torch.cat((stem_encoded, goals_encoded_batch), dim=1))
        hidden = initial_hidden
        for i in range(hyps_batch.size()[1]):
            token_batch = self._token_embedding(hyps_batch[:,i])\
                .view(1, batch_size, self.hidden_size)
            token_batch = F.relu(token_batch)
            token_out, hidden = self._hyp_gru(token_batch, hidden)
        hyp_likelyhoods = self._likelyhood_decoder(
            torch.cat((token_out, hypfeatures_batch), dim=1))
        return hyp_likelyhoods

class FeaturesClassifier(nn.Module):
    def __init__(self,
                 word_features : List[WordFeature],
                 vec_features : List[VecFeature],
                 hidden_size : int,
                 num_layers : int,
                 stem_vocab_size : int)\
        -> None:
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
            word_features_batch)
        stem_distribution = self._softmax(self._features_classifier(
            torch.cat((encoded_word_features, vec_features_batch), dim=1)))
        return stem_distribution

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
        self._stem_classifier : Optional[FeaturesClassifier] = None
        self._goal_args_model : Optional[GoalTokenArgModel] = None
        self._goal_encoder : Optional[EncoderRNN] = None
        self._hyp_model : Optional[HypArgModel] = None

    def predictTactic(self,
                      goal_model : GoalTokenArgModel, hyp_model : HypArgModel,
                      stem_model : FeaturesClassifier,
                      inter : ScrapedTactic):
        assert self._word_feature_functions
        assert self._vec_feature_functions
        assert self._tokenizer
        assert self._embedding
        assert self.training_args
        assert self._stem_classifier
        assert self._goal_args_model
        assert self._goal_encoder
        assert self._hyp_model

        context = strip_scraped_output(inter)
        word_features = LongTensor([[feature(context) for feature in
                                     self._word_feature_functions]])
        vec_features = LongTensor([[feature_val for feature in
                                    self._vec_feature_functions
                                    for feature_val in feature(context)]])
        stem_distribution = self._stem_classifier(word_features, vec_features)
        stem_certaintys, stem_idxs = stem_distribution.topk(1)
        goals_batch = torch.LongTensor([normalizeSentenceLength(
            self._tokenizer.toTokenList(inter.goal),
            self.training_args.max_length)])
        hyps_batch = torch.LongTensor([normalizeSentenceLength(
            self._tokenizer.toTokenList(hyp),
            self.training_args.max_length)
                                       for hyp in inter.hypotheses])
        hypfeatures_batch = torch.FloatTensor([
            [SequenceMatcher(None, inter.goal, serapi_instance.get_hyp_type(hyp)).ratio(),
             len(hyp)]
            for hyp in inter.hypotheses])
        goal_arg_values = self._goal_args_model(stem_idxs, goals_batch)
        encoded_goals = self._goal_encoder(goals_batch)
        hyp_arg_values = \
            self._hyp_model(stem_idxs,
                            encoded_goals.view(1, 1,
                                               self.training_args.max_length)\
                            .expand(-1, len(inter.hypotheses), -1).contiguous()\
                            .view(len(inter.hypotheses),
                                  self.training_args.max_length),
                            hyps_batch, hypfeatures_batch)
        total_arg_values = torch.cat((goal_arg_values, hyp_arg_values),
                                     dim=1)
        total_arg_distribution = self._softmax(total_arg_values)
        total_arg_certainties, total_arg_idxs = total_arg_distribution.topk(1)
        total_arg_idx = total_arg_idxs[0]
        stem = self._embedding.decode_token(stem_idxs[0])
        if total_arg_idx == 0:
            return stem + "."
        elif total_arg_idx < goal_arg_values.size()[1]:
            return stem + " " + tokenizer.get_symbols(inter.goal)[total_arg_idx - 1] + "."
        else:
            return stem + " " + \
                serapi_instance.get_var_term_in_hyp(
                    inter.hypotheses[total_arg_idx - goal_arg_values.size()[1]]) + \
                    "."
    def getOptions(self) -> List[Tuple[str, str]]:
        assert self.training_args
        assert self.training_loss
        assert self.num_epochs
        return list(vars(self.training_args).items()) + \
            [("training loss", self.training_loss),
             ("# epochs", self.num_epochs),
             ("predictor", "features_polyarg")]
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
                                  arg_values.max_length,
                                  self._word_feature_functions,
                                  self._vec_feature_functions),
                zip(preprocessed_data, tokenized_goals))))
            print("{:.2f}s".format(time.time() - start))
        return result_data, (tokenizer, embedding, self._word_feature_functions,
                             self._vec_feature_functions)

def mkFPASample(max_length : int,
                embedding : Embedding,
                mytokenizer : Tokenizer,
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
    tokenized_hyp_types = [mytokenizer.toTokenList(serapi_instance.get_hyp_type(hyp))
                           for hyp in hypotheses]
    hypfeatures = [[SequenceMatcher(None, goal_str,
                                    serapi_instance.get_hyp_type(hyp)).ratio(),
                    len(hyp)] for hyp in hypotheses]
    tactic_stem, tactic_argstr = serapi_instance.split_tactic(tactic)
    stem_idx = embedding.encode_token(tactic_stem)
    argstr_tokens = tactic_argstr.strip().split()
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
            arg = GoalTokenArg(goal_symbols.index(arg_token))
        else:
            hyp_vars = [serapi_instance.get_var_term_in_hyp(hyp)
                        for hyp in hypotheses]
            assert arg_token in hyp_vars, "Tactic {} doesn't fit our argument model! "\
                "Token {} is not a hyp far or goal token.".format(tactic, arg_token)
            arg_type = ArgType.HYP_ID
            arg = HypIdArg(hyp_vars.index(arg_token))
    return FeaturesPolyArgSample(
        tokenized_hyp_types,
        hypfeatures,
        tokenized_goal,
        word_features,
        vec_features,
        stem_idx,
        arg_type,
        arg)
