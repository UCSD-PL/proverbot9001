
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
                               DNNClassifier, add_nn_args)
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
from argparse import Namespace
from typing import (List, Tuple, NamedTuple, Optional, Sequence, Dict,
                    cast)

class CopyArgSample(NamedTuple):
    tokenized_goal : List[int]
    word_features : List[int]
    vec_features : List[float]
    tactic_stem : int
    tactic_arg_token_index : int

class CopyArgDataset(ListDataset[CopyArgSample]):
    pass

class FindArgModel(nn.Module):
    def __init__(self, input_vocab_size : int, input_length : int,
                 hidden_size : int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self._word_embedding = maybe_cuda(nn.Embedding(input_vocab_size, hidden_size))
        self._gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self._likelyhood_layer = maybe_cuda(nn.Linear(hidden_size, 1))
        self._softmax = maybe_cuda(nn.LogSoftmax(dim=1))
    def forward(self, goal_batch : torch.LongTensor) -> torch.FloatTensor:
        goal_var = maybe_cuda(Variable(goal_batch))
        batch_size = goal_batch.size()[0]
        hidden = maybe_cuda(Variable(torch.zeros(1, batch_size, self.hidden_size)))
        copy_likelyhoods : List[torch.FloatTensor] = []
        for i in range(goal_batch.size()[1]):
            token_batch = self._word_embedding(goal_var[:,i])\
                .view(1, batch_size, self.hidden_size)
            token_batch = F.relu(token_batch)
            token_out, hidden = self._gru(token_batch, hidden)
            copy_likelyhood = self._likelyhood_layer(F.relu(token_out))
            copy_likelyhoods.append(copy_likelyhood[0])
        end_token_embedded = self._word_embedding(LongTensor([EOS_token])
                                                   .expand(batch_size))\
                                                   .view(1, batch_size, self.hidden_size)
        final_out, final_hidden = self._gru(F.relu(end_token_embedded), hidden)
        final_likelyhood = self._likelyhood_layer(F.relu(final_out))
        copy_likelyhoods.insert(0, final_likelyhood[0])
        catted = torch.cat(copy_likelyhoods, dim=1)
        result = self._softmax(catted)
        return result

class CopyArgModel(nn.Module):
    def __init__(self, find_arg_rnn : FindArgModel,
                 word_features_encoder : WordFeaturesEncoder,
                 features_classifier : nn.Linear) -> None:
        super().__init__()
        self.find_arg_rnn = maybe_cuda(find_arg_rnn)
        self.word_features_encoder = maybe_cuda(word_features_encoder)
        self.features_classifier = maybe_cuda(features_classifier)

class CopyArgPredictor(TrainablePredictor[CopyArgDataset,
                                          Tuple[Tokenizer, Embedding,
                                                List[WordFeature], List[VecFeature]],
                                          NeuralPredictorState]):
    def __init__(self) -> None:
        self._word_feature_functions: Optional[List[WordFeature]] = None
        self._vec_feature_functions: Optional[List[VecFeature]] = None
        self._criterion = maybe_cuda(nn.NLLLoss())
        self._softmax = maybe_cuda(nn.LogSoftmax(dim=1))
        self._lock = threading.Lock()
        self._tokenizer : Optional[Tokenizer] = None
        self._embedding : Optional[Embedding] = None
    def _get_word_features(self, context : TacticContext) -> List[int]:
        assert self._word_feature_functions
        assert len(self._word_feature_functions) == 3
        result = [feature(context) for feature in self._word_feature_functions]
        assert len(result) == 3
        return result
    def _get_vec_features(self, context : TacticContext) -> List[float]:
        assert self._vec_feature_functions
        return [feature_val for feature in self._vec_feature_functions
                for feature_val in feature(context)]
    def _predictStemDistributions(self, in_datas : List[TacticContext]) \
        -> torch.FloatTensor:
        word_features_batch = LongTensor([self._get_word_features(in_data)
                                           for in_data in in_datas])
        vec_features_batch = FloatTensor([self._get_vec_features(in_data)
                                          for in_data in in_datas])
        encoded_word_features = self._model.word_features_encoder(word_features_batch)
        stem_distribution = \
            self._softmax(self._model.features_classifier(torch.cat((
                encoded_word_features, vec_features_batch), dim=1)))
        return stem_distribution

    def _predictArgIdxDistributions(self, in_datas : List[TacticContext]) \
        -> torch.FloatTensor:
        assert self._tokenizer
        goals_batch = torch.LongTensor([normalizeSentenceLength(
            self._tokenizer.toTokenList(goal),
            self.training_args.max_length)
                                        for _, _, goal in in_datas])
        arg_token_idx_distributions = self._model.find_arg_rnn(goals_batch)
        return arg_token_idx_distributions

    def _getBatchPredictionLoss(self, data_batch : Sequence[torch.Tensor],
                                model : CopyArgModel) -> torch.FloatTensor:
        goals_batch, word_features_batch, vec_features_batch, \
            stems_batch, arg_idxs_batch = \
            cast(Tuple[torch.LongTensor, torch.FloatTensor,
                       torch.LongTensor, torch.LongTensor,
                       torch.LongTensor],
                 data_batch)
        batch_size = goals_batch.size()[0]
        encoded_word_features = model.word_features_encoder(
            maybe_cuda(Variable(word_features_batch)))
        catted_data = torch.cat((encoded_word_features, maybe_cuda(Variable(vec_features_batch))), dim=1)
        stemDistributions = model.features_classifier(catted_data)
        stem_var = maybe_cuda(Variable(stems_batch)).view(batch_size)
        argTokenIdxDistributions = model.find_arg_rnn(goals_batch)
        argToken_var = maybe_cuda(Variable(arg_idxs_batch)).view(batch_size)
        loss = 0.
        loss += self._criterion(stemDistributions, stem_var)
        loss += self._criterion(argTokenIdxDistributions, argToken_var)
        return loss

    def topkPredictionsFromDistributions(self,
                                         stem_distribution : torch.FloatTensor,
                                         argtoken_distribution : torch.FloatTensor,
                                         goal : str,
                                         k : int) -> List[Prediction]:
        assert self._embedding
        all_predictions : List[Prediction] = []
        s_certs, s_idxs = stem_distribution.topk(min(k, self._embedding.num_tokens()))
        t_certs, t_idxs = argtoken_distribution.topk(min(k, self.training_args.max_length))
        for s_cert, s_idx in zip(s_certs, s_idxs):
            for t_cert, t_idx in zip(t_certs, t_idxs):
                goal_symbols = tokenizer.get_symbols(goal.strip("."))
                if t_idx.item() > 0 and t_idx.item() <= len(goal_symbols):
                    arg = goal_symbols[t_idx.item() - 1]
                else:
                    arg = ""
                certainty = math.exp(s_cert + t_cert)
                all_predictions.append(Prediction(
                    self._embedding.decode_token(s_idx.item()) + " " + arg + ".",
                    certainty))
        idxs, best_predictions = list_topk(all_predictions, k, lambda p: p.certainty)
        return best_predictions

    def predictKTactics(self, in_data : TacticContext, k : int) \
        -> List[Prediction]:
        assert self._embedding
        with self._lock:
            stem_distribution = self._predictStemDistributions([in_data])[0]
            argtoken_distribution = self._predictArgIdxDistributions([in_data])[0]
        max_predictions = self._embedding.num_tokens() * self.training_args.max_length
        if k > max_predictions:
            k = max_predictions
        return self.topkPredictionsFromDistributions(stem_distribution,
                                                     argtoken_distribution,
                                                     in_data.goal,
                                                     k)
    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) -> \
        Tuple[List[Prediction], float]:
        predictions, loss = self.predictKTacticsWithLoss_batch([in_data], k, [correct])
        return predictions[0], loss
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, corrects : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        assert self._embedding
        with self._lock:
            stem_distributions = self._predictStemDistributions(in_data)
            argtoken_distribution = self._predictArgIdxDistributions(in_data)
            correct_stems, correct_rests = \
                zip(*[serapi_instance.split_tactic(correct) for correct in corrects])
            stem_correct_var = maybe_cuda(Variable(
                LongTensor([self._embedding.encode_token(correct_stem)
                            if self._embedding.has_token(correct_stem)
                            else 0
                            for correct_stem in correct_stems])))
            argtoken_correct_var = maybe_cuda(Variable(
                LongTensor([get_arg_idx(self.training_args.max_length,
                                        serapi_instance.normalizeInductionArgs(
                                            ScrapedTactic(data.prev_tactics,
                                                          data.hypotheses,
                                                          data.goal, correct_tactic)))
                            for data, correct_tactic in zip(in_data, corrects)])))
            loss = self._criterion(stem_distributions, stem_correct_var) + \
                self._criterion(argtoken_distribution, argtoken_correct_var)
        predictions = [self.topkPredictionsFromDistributions(stem_distributions[idx],
                                                             argtoken_distribution[idx],
                                                             datum.goal,
                                                             k)
                       for idx, datum in enumerate(in_data)]
        return predictions, loss

    def getOptions(self) -> List[Tuple[str, str]]:
        return list(vars(self.training_args).items()) + \
            [("training loss", self.training_loss),
             ("# epochs", self.num_epochs),
             ("predictor", "copyarg")]

    def _description(self) -> str:
        return "A predictor features to predict a stem, and an rnn to predict a " \
            "token from the goal to use as an arugment"
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) -> None:
        super().add_args_to_parser(parser, {"learning-rate": 0.4,
                                            **default_values})
        add_nn_args(parser, default_values)
        add_tokenizer_args(parser, default_values)
        parser.add_argument("--max-length", dest="max_length", type=int,
                            default=default_values.get("max-length", 20))
        parser.add_argument("--num-head-keywords", dest="num_head_keywords", type=int,
                            default=default_values.get("num-head-keywords", 100))
        parser.add_argument("--num-tactic-keywords", dest="num_tactic_keywords", type=int,
                            default=default_values.get("num-tactic-keywords", 50))
    def _preprocess_data(self, data : RawDataset, arg_values : Namespace) \
        -> Iterable[ScrapedTactic]:
        data_iter = super()._preprocess_data(data, arg_values)
        yield from map(serapi_instance.normalizeInductionArgs, data_iter)

    def _encode_data(self, data : RawDataset, arg_values : Namespace) \
        -> Tuple[CopyArgDataset, Tuple[Tokenizer, Embedding,
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
            arg_idxs = pool.imap(functools.partial(get_arg_idx, arg_values.max_length),
                                 preprocessed_data)

            start = time.time()
            print("Creating dataset...", end="")
            sys.stdout.flush()
            result_data = CopyArgDataset(list(pool.imap(
                functools.partial(mkCopySample, arg_values.max_length,
                                  self._word_feature_functions,
                                  self._vec_feature_functions),
                zip(embedded_data, tokenized_goals, arg_idxs))))
            print("{:.2f}s".format(time.time() - start))
        return result_data, (tokenizer, embedding,
                             self._word_feature_functions,
                             self._vec_feature_functions)
    def _optimize_model_to_disc(self,
                                encoded_data : CopyArgDataset,
                                metadata : Tuple[Tokenizer, Embedding,
                                                 List[WordFeature], List[VecFeature]],
                                arg_values : Namespace) \
        -> None:
        tokenizer, embedding, word_features, vec_features = metadata
        save_checkpoints("copyarg",
                         metadata, arg_values,
                         self._optimize_checkpoints(encoded_data, arg_values,
                                                    tokenizer, embedding))
    def _optimize_checkpoints(self, encoded_data : CopyArgDataset,
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
        self._model = maybe_cuda(self._get_model(args,
                                                 self._embedding.num_tokens(),
                                                 self._tokenizer.numTokens()))
        self._model.load_state_dict(state.weights)
        self.training_loss = state.loss
        self.num_epochs = state.epoch
        self.training_args = args
    def _data_tensors(self, encoded_data : CopyArgDataset,
                      arg_values : Namespace) \
        -> List[torch.Tensor]:
        goals, word_features, vec_features, tactic_stems, tactic_arg_idxs \
            = zip(*encoded_data)
        return [torch.LongTensor(goals),
                torch.LongTensor(word_features),
                torch.FloatTensor(vec_features),
                torch.LongTensor(tactic_stems),
                torch.LongTensor(tactic_arg_idxs)]
    def _get_model(self, arg_values : Namespace,
                   tactic_vocab_size : int,
                   goal_vocab_size : int) \
        -> CopyArgModel:
        assert self._word_feature_functions
        assert self._vec_feature_functions
        feature_vec_size = sum([feature.feature_size()
                                for feature in self._vec_feature_functions])
        word_feature_vocab_sizes = [feature.vocab_size()
                                    for feature in self._word_feature_functions]
        return CopyArgModel(FindArgModel(goal_vocab_size, arg_values.max_length,
                                         arg_values.hidden_size),
                            WordFeaturesEncoder(word_feature_vocab_sizes,
                                                arg_values.hidden_size, 1,
                                                arg_values.hidden_size),
                            DNNClassifier(arg_values.hidden_size + feature_vec_size,
                                          arg_values.hidden_size, tactic_vocab_size,
                                          3))

def mkCopySample(max_length : int,
                 word_feature_functions : List[WordFeature],
                 vec_feature_functions : List[VecFeature],
                 zipped : Tuple[EmbeddedSample, List[int], int]) \
                 -> CopyArgSample:
    context, goal, arg_idx = zipped
    (prev_tactic_list, hypotheses, goal_str, tactic_idx) = context
    tac_context = TacticContext(prev_tactic_list, hypotheses, goal_str)
    word_features = [feature(tac_context)
                     for feature in word_feature_functions]
    assert len(word_features) == 3
    return CopyArgSample(normalizeSentenceLength(goal, max_length),
                         word_features,
                         [feature_val for feature in vec_feature_functions
                          for feature_val in feature(tac_context)],
                         tactic_idx, arg_idx)

def get_arg_idx(max_length : int, inter : ScrapedTactic) -> int:
    tactic_stem, tactic_rest = serapi_instance.split_tactic(inter.tactic)
    symbols = tokenizer.get_symbols(inter.goal)
    arg = tactic_rest.split()[0].strip(".")
    assert arg in symbols, "tactic: {}, arg: {}, goal: {}, symbols: {}"\
        .format(inter.tactic, arg, inter.goal, symbols)
    idx = symbols.index(arg)
    if idx >= max_length:
        return 0
    else:
        return idx + 1

def main(arg_list : List[str]) -> None:
    predictor = CopyArgPredictor()
    predictor.train(arg_list)
