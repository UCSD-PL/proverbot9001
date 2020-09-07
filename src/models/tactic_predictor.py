#!/usr/bin/env python3

from typing import (Dict, List, Union, Tuple, Iterable, NamedTuple,
                    Sequence, Any, Optional, cast, BinaryIO)
from format import ScrapedTactic, TacticContext
from abc import ABCMeta, abstractmethod
import argparse
from data import (Dataset, RawDataset, ScrapedTactic, get_text_data,
                  TokenizedDataset, DatasetMetadata, stemmify_data,
                  tactic_substitutions, EmbeddedSample,
                  EmbeddedDataset, StrictEmbeddedDataset,
                  LazyEmbeddedDataset, DatasetMetadata, tokenize_data,
                  TOKEN_START, Sentence)

class Prediction(NamedTuple):
    prediction : str
    certainty : float

class TacticPredictor(metaclass=ABCMeta):
    training_args : Optional[argparse.Namespace]
    unparsed_args : List[str]
    def __init__(self) -> None:
        pass

    @abstractmethod
    def getOptions(self) -> List[Tuple[str, str]]: pass

    @abstractmethod
    def predictKTactics(self, in_data : TacticContext, k : int) \
        -> List[Prediction]: pass
    @abstractmethod
    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) -> \
        Tuple[List[Prediction], float]: pass
    @abstractmethod
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, correct : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]: pass

from typing import TypeVar, Generic, Sized
import argparse
import sys
from argparse import Namespace
from serapi_instance import get_stem
from pathlib_revised import Path2
from models.components import NeuralPredictorState, PredictorState

DatasetType = TypeVar('DatasetType')
RestrictedDatasetType = TypeVar('RestrictedDatasetType', bound=Sized)
MetadataType = TypeVar('MetadataType')
StateType = TypeVar('StateType', bound=PredictorState)

class TrainablePredictor(TacticPredictor, Generic[DatasetType, MetadataType, StateType],
                         metaclass=ABCMeta):
    def train(self, args : List[str]) -> None:
        argparser = argparse.ArgumentParser(self._description())
        self.add_args_to_parser(argparser)
        arg_values = argparser.parse_args(args)
        text_data = get_text_data(arg_values)
        encoded_data, encdec_state = self._encode_data(text_data, arg_values)
        del text_data
        gc.collect()
        self._optimize_model_to_disc(encoded_data, encdec_state, arg_values)

    def add_args_to_parser(self,
                           parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) \
        -> None:
        parser.add_argument("scrape_file", type=Path2)
        parser.add_argument("save_file", type=Path2)
        parser.add_argument("--save-all-epochs", action='store_true', dest='save_all_epochs')
        parser.add_argument("--num-threads", "-j", dest="num_threads", type=int,
                            default=default_values.get("num-threads", None))
        parser.add_argument("--max-tuples", dest="max_tuples", type=int,
                            default=default_values.get("max-tuples", None))
        parser.add_argument("--context-filter", dest="context_filter", type=str,
                            default=default_values.get("context-filter", "goal-changes"))
        parser.add_argument("--no-truncate-semicolons",
                            dest="truncate_semicolons",
                            action='store_false')
        parser.add_argument("--no-use-substitutions", dest="use_substitutions",
                            action='store_false')
        parser.add_argument("--no-normalize-numeric-args",
                            dest="normalize_numeric_args",
                            action='store_false')
        parser.add_argument("--verbose", "-v", help="verbose output",
                            action='store_const', const=True, default=False)
        pass

    @abstractmethod
    def _encode_data(self, data : RawDataset, arg_values : Namespace) \
        -> Tuple[DatasetType, MetadataType]: pass

    def _optimize_model_to_disc(self,
                                encoded_data : DatasetType,
                                encdec_state : MetadataType,
                                arg_values : Namespace) \
        -> None: pass
    @abstractmethod
    def _description(self) -> str: pass
    @abstractmethod
    def load_saved_state(self,
                         args : Namespace,
                         unparsed_args : List[str],
                         metadata : MetadataType,
                         state : StateType) -> None: pass
    pass

from tokenizer import (make_keyword_tokenizer_relevance, tokenizers,
                       Tokenizer, get_words)
from models.components import SimpleEmbedding, Embedding

import pickle
import time
import sys
import random
import multiprocessing
import itertools
import functools
import gc
from dataclasses import dataclass, astuple

@dataclass(init=True)
class TokenizerEmbeddingState(DatasetMetadata, metaclass=ABCMeta):
    tokenizer : Tokenizer
    embedding : Embedding

class TokenizingPredictor(TrainablePredictor[DatasetType, TokenizerEmbeddingState, StateType],
                          metaclass=ABCMeta):
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) \
        -> None:
        super().add_args_to_parser(parser)
        parser.add_argument("--num-keywords", dest="num_keywords", type=int,
                            default=default_values.get("num-keywordes", 60))
        parser.add_argument("--tokenizer", choices=list(tokenizers.keys()), type=str,
                            default=default_values.get("tokenizer",
                                                       list(tokenizers.keys())[0]))
        parser.add_argument("--num-relevance-samples",
                            dest="num_relevance_samples",
                            type=int,
                            default=default_values.get("num_relevance_samples",
                                                       1000))
        parser.add_argument("--save-tokens", dest="save_tokens",
                            default=default_values.get("save-tokens", None),
                            type=Path2)
        parser.add_argument("--load-tokens", dest="load_tokens",
                            default=default_values.get("load-tokens", None),
                            type=Path2)
        parser.add_argument("--print-keywords", dest="print_keywords",
                            default=False, action='store_const', const=True)
    @abstractmethod
    def _encode_tokenized_data(self, data : TokenizedDataset, arg_values : Namespace,
                               tokenizer : Tokenizer, embedding : Embedding) \
        -> DatasetType:
        pass

    def _encode_data(self, data: RawDataset, args: Namespace) \
            -> Tuple[DatasetType, TokenizerEmbeddingState]:
        embedding = SimpleEmbedding()
        embedded_data: EmbeddedDataset
        with multiprocessing.Pool(args.num_threads) as pool:
            stemmed_data = pool.imap(
                stemmify_data, data, chunksize=10240)
            lazy_embedded_data = LazyEmbeddedDataset((
                EmbeddedSample(relevant_lemmas, prev_tactics,
                               context.focused_hyps,
                               context.focused_goal,
                               embedding.encode_token(tactic))
                for (relevant_lemmas, prev_tactics, context, tactic)
                in stemmed_data))
            if args.load_tokens:
                print("Loading tokens from {}".format(args.load_tokens))
                with open(args.load_tokens, 'rb') as f:
                    tokenizer = pickle.load(f)
                    assert isinstance(tokenizer, Tokenizer)
                embedded_data = lazy_embedded_data
            else:
                # Force the embedded data for picking keywords
                forced_embedded_data = StrictEmbeddedDataset(list(lazy_embedded_data.data))
                subset = StrictEmbeddedDataset(
                    random.sample(forced_embedded_data, args.num_relevance_samples))
                embedded_data = forced_embedded_data
                start = time.time()
                print("Picking tokens...", end="")
                sys.stdout.flush()
                tokenizer = make_keyword_tokenizer_relevance(
                    [(goal, next_tactic) for
                     prev_tactics, hypotheses, goal, next_tactic in subset],
                    tokenizers[args.tokenizer], args.num_keywords, TOKEN_START, args.num_threads)
                del subset
                print("{}s".format(time.time() - start))
            if args.save_tokens:
                print("Saving tokens to {}".format(args.save_tokens))
                assert isinstance(tokenizer, Tokenizer)
                with open(args.save_tokens, 'wb') as f:
                    pickle.dump(tokenizer, f)
            if args.print_keywords:
                print("Keywords are {}".format(tokenizer.listTokens()))


            print("Tokenizing...")
            tokenized_data = tokenize_data(tokenizer, embedded_data, args.num_threads)
            gc.collect()

        return self._encode_tokenized_data(tokenized_data, args, tokenizer, embedding), \
            TokenizerEmbeddingState(tokenizer, embedding)
    def load_saved_state(self,
                         args : Namespace,
                         unparsed_args : List[str],
                         metadata : TokenizerEmbeddingState,
                         state : StateType) -> None:
        self._tokenizer = metadata.tokenizer
        self._embedding = metadata.embedding

import torch
import torch.utils.data as data
import torch.optim.lr_scheduler as scheduler
from torch import optim
import torch.nn as nn
from util import *
from util import chunks, maybe_cuda

optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
}

ModelType = TypeVar('ModelType', bound=nn.Module)

class NeuralPredictor(Generic[RestrictedDatasetType, ModelType],
                      TokenizingPredictor[RestrictedDatasetType, NeuralPredictorState],
                      metaclass=ABCMeta):
    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) \
        -> None:
        super().add_args_to_parser(parser, default_values)
        parser.add_argument("--num-epochs", dest="num_epochs", type=int,
                            default=default_values.get("num-epochs", 20))
        parser.add_argument("--batch-size", dest="batch_size", type=int,
                            default=default_values.get("batch-size", 256))
        parser.add_argument("--start-from", dest="start_from", type=str,
                            default=default_values.get("start-from", None))
        parser.add_argument("--print-every", dest="print_every", type=int,
                            default=default_values.get("print-every", 5))
        parser.add_argument("--learning-rate", dest="learning_rate", type=float,
                            default=default_values.get("learning-rate", .7))
        parser.add_argument("--max-premises", dest="max_premises", type=int,
                            default=default_values.get("max-premises", 20))
        parser.add_argument("--epoch-step", dest="epoch_step", type=int,
                            default=default_values.get("epoch-step", 10))
        parser.add_argument("--gamma", dest="gamma", type=float,
                            default=default_values.get("gamma", 0.8))
        parser.add_argument("--optimizer",
                            choices=list(optimizers.keys()), type=str,
                            default=default_values.get("optimizer",
                                                       list(optimizers.keys())[0]))

    def _optimize_model_to_disc(self,
                                encoded_data : RestrictedDatasetType,
                                encdec_state : TokenizerEmbeddingState,
                                arg_values : Namespace) \
        -> None:
        for epoch, predictor_state in enumerate(
                self._optimize_checkpoints(encoded_data, arg_values,
                                           encdec_state.embedding.num_tokens(),
                                           encdec_state.tokenizer.numTokens()),
                start=1):
            with open(arg_values.save_file, 'wb') as f:
                print("=> Saving checkpoint at epoch {}".format(epoch))
                torch.save((arg_values, encdec_state, predictor_state), f)
    def _optimize_checkpoints(self, encoded_data : RestrictedDatasetType,
                              arg_values : Namespace,
                              tactic_vocab_size : int, term_vocab_size : int) \
        -> Iterable[NeuralPredictorState]:
        dataloader = data.DataLoader(data.TensorDataset(
            *(self._data_tensors(encoded_data, arg_values))),
                                     batch_size=arg_values.batch_size, num_workers=0,
                                     shuffle=True, pin_memory=True, drop_last=True)
        # Drop the last batch in the count
        num_batches = int(len(encoded_data) / arg_values.batch_size)
        dataset_size = num_batches * arg_values.batch_size

        print("Initializing model...")
        if arg_values.start_from:
            print("Starting from file")
            with open(arg_values.start_from, 'rb') as f:
                state = torch.load(f)
                self.load_saved_state(*state) # type: ignore
            model = self._model
            epoch_start = state[2].epoch
        else:
            epoch_start = 1
            model = maybe_cuda(self._get_model(arg_values, tactic_vocab_size,
                                               term_vocab_size))
        optimizer = optimizers[arg_values.optimizer](model.parameters(),
                                                     lr=arg_values.learning_rate)
        adjuster = scheduler.StepLR(optimizer, arg_values.epoch_step,
                                    gamma=arg_values.gamma)

        training_start=time.time()

        print("Training...")
        for epoch in range(1, epoch_start):
            adjuster.step()
        for epoch in range(epoch_start, arg_values.num_epochs + 1):
            print("Epoch {} (learning rate {:.6f})".format(epoch, optimizer.param_groups[0]['lr']))

            epoch_loss = 0.

            for batch_num, data_batch in enumerate(dataloader, start=1):
                optimizer.zero_grad()
                loss = self._getBatchPredictionLoss(data_batch, model)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if batch_num % arg_values.print_every == 0:
                    items_processed = batch_num * arg_values.batch_size + \
                        (epoch - 1) * len(encoded_data)
                    progress = items_processed / (len(encoded_data) * arg_values.num_epochs)
                    print("{} ({:7} {:5.2f}%) {:.4f}"
                          .format(timeSince(training_start, progress),
                                  items_processed, progress * 100,
                                  epoch_loss / batch_num))
            adjuster.step()
            yield NeuralPredictorState(epoch,
                                       epoch_loss / num_batches,
                                       model.state_dict())
    def load_saved_state(self,
                         args : Namespace,
                         unparsed_args : List[str],
                         metadata : TokenizerEmbeddingState,
                         state : NeuralPredictorState) -> None:
        self._tokenizer = metadata.tokenizer
        self._embedding = metadata.embedding
        self._model = maybe_cuda(self._get_model(args, self._embedding.num_tokens(),
                                                 self._tokenizer.numTokens()))
        self._model.load_state_dict(state.weights)
        self.training_loss = state.loss
        self.num_epochs = state.epoch
        self.training_args = args
        pass
    def getOptions(self) -> List[Tuple[str, str]]:
        return list(vars(self.training_args).items()) + \
            [("training loss", self.training_loss),
             ("# epochs", self.num_epochs)]
    @abstractmethod
    def _encode_tokenized_data(self, data : TokenizedDataset, arg_values : Namespace,
                               tokenizer : Tokenizer, embedding : Embedding) \
        -> RestrictedDatasetType:
        pass
    @abstractmethod
    def _data_tensors(self, encoded_data : RestrictedDatasetType,
                      arg_values : Namespace) \
        -> List[torch.Tensor]:
        pass
    @abstractmethod
    def _get_model(self, arg_values : Namespace,
                   tactic_vocab_size : int, term_vocab_size : int) \
        -> ModelType:
        pass
    @abstractmethod
    def _getBatchPredictionLoss(self, data_batch : Sequence[torch.Tensor],
                                model : ModelType) \
        -> torch.FloatTensor:
        pass
    @abstractmethod
    def _description(self) -> str: pass

import threading
from torch.autograd import Variable

class NeuralClassifier(NeuralPredictor[RestrictedDatasetType, ModelType],
                       metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        self._criterion = maybe_cuda(nn.NLLLoss())
        self._lock = threading.Lock()

    @abstractmethod
    def _predictDistributions(self, in_datas : List[TacticContext]) \
        -> torch.FloatTensor:
        pass

    def predictKTactics(self, in_data : TacticContext, k : int) \
        -> List[Prediction]:
        with self._lock:
            prediction_distribution = self._predictDistributions([in_data])[0]
            if k > self._embedding.num_tokens():
                k = self._embedding.num_tokens()
            certainties_and_idxs = prediction_distribution.view(-1).topk(k)
            results = [Prediction(self._embedding.decode_token(stem_idx.data[0]) + ".",
                                  math.exp(certainty.data[0]))
                       for certainty, stem_idx in zip(*certainties_and_idxs)]
        return results

    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int,
                                correct : str) -> Tuple[List[Prediction], float]:
        with self._lock:
            prediction_distribution = self._predictDistributions([in_data])[0]
            correct_stem = get_stem(correct)
            if self._embedding.has_token(correct_stem):
                output_var = maybe_cuda(Variable(
                    torch.LongTensor([self._embedding.encode_token(correct_stem)])))
                loss = self._criterion(prediction_distribution.view(1, -1), output_var).item()
            else:
                loss = 0

            if k > self._embedding.num_tokens():
                k = self._embedding.num_tokens()
            certainties_and_idxs = prediction_distribution.view(-1).topk(k)
            results = [Prediction(self._embedding.decode_token(stem_idx.item()) + ".",
                                  math.exp(certainty.item()))
                       for certainty, stem_idx in zip(*certainties_and_idxs)]

        return results, loss

    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, correct_stems : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        if len(in_data) == 0:
            return [], 0
        with self._lock:
            prediction_distributions = self._predictDistributions(in_data)
            output_var = maybe_cuda(Variable(
                torch.LongTensor([self._embedding.encode_token(correct_stem)
                                  if self._embedding.has_token(correct_stem)
                                  else 0
                                  for correct_stem in correct_stems])))
            loss = self._criterion(prediction_distributions, output_var).item()
            if k > self._embedding.num_tokens():
                k = self._embedding.num_tokens()
            certainties_and_idxs_list = [single_distribution.view(-1).topk(k)
                                         for single_distribution in
                                         list(prediction_distributions)]
            results = [[Prediction(self._embedding.decode_token(stem_idx.item()) + ".",
                                   math.exp(certainty.item()))
                        for certainty, stem_idx in zip(*certainties_and_idxs)]
                       for certainties_and_idxs in certainties_and_idxs_list]
            return results, loss
        pass

def save_checkpoints(predictor_name : str,
                     metadata : MetadataType, arg_values : Namespace,
                     checkpoints_stream : Iterable[StateType]):
    for predictor_state in checkpoints_stream:
        epoch = predictor_state.epoch
        if arg_values.save_all_epochs:
            epoch_filename = Path2(str(arg_values.save_file.with_suffix("")) + f"-{epoch}.dat")
        else:
            epoch_filename = arg_values.save_file
        with cast(BinaryIO, epoch_filename.open(mode='wb')) as f:
            print("=> Saving checkpoint at epoch {}".format(epoch))
            torch.save((predictor_name, (arg_values, sys.argv, metadata, predictor_state)), f)

def optimize_checkpoints(data_tensors : List[torch.Tensor],
                         arg_values : Namespace,
                         model : ModelType,
                         batchLoss :
                         Callable[[Sequence[torch.Tensor], ModelType],
                                  torch.FloatTensor],
                         epoch_start : int = 1) \
    -> Iterable[NeuralPredictorState]:
    dataloader = data.DataLoader(data.TensorDataset(*data_tensors),
                                 batch_size=arg_values.batch_size, num_workers=0,
                                 shuffle=True, pin_memory=True, drop_last=True)
    # Drop the last batch in the count
    dataset_size = data_tensors[0].size()[0]
    num_batches = int(dataset_size / arg_values.batch_size)
    dataset_size = num_batches * arg_values.batch_size
    assert dataset_size > 0
    print("Initializing model...")
    model = maybe_cuda(model)
    optimizer = optimizers[arg_values.optimizer](model.parameters(),
                                                 lr=arg_values.learning_rate)
    adjuster = scheduler.StepLR(optimizer, arg_values.epoch_step,
                                gamma=arg_values.gamma)
    training_start = time.time()
    print("Training...")
    for epoch in range(1, epoch_start):
        adjuster.step()
    for epoch in range(epoch_start, arg_values.num_epochs + 1):
        print("Epoch {} (learning rate {:.6f})"
              .format(epoch, optimizer.param_groups[0]['lr']))
        epoch_loss = 0.
        for batch_num, data_batch in enumerate(dataloader, start=1):
            optimizer.zero_grad()
            with autograd.detect_anomaly():
                loss = batchLoss(data_batch, model)
                loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if batch_num % arg_values.print_every == 0:
                items_processed = batch_num * arg_values.batch_size + \
                    (epoch - epoch_start) * dataset_size
                assert items_processed > 0
                progress = items_processed / \
                    (dataset_size * (arg_values.num_epochs - (epoch_start - 1)))
                assert progress > 0
                print("{} ({:7} {:5.2f}%) {:.4f}"
                      .format(timeSince(training_start, progress),
                              items_processed, progress * 100,
                              epoch_loss / batch_num))
        adjuster.step()

        yield NeuralPredictorState(epoch,
                                   epoch_loss / num_batches,
        model.state_dict())

def embed_data(data : RawDataset, embedding : Optional[Embedding] = None) \
    -> Tuple[Embedding, StrictEmbeddedDataset]:
    if not embedding:
        embedding = SimpleEmbedding()
    start = time.time()
    print("Embedding data...", end="")
    sys.stdout.flush()
    dataset = StrictEmbeddedDataset([EmbeddedSample(
        relevant_lemmas, prev_tactics, hypotheses, goal,
        embedding.encode_token(get_stem(tactic)))
                                     for relevant_lemmas, prev_tactics,
                                     hypotheses, goal, tactic
                                     in data])
    print("{:.2f}s".format(time.time() - start))
    return embedding, dataset

def tokenize_goals(data : StrictEmbeddedDataset, args : Namespace,
                   tokenizer:Optional[Tokenizer]=None) \
    -> Tuple[Tokenizer, List[Sentence]]:
    if not tokenizer:
        if args.load_tokens and Path2(args.load_tokens).exists():
            print("Loading tokens from {}".format(args.load_tokens))
            with open(args.load_tokens, 'rb') as f:
                tokenizer = pickle.load(f)
                assert isinstance(tokenizer, Tokenizer)
        else:
            start = time.time()
            print("Picking tokens...", end="")
            sys.stdout.flush()
            subset : Sequence[EmbeddedSample]
            if args.num_relevance_samples > len(data):
                subset = data
            else:
                subset = random.sample(data, args.num_relevance_samples)
            tokenizer = make_keyword_tokenizer_relevance(
                [(goal, next_tactic) for
                 relevant_lemmas, prev_tactics, hypotheses, goal, next_tactic in subset],
                tokenizers[args.tokenizer], args.num_keywords, TOKEN_START, args.num_threads)
            print("{}s".format(time.time() - start))
    if args.save_tokens:
        print("Saving tokens to {}".format(args.save_tokens))
        with open(args.save_tokens, 'wb') as f:
            pickle.dump(tokenizer, f)
    if args.print_keywords:
        print("Keywords are {}".format(tokenizer.listTokens()))
    start = time.time()
    print("Tokenizing...", end="")
    sys.stdout.flush()
    tokenized_data = tokenize_data(tokenizer, data, args.num_threads)
    print("{:.2f}s".format(time.time() - start))
    return tokenizer, [goal for rel_lemmas, prev_tactics,
                       hypotheses, goal, tactic in tokenized_data]

def tokenize_hyps(data : RawDataset, args : Namespace, tokenizer : Tokenizer) \
    -> List[List[Sentence]]:
    return [[tokenizer.toTokenList(hyp.partition(":")[2].strip())
             for hyp in hyp_list]
            for prevs, hyp_list, goal, tactic in data]

def predictKTactics(prediction_distribution : torch.FloatTensor,
                    embedding : Embedding, k : int) \
    -> List[Prediction]:
    if k > embedding.num_tokens():
        k = embedding.num_tokens()
    certainties_and_idxs = prediction_distribution.view(-1).topk(k)
    results = [Prediction(embedding.decode_token(stem_idx.data[0]) + ".",
                          math.exp(certainty.data[0]))
               for certainty, stem_idx in zip(*certainties_and_idxs)]
    return results

def predictKTacticsWithLoss(prediction_distribution : torch.FloatTensor,
                            embedding : Embedding,
                            k : int,
                            correct : str,
                            criterion : nn.Module) -> Tuple[List[Prediction], float]:
    if k > embedding.num_tokens():
        k = embedding.num_tokens()
    correct_stem = get_stem(correct)
    if embedding.has_token(correct_stem):
        output_var = maybe_cuda(Variable(
            torch.LongTensor([embedding.encode_token(correct_stem)])))
        loss = criterion(prediction_distribution.view(1, -1), output_var).item()
    else:
        loss = 0

    certainties_and_idxs = prediction_distribution.view(-1).topk(k)
    results = [Prediction(embedding.decode_token(stem_idx.item()) + ".",
                          math.exp(certainty.item()))
               for certainty, stem_idx in zip(*certainties_and_idxs)]

    return results, loss

def predictKTacticsWithLoss_batch(prediction_distributions : torch.FloatTensor,
                                  embedding : Embedding,
                                  k : int,
                                  correct_stems : List[str],
                                  criterion : nn.Module) -> \
                                  Tuple[List[List[Prediction]], float]:
    output_var = maybe_cuda(Variable(
        torch.LongTensor([embedding.encode_token(correct_stem)
                          if embedding.has_token(correct_stem)
                          else 0
                          for correct_stem in correct_stems])))
    loss = criterion(prediction_distributions, output_var).item()
    if k > embedding.num_tokens():
        k = embedding.num_tokens()
    certainties_and_idxs_list = [single_distribution.view(-1).topk(k)
                                 for single_distribution in
                                 list(prediction_distributions)]
    results = [[Prediction(embedding.decode_token(stem_idx.item()) + ".",
                           math.exp(certainty.item()))
                for certainty, stem_idx in zip(*certainties_and_idxs)]
               for certainties_and_idxs in certainties_and_idxs_list]
    return results, loss

def add_tokenizer_args(parser : argparse.ArgumentParser,
                       default_values : Dict[str, Any] = {}) -> None:
    parser.add_argument("--num-keywords", dest="num_keywords", type=int,
                        default=default_values.get("num-keywordes", 60))
    parser.add_argument("--tokenizer", choices=list(tokenizers.keys()), type=str,
                        default=default_values.get("tokenizer",
                                                   list(tokenizers.keys())[0]))
    parser.add_argument("--num-relevance-samples", dest="num_relevance_samples",
                        type=int, default=default_values.get("num_relevance_samples",
                                                             1000))
    parser.add_argument("--save-tokens", dest="save_tokens",
                        default=default_values.get("save-tokens", None))
    parser.add_argument("--load-tokens", dest="load_tokens",
                        default=default_values.get("load-tokens", None))
    parser.add_argument("--print-keywords", dest="print_keywords",
                        default=False, action='store_const', const=True)
