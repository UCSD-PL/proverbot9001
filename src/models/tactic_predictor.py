#!/usr/bin/env python3

from typing import Dict, List, Union, Tuple, Iterable, NamedTuple, Sequence, Any
from abc import ABCMeta, abstractmethod

class Prediction(NamedTuple):
    prediction : str
    certainty : float

ContextInfo = Dict[str, Union[str, List[str]]]
class TacticContext(NamedTuple):
    prev_tactics : List[str]
    hypotheses : List[str]
    goal : str

class TacticPredictor(metaclass=ABCMeta):
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

from data import Dataset, RawDataset, ScrapedTactic, get_text_data, TokenizedDataset, \
    DatasetMetadata, stemmify_data, tactic_substitutions
from typing import TypeVar, Generic
import argparse
from argparse import Namespace
from serapi_instance import get_stem

class PredictorState(metaclass=ABCMeta):
    pass

DatasetType = TypeVar('DatasetType', bound=Dataset)
MetadataType = TypeVar('MetadataType', bound=DatasetMetadata)
StateType = TypeVar('StateType', bound=PredictorState)

class TrainablePredictor(TacticPredictor, Generic[DatasetType, MetadataType, StateType],
                         metaclass=ABCMeta):
    def train(self, args : List[str]) -> None:
        argparser = argparse.ArgumentParser(self._description())
        self.add_args_to_parser(argparser)
        arg_values = argparser.parse_args(args)
        text_data = get_text_data(arg_values.scrape_file, arg_values.context_filter,
                                  max_tuples=arg_values.max_tuples, verbose=True)
        encoded_data, encdec_state = self._encode_data(text_data, arg_values)
        del text_data
        gc.collect()
        self._optimize_model_to_disc(encoded_data, encdec_state, arg_values)

    def add_args_to_parser(self,
                           parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) \
        -> None:
        parser.add_argument("scrape_file")
        parser.add_argument("save_file")
        parser.add_argument("--num-threads", "-j", dest="num_threads", type=int,
                            default=default_values.get("num-threads", None))
        parser.add_argument("--max-tuples", dest="max_tuples", type=int,
                            default=default_values.get("max-tuples", None))
        parser.add_argument("--context-filter", dest="context_filter", type=str,
                            default=default_values.get("context-filter",
                                                       "goal-changes%no-args"))
        parser.add_argument("--use-substitutions", dest="use_substitutions", type=bool,
                            default=default_values.get("use_substitutions", True))
        pass

    @abstractmethod
    def _encode_data(self, data : RawDataset, arg_values : Namespace) \
        -> Tuple[DatasetType, MetadataType]: pass
    def _preprocess_data(self, text_dataset : RawDataset, arg_values : Namespace) \
        -> Iterable[ScrapedTactic]:
        if arg_values.use_substitutions:
            print("Preprocessing...")
            substitutions = {"auto": "eauto.",
                             "intros until": "intros.",
                             "intro": "intros.",
                             "constructor": "econstructor."}
            with multiprocessing.Pool(arg_values.num_threads) as pool:
                iterator = pool.imap(
                    functools.partial(tactic_substitutions, substitutions),
                    text_dataset)
                yield from iterator
        else:
            yield from text_dataset
    @abstractmethod
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
                         metadata : MetadataType,
                         state : StateType) -> None: pass
    pass

from tokenizer import make_keyword_tokenizer_relevance, tokenizers, Tokenizer, get_words
from data import EmbeddedSample, EmbeddedDataset, StrictEmbeddedDataset, LazyEmbeddedDataset, DatasetMetadata, tokenize_data
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
        parser.add_argument("--num-relevance-samples", dest="num_relevance_samples",
                            type=int, default=default_values.get("num_relevance_samples",
                                                                 1000))
        parser.add_argument("--save-tokens", dest="save_tokens",
                            default=default_values.get("save-tokens", None))
        parser.add_argument("--load-tokens", dest="load_tokens",
                            default=default_values.get("load-tokens", None))
        parser.add_argument("--print-keywords", dest="print_keywords",
                            default=False, action='store_const', const=True)
    @abstractmethod
    def _encode_tokenized_data(self, data : TokenizedDataset, arg_values : Namespace,
                               term_vocab_size : int, tactic_vocab_size : int) \
        -> DatasetType:
        pass

    def _encode_data(self, data : RawDataset, args : Namespace) \
        -> Tuple[DatasetType, TokenizerEmbeddingState]:
        preprocessed_data = super()._preprocess_data(data, args)
        embedding = SimpleEmbedding()
        embedded_data : EmbeddedDataset
        with multiprocessing.Pool(args.num_threads) as pool:
            stemmed_data = pool.imap(
                stemmify_data, preprocessed_data, chunksize=10240)
            lazy_embedded_data = LazyEmbeddedDataset((
                EmbeddedSample(prev_tactics, hypotheses, goal,
                               embedding.encode_token(tactic))
                for (prev_tactics, hypotheses, goal, tactic)
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
                    tokenizers[args.tokenizer], args.num_keywords, 2, args.num_threads)
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
            tokenized_data = tokenize_data(tokenizer, embedding, embedded_data, args.num_threads)
            gc.collect()

        return self._encode_tokenized_data(tokenized_data, args,
                                           tokenizer.numTokens(),
                                           embedding.num_tokens()), \
            TokenizerEmbeddingState(tokenizer, embedding)
    @abstractmethod
    def load_saved_state(self,
                         args : Namespace,
                         metadata : TokenizerEmbeddingState,
                         state : StateType) -> None:
        pass
import torch
import torch.utils.data as data
import torch.optim.lr_scheduler as scheduler
from torch import optim
import torch.nn as nn
from util import *
from util import chunks

optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
}

ModelType = TypeVar('ModelType', bound=nn.Module)

@dataclass(init=True)
class NeuralPredictorState(PredictorState):
    epoch : int
    loss : float
    weights : Dict[str, Any]

class NeuralPredictor(Generic[DatasetType, ModelType],
                      TokenizingPredictor[DatasetType, NeuralPredictorState],
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
        parser.add_argument("--epoch-step", dest="epoch_step", type=int,
                            default=default_values.get("epoch-step", 10))
        parser.add_argument("--gamma", dest="gamma", type=float,
                            default=default_values.get("gamma", 0.8))
        parser.add_argument("--optimizer",
                            choices=list(optimizers.keys()), type=str,
                            default=default_values.get("optimizer",
                                                       list(optimizers.keys())[0]))

    def _optimize_model_to_disc(self,
                                encoded_data : DatasetType,
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
    def _optimize_checkpoints(self, encoded_data : DatasetType, arg_values : Namespace,
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
            with open(arg_values.start_from, 'rb') as f:
                state = torch.load(f)
                self.load_saved_state(*state) # type: ignore
            model = self._model
        else:
            model = maybe_cuda(self._get_model(arg_values, tactic_vocab_size,
                                               term_vocab_size))
        optimizer = optimizers[arg_values.optimizer](model.parameters(),
                                                     lr=arg_values.learning_rate)
        adjuster = scheduler.StepLR(optimizer, arg_values.epoch_step,
                                    gamma=arg_values.gamma)

        training_start=time.time()

        print("Training...")
        for epoch in range(1, arg_values.num_epochs + 1):
            adjuster.step()
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
            yield NeuralPredictorState(epoch,
                                       epoch_loss / num_batches,
                                       model.state_dict())
    def load_saved_state(self,
                         args : Namespace,
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
                               term_vocab_size : int, tactic_vocab_size : int) \
        -> DatasetType:
        pass
    @abstractmethod
    def _data_tensors(self, encoded_data : DatasetType,
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
