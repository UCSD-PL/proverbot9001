#!/usr/bin/env python3

import signal
import argparse
from argparse import Namespace
import time
import sys
import threading
import math
import random
import multiprocessing
import functools
from dataclasses import dataclass
from itertools import chain

from models.encdecrnn_predictor import inputFromSentence
from models.components import SimpleEmbedding
from tokenizer import Tokenizer, tokenizers, make_keyword_tokenizer_relevance
from data import get_text_data, filter_data, \
    encode_seq_classify_data, ScrapedTactic, Sentence, Dataset, RawDataset
from util import *
from context_filter import get_context_filter
from serapi_instance import get_stem
from models.args import start_std_args, optimizers

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim import Optimizer
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
import torch.utils.data as data
import torch.cuda

from models.tactic_predictor import TacticPredictor, Prediction, TrainablePredictor, TokenizerEmbeddingState, NeuralPredictorState
from typing import Dict, List, Union, Any, Tuple, NamedTuple, Iterable, cast, Callable, Optional

class ECSample(NamedTuple):
    goal : Sentence
    tactic : int

@dataclass(init=True, repr=True)
class ECDataset(Dataset):
    data : List[ECSample]
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i : Any):
        return self.data[i]

class EncClassPredictor(TrainablePredictor[ECDataset, TokenizerEmbeddingState, NeuralPredictorState]):
    def __init__(self) -> None:
        self.criterion = maybe_cuda(nn.NLLLoss())
        self.lock = threading.Lock()

    def predictDistribution(self, in_data : Dict[str, Union[List[str], str]]) \
        -> torch.FloatTensor:
        tokenized_goal = self.tokenizer.toTokenList(in_data["goal"])
        input_list = inputFromSentence(tokenized_goal, self.max_length)
        input_tensor = LongTensor(input_list).view(1, -1)
        return self.encoder.run(input_tensor)

    def predictKTactics(self, in_data : Dict[str, Union[List[str], str]], k : int) \
        -> List[Prediction]:
        self.lock.acquire()
        prediction_distribution = self.predictDistribution(in_data)
        if k > self.embedding.num_tokens():
            k= self.embedding.num_tokens()
        certainties_and_idxs = prediction_distribution.view(-1).topk(k)
        results = [Prediction(self.embedding.decode_token(stem_idx.data[0]) + ".",
                              math.exp(certainty.data[0]))
                   for certainty, stem_idx in zip(*certainties_and_idxs)]
        self.lock.release()
        return results

    def predictKTacticsWithLoss(self, in_data : Dict[str, Union[List[str], str]], k : int,
                                correct : str) -> Tuple[List[Prediction], float]:
        self.lock.acquire()
        prediction_distribution = self.predictDistribution(in_data)
        correct_stem = get_stem(correct)
        if self.embedding.has_token(correct_stem):
            output_var = maybe_cuda(Variable(
                torch.LongTensor([self.embedding.encode_token(correct_stem)])))
            loss = self.criterion(prediction_distribution.view(1, -1), output_var).item()
        else:
            loss = 0

        if k > self.embedding.num_tokens():
            k = self.embedding.num_tokens()
        certainties_and_idxs = prediction_distribution.view(-1).topk(k)
        results = [Prediction(self.embedding.decode_token(stem_idx.item()) + ".",
                              math.exp(certainty.item()))
                   for certainty, stem_idx in zip(*certainties_and_idxs)]

        self.lock.release()
        return results, loss

    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[Dict[str, Union[str, List[str]]]],
                                      k : int, corrects : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]:
        if len(in_data) == 0:
            return [], 0
        self.lock.acquire()
        tokenized_goals = [self.tokenizer.toTokenList(in_data_point["goal"])
                           for in_data_point in in_data]
        input_tensor = LongTensor([inputFromSentence(tokenized_goal, self.max_length)
                                  for tokenized_goal in tokenized_goals])
        prediction_distributions = self.encoder.run(input_tensor, batch_size=len(in_data))
        correct_stems = [get_stem(correct) for correct in corrects]
        output_var = maybe_cuda(Variable(
            torch.LongTensor([self.embedding.encode_token(correct_stem)
                              if self.embedding.has_token(correct_stem)
                              else 0
                              for correct_stem in correct_stems])))
        loss = self.criterion(prediction_distributions, output_var).item()

        if k > self.embedding.num_tokens():
            k = self.embedding.num_tokens()

        certainties_and_idxs_list = [single_distribution.view(-1).topk(k)
                                for single_distribution in list(prediction_distributions)]
        results = [[Prediction(self.embedding.decode_token(stem_idx.item()) + ".",
                               math.exp(certainty.item()))
                    for certainty, stem_idx in zip(*certainties_and_idxs)]
                   for certainties_and_idxs in certainties_and_idxs_list]
        self.lock.release()
        return results, loss

    def getOptions(self) -> List[Tuple[str, str]]:
        return self.options

    def add_args_to_parser(self, parser : argparse.ArgumentParser,
                           default_values : Dict[str, Any] = {}) -> None:
        super().add_args_to_parser(parser, default_values)
        parser.add_argument("--print-keywords", dest="print_keywords",
                            default=False, action='store_const', const=True)
        parser.add_argument("--num-epochs", dest="num_epochs", type=int,
                            default=default_values.get("num-epochs", 20))
        parser.add_argument("--batch-size", dest="batch_size", type=int,
                            default=default_values.get("batch-size", 256))
        parser.add_argument("--max-length", dest="max_length", type=int,
                            default=default_values.get("max-length", 100))
        parser.add_argument("--start-from", dest="start_from", type=str,
                            default=default_values.get("start-from", None))
        parser.add_argument("--print-every", dest="print_every", type=int,
                            default=default_values.get("print-every", 5))
        parser.add_argument("--hidden-size", dest="hidden_size", type=int,
                            default=default_values.get("hidden-size", 128))
        parser.add_argument("--learning-rate", dest="learning_rate", type=float,
                            default=default_values.get("learning-rate", .7))
        parser.add_argument("--epoch-step", dest="epoch_step", type=int,
                            default=default_values.get("epoch-step", 10))
        parser.add_argument("--gamma", dest="gamma", type=float,
                            default=default_values.get("gamma", 0.8))
        parser.add_argument("--num-encoder-layers", dest="num_encoder_layers", type=int,
                            default=default_values.get("num-encoder-layers", 3))
        parser.add_argument("--num-keywords", dest="num_keywords", type=int,
                            default=default_values.get("num-keywordes", 60))
        parser.add_argument("--tokenizer", choices=list(tokenizers.keys()), type=str,
                            default=default_values.get("tokenizer",
                                                       list(tokenizers.keys())[0]))
        parser.add_argument("--save-tokens", dest="save_tokens",
                            default=default_values.get("save-tokens", None))
        parser.add_argument("--load-tokens", dest="load_tokens",
                            default=default_values.get("load-tokens", None))
        parser.add_argument("--optimizer",
                            choices=list(optimizers.keys()), type=str,
                            default=default_values.get("optimizer",
                                                       list(optimizers.keys())[0]))
    def _encode_data(self, data : RawDataset, arg_values : Namespace) \
        -> Tuple[ECDataset, TokenizerEmbeddingState]:
        preprocessed_data = self._preprocess_data(data, arg_values)
        dataset, tokenizer, embedding = \
            encode_seq_classify_data(preprocessed_data,
                                     tokenizers[arg_values.tokenizer],
                                     arg_values.num_keywords, 2,
                                     arg_values.save_tokens,
                                     arg_values.load_tokens)
        if arg_values.print_keywords:
            print("Keywords are {}".format(tokenizer.listTokens()))
        return dataset, TokenizerEmbeddingState(tokenizer, embedding)
    def _optimize_model_to_disc(self,
                                encoded_data : ECDataset,
                                encdec_state : TokenizerEmbeddingState,
                                arg_values : Namespace) \
        -> None:
        for epoch, predictor_state in enumerate(
                train(encoded_data,
                      encdec_state.tokenizer.numTokens(),
                      encdec_state.embedding.num_tokens(),
                      arg_values),
                start=1):
            with open(arg_values.save_file, 'wb') as f:
                print("=> Saving checkpoint at epoch {}".format(epoch))
                torch.save((arg_values, encdec_state, predictor_state), f)
    def _description(self) -> str:
        return "a classifier pytorch model for proverbot"
    def load_saved_state(self,
                         args : Namespace,
                         metadata : TokenizerEmbeddingState,
                         state : NeuralPredictorState) -> None:
        self.options = list(vars(args).items()) + \
            [("training loss", self.training_loss),
             ("# epochs", self.num_epochs),
             ("skip nochange tactics:", str(options["skip-nochange-tac"]))]
        self.tokenizer, self.embedding = metadata
        self.encoder = maybe_cuda(RNNClassifier(self.tokenizer.numTokens(),
                                                arg_values.hidden_size,
                                                self.embedding.num_tokens(),
                                                arg_values.num_encoder_layers,
                                                1,
                                                batch_size=1))
        self.encoder.load_state_dict(state.weights)
        self.max_length = arg_values.max_length

class RNNClassifier(nn.Module):
    def __init__(self, input_vocab_size : int, hidden_size : int, output_vocab_size: int,
                 num_encoder_layers : int, num_decoder_layers : int =1,
                 batch_size : int=1) -> None:
        super(RNNClassifier, self).__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = maybe_cuda(nn.Embedding(input_vocab_size, hidden_size))
        self.gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self.decoder_out = maybe_cuda(nn.Linear(hidden_size, output_vocab_size))
        self.softmax = maybe_cuda(nn.LogSoftmax(dim=1))

    def forward(self, input : torch.FloatTensor, hidden : torch.FloatTensor) \
        -> Tuple[torch.FloatTensor, torch.FloatTensor] :
        output = self.embedding(input).view(1, self.batch_size, -1)
        for i in range(self.num_encoder_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.decoder_out(output[0]))
        return output, hidden

    def initHidden(self):
        return maybe_cuda(Variable(torch.zeros(1, self.batch_size, self.hidden_size)))

    def run(self, input : torch.LongTensor, batch_size : int=1):
        self.batch_size = batch_size
        in_var = maybe_cuda(Variable(input))
        hidden = self.initHidden()
        for i in range(in_var.size()[1]):
            output, hidden = self(in_var[:,i], hidden)
        return output.view(self.batch_size, -1)

Checkpoint = Tuple[Dict[Any, Any], float]

def train(dataset : ECDataset,
          input_vocab_size : int,
          output_vocab_size : int,
          args : argparse.Namespace) -> Iterable[Checkpoint]:
    print("Initializing PyTorch...")
    in_stream = [inputFromSentence(datum[0], args.max_length) for datum in dataset]
    out_stream = [datum[1] for datum in dataset]
    dataloader = data.DataLoader(data.TensorDataset(torch.LongTensor(in_stream),
                                                    torch.LongTensor(out_stream)),
                                 batch_size=args.batch_size, num_workers=0,
                                 shuffle=True, pin_memory=True, drop_last=True)

    encoder = maybe_cuda(
        RNNClassifier(input_vocab_size, args.hidden_size, output_vocab_size,
                      args.num_encoder_layers,
                      batch_size=args.batch_size))
    optimizer = optimizers[args.optimizer](encoder.parameters(), lr=args.learning_rate)
    criterion = maybe_cuda(nn.NLLLoss())
    adjuster = scheduler.StepLR(optimizer, args.epoch_step, gamma=args.gamma)
    lsoftmax = maybe_cuda(nn.LogSoftmax(1))

    start=time.time()
    num_items = len(dataset) * args.num_epochs
    total_loss = 0

    print("Training...")
    for epoch in range(1, args.num_epochs+1):
        print("Epoch {}".format(epoch))
        adjuster.step()
        for batch_num, (input_batch, output_batch) in enumerate(dataloader):

            optimizer.zero_grad()

            prediction_distribution = encoder.run(
                cast(torch.LongTensor, input_batch),
                batch_size=args.batch_size)
            loss = cast(torch.FloatTensor, 0)
            output_var = maybe_cuda(Variable(output_batch))
            loss += criterion(prediction_distribution, output_var)
            loss.backward()

            optimizer.step()

            total_loss += loss.data.item() * args.batch_size

            if (batch_num + 1) % args.print_every == 0:

                items_processed = (batch_num + 1) * args.batch_size + \
                    (epoch - 1) * len(dataset)
                progress = items_processed / num_items
                print("{} ({:7} {:5.2f}%) {:.4f}".
                      format(timeSince(start, progress),
                             items_processed, progress * 100,
                             total_loss / items_processed))

        yield (epoch, NeuralPredictorState(epoch,
                                           total_loss / items_processed,
                                           encoder.state_dict()))

def main(arg_list : List[str]) -> None:
    predictor = EncClassPredictor()
    predictor.train(arg_list)

def encode_seq_classify_data(data : RawDataset,
                             tokenizer_type : Callable[[List[str], int], Tokenizer],
                             num_keywords : int,
                             num_reserved_tokens : int,
                             save_tokens : Optional[str] = None,
                             load_tokens : Optional[str] = None,
                             num_relevance_samples : int = 1000) \
    -> Tuple[ECDataset, Tokenizer, SimpleEmbedding]:
    embedding = SimpleEmbedding()
    data = list(data)
    subset = RawDataset(random.sample(data, num_relevance_samples))
    if load_tokens:
        print("Loading tokens from {}".format(load_tokens))
        tokenizer = torch.load(load_tokens)
    else:
        start = time.time()
        print("Picking tokens...", end="")
        sys.stdout.flush()
        tokenizer = make_keyword_tokenizer_relevance([(context,
                                                       embedding.encode_token(
                                                           get_stem(tactic)))
                                                      for prev_tactics, hyps,
                                                      context, tactic
                                                      in subset],
                                                     tokenizer_type,
                                                     num_keywords, num_reserved_tokens)
        print("{}s".format(time.time() - start))
    if save_tokens:
        print("Saving tokens to {}".format(save_tokens))
        torch.save(tokenizer, save_tokens)
    with multiprocessing.Pool(None) as pool:
        result = ECDataset([ECSample(goal, embedding.encode_token(tactic))
                            for goal, tactic in
                            chain.from_iterable(pool.imap_unordered(functools.partial(
                                encode_seq_classify_data_worker__, tokenizer),
                                                                    chunks(data, 1024)))])
    tokenizer.freezeTokenList()
    return result, tokenizer, embedding
def encode_seq_classify_data_worker__(tokenizer : Tokenizer,
                                      chunk : List[Tuple[List[str], List[str], str, str]])\
    -> List[Tuple[Sentence, str]]:
    return [(tokenizer.toTokenList(goal), get_stem(tactic))
            for prev_tactics, hyps, goal, tactic in chunk]
