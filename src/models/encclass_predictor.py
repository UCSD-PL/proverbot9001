#!/usr/bin/env python3

import signal
import argparse
import time
import sys
import threading
import math

from models.encdecrnn_predictor import inputFromSentence
from tokenizer import Tokenizer, tokenizers
from data import get_text_data, filter_data, \
    encode_seq_classify_data, ClassifySequenceDataset
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

from models.tactic_predictor import TacticPredictor, Prediction
from typing import Dict, List, Union, Any, Tuple, Iterable, cast, Callable

class EncClassPredictor(TacticPredictor):
    def load_saved_state(self, filename : str) -> None:
        checkpoint = torch.load(filename)
        self.options = [("tokenizer", checkpoint['tokenizer-name']),
                        ("optimizer", checkpoint['optimizer']),
                        ("# encoder layers", checkpoint['num-encoder-layers']),
                        # ("# decoder layers", checkpoint['num-decoder-layers']),
                        ("input length", checkpoint['max-length']),
                        ("hidden size", checkpoint['hidden-size']),
                        ("# keywords", checkpoint['num-keywords']),
                        ("batch size", checkpoint['batch-size']),
                        ("learning rate", checkpoint['learning-rate']),
                        ("epoch step", checkpoint['epoch-step']),
                        ("gamma", checkpoint['gamma']),
                        ("epoch", checkpoint['epoch']),
                        ("training loss", "{:.4f}".format(checkpoint['training-loss'])),
                        ("context filter", checkpoint['context-filter']),
        ]

        self.tokenizer = checkpoint['tokenizer']
        self.embedding = checkpoint['embedding']
        self.encoder = maybe_cuda(RNNClassifier(self.tokenizer.numTokens(),
                                                checkpoint['hidden-size'],
                                                self.embedding.num_tokens(),
                                                checkpoint['num-encoder-layers'],
                                                1,
                                                # checkpoint['num-decoder-layers'],
                                                batch_size=1))
        self.encoder.load_state_dict(checkpoint['neural-encoder'])
        self.max_length = checkpoint["max-length"]

    def __init__(self, options : Dict[str, Any]) -> None:
        assert(options["filename"])
        self.load_saved_state(options["filename"])
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

def train(dataset : ClassifySequenceDataset,
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
                      args.num_decoder_layers,
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
                cast(torch.LongTensor, input_batch))
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

        yield (encoder.state_dict(), total_loss / items_processed)

def main(arg_list : List[str]) -> None:
    parser = start_std_args("a classifier pytorch model for proverbot")
    parser.add_argument("--print-keywords", dest="print_keywords",
                        default=False, action='store_const', const=True)
    args = parser.parse_args(arg_list)

    text_dataset = get_text_data(args.scrape_file, args.context_filter, verbose=True,
                                 max_tuples=args.max_tuples)
    substitutions = {"auto": "eauto.",
                     "intros until": "intros.",
                     "intro": "intros.",
                     "constructor": "econstructor."}
    preprocessed_dataset = [(hyps, goal, tactic
                             if get_stem(tactic) not in substitutions
                             else substitutions[get_stem(tactic)])
                            for hyps, goal, tactic in text_dataset]
    print("Encoding data...")
    start = time.time()
    dataset, tokenizer, embedding = encode_seq_classify_data(preprocessed_dataset,
                                                             tokenizers[args.tokenizer],
                                                             args.num_keywords, 2)
    if args.print_keywords:
        print("Keywords are {}".format(tokenizer.listTokens()))
    timeTaken = time.time() - start
    print("Encoded data in {:.2f}".format(timeTaken))

    checkpoints = train(dataset, tokenizer.numTokens(), embedding.num_tokens(), args)

    for epoch, (encoder_state, training_loss) in enumerate(checkpoints, start=1):
        state = {'epoch':epoch,
                 'training-loss': training_loss,
                 'tokenizer':tokenizer,
                 'tokenizer-name':args.tokenizer,
                 'optimizer':args.optimizer,
                 'batch-size':args.batch_size,
                 'learning-rate':args.learning_rate,
                 'gamma':args.gamma,
                 'epoch-step':args.epoch_step,
                 'embedding': embedding,
                 'neural-encoder':encoder_state,
                 'num-encoder-layers':args.num_encoder_layers,
                 'num-decoder-layers':args.num_decoder_layers,
                 'max-length': args.max_length,
                 'hidden-size' : args.hidden_size,
                 'num-keywords' : args.num_keywords,
                 'context-filter' : args.context_filter,
        }
        with open(args.save_file, 'wb') as f:
            print("=> Saving checkpoint at epoch {}".
                  format(epoch))
            torch.save(state, f)
