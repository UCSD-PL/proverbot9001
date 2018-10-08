#!/usr/bin/env python3

import re
import time
import argparse
from itertools import chain

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim import Optimizer
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
import torch.utils.data as data
import torch.cuda

from util import *
from models.args import add_std_args, optimizers
from data import read_text_data, Sentence
import tokenizer as tk

from typing import List, Dict, Tuple, NamedTuple, Iterable, Callable
from typing import cast

SOS_token = 1
EOS_token = 0

class EncoderRNN(nn.Module):
    def __init__(self, input_size : int, hidden_size : int,
                 num_layers : int, batch_size : int=1) -> None:
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = maybe_cuda(nn.Embedding(input_size, hidden_size))
        self.gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
    def forward(self, input : torch.LongTensor, hidden : torch.LongTensor) \
        -> Tuple[torch.LongTensor, torch.LongTensor] :
        embedded = self.embedding(input)
        output = embedded.view(1, self.batch_size, -1)
        for i in range(self.num_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        return output, hidden
    def initHidden(self) -> torch.FloatTensor:
        zeroes = cast(torch.FloatTensor, maybe_cuda(
            Variable(torch.zeros(1, self.batch_size, self.hidden_size))))
        return Variable(zeroes)
    def run(self, sentence : torch.LongTensor) -> torch.FloatTensor:
        encoder_input = maybe_cuda(Variable(sentence))
        encoder_hidden = self.initHidden()
        assert encoder_input.size()[0] == self.batch_size, \
            "input var has size {}, batch_size is {}".format(encoder_input.size()[0],
                                                             self.batch_size)
        for ei in range(encoder_input.size()[1]):
            encoder_output, encoder_hidden = self(encoder_input[:, ei], encoder_hidden)
        return encoder_hidden
    pass
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size : int, output_size : int, num_layers : int,
                 batch_size : int =1, beam_width : int =1) -> None:
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = maybe_cuda(nn.Embedding(output_size, hidden_size))
        self.gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        self.out = maybe_cuda(nn.Linear(hidden_size, output_size))
        self.softmax = maybe_cuda(nn.LogSoftmax(1))
        self.beam_width = beam_width
        self.batch_size = batch_size


    def forward(self, input : torch.LongTensor, hidden : torch.LongTensor) \
        -> Tuple[torch.LongTensor, torch.LongTensor]:
        output = self.embedding(input).view(1, self.batch_size * self.beam_width, -1)
        for i in range(self.num_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initInput(self) -> torch.LongTensor:
        return Variable(LongTensor([[SOS_token] * self.batch_size]))

    def initHidden(self) -> torch.LongTensor:
        zeroes = cast(torch.LongTensor, maybe_cuda(torch.zeros(1, 1, self.hidden_size)))
        return Variable(zeroes)
    def run_teach(self, hidden : torch.FloatTensor,
                  output_batch : torch.LongTensor) -> List[torch.FloatTensor]:
        output_variable = maybe_cuda(Variable(output_batch))
        decoder_hidden = hidden
        decoder_input = self.initInput()
        prediction = []

        for di in range(output_variable.size()[1]):
            decoder_output, decoder_hidden = self(decoder_input, decoder_hidden)
            decoder_input = output_variable[:,di]
            prediction.append(decoder_output)
        return prediction
    pass

class Checkpoint(NamedTuple):
    encoder_state : Dict[Any, Any]
    decoder_state : Dict[Any, Any]
    training_loss : float

def train(dataset : List[Sentence],
          token_vocab_size : int, max_length : int, hidden_size : int,
          learning_rate : float, epoch_step : int, gamma : float,
          num_encoder_layers : int, num_decoder_layers : int,
          num_epochs : int, batch_size : int, print_every : int,
          optimizer_f : Callable[..., Optimizer]) \
          -> Iterable[Checkpoint]:
    data_loader = data.DataLoader(data.TensorDataset(torch.LongTensor(dataset[:]),
                                                     torch.LongTensor(dataset[:])),
                                  batch_size=batch_size, num_workers=0,
                                  shuffle=True, pin_memory=True,
                                  drop_last=True)

    encoder = maybe_cuda(EncoderRNN(token_vocab_size, hidden_size,
                                    num_encoder_layers, batch_size=batch_size))
    decoder = maybe_cuda(DecoderRNN(hidden_size, token_vocab_size,
                                    num_decoder_layers, batch_size))
    encoder_optimizer = optimizer_f(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optimizer_f(decoder.parameters(), lr=learning_rate)
    encoder_adjuster = scheduler.StepLR(encoder_optimizer, epoch_step, gamma)
    decoder_adjuster = scheduler.StepLR(decoder_optimizer, epoch_step, gamma)
    criterion = maybe_cuda(nn.NLLLoss())

    start=time.time()
    num_items = len(dataset) * num_epochs
    total_loss = 0

    print("Training...")
    for epoch in range(num_epochs):
        print("Epoch {}".format(epoch))
        # Adjust learning rates if needed
        encoder_adjuster.step()
        decoder_adjuster.step()

        # Process batches of data
        for batch_num, (input_batch, output_batch) in enumerate(data_loader):
            # Reset the optimizers
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Run the autoencoder
            decoded_output = \
                decoder.run_teach(
                    encoder.run(cast(torch.LongTensor, input_batch)),
                    cast(torch.LongTensor, output_batch))

            # Gather the losses
            loss = maybe_cuda(Variable(torch.zeros(1, dtype=torch.float32)))
            output_var = maybe_cuda(Variable(output_batch))
            target_length = output_batch.size()[1]
            for i in range(target_length):
                loss += criterion(decoded_output[i], output_var[:,i])
            total_loss += (loss.data.item() / target_length) * batch_size
            assert isinstance(total_loss, float)

            # Update the weights
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            # Print status every once in a while
            if (batch_num + 1) % print_every == 0:
                items_processed = (batch_num + 1) * batch_size + epoch * len(dataset)
                progress = items_processed / num_items
                print("{} ({} {:.2f}%) {:.4f}".
                      format(timeSince(start, progress),
                             items_processed, progress * 100,
                             total_loss / items_processed))

        yield Checkpoint(encoder_state=encoder.state_dict(),
                         decoder_state=decoder.state_dict(),
                         training_loss=total_loss)
    pass

def normalizeSentenceLength(sentence : Sentence, max_length : int) -> Sentence:
    if len(sentence) > max_length:
        sentence = sentence[:max_length]
    elif len(sentence) < max_length:
        sentence.extend([EOS_token] * (max_length - len(sentence)))
    return sentence

def main(args_list : List[str]) -> None:
    parser = argparse.ArgumentParser(description="Autoencoder for coq terms")
    add_std_args(parser)
    parser.add_argument("--gamma", default=.9, type=float)
    parser.add_argument("--epoch-step", default=5, type=int)
    parser.add_argument("--num-decoder-layers", dest="num_decoder_layers",
                        default=3, type=int)
    args = parser.parse_args(args_list)
    print("Loading data...")
    dataset = read_text_data(args.scrape_file, args.max_tuples)
    for hyps, goal, tactic in dataset:
        for hyp in hyps:
            assert ":" in hyp, "hyps: {}".format(hyps)
    term_strings = list(chain.from_iterable(
        [[hyp.split(":")[1].strip() for hyp in hyps] + [goal]
         for hyps, goal, tactic in dataset]))

    print("Parsing data...")
    tokenizer = tk.make_keyword_tokenizer(term_strings,
                                          tk.tokenizers[args.tokenizer],
                                          args.num_keywords, 2)
    tokenized_data = [normalizeSentenceLength(tokenizer.toTokenList(term), args.max_length)
                      for term in term_strings]
    checkpoints = train(tokenized_data,
                        tokenizer.numTokens(), args.max_length, args.hidden_size,
                        args.learning_rate, args.epoch_step, args.gamma,
                        args.num_encoder_layers, args.num_decoder_layers,
                        args.num_epochs, args.batch_size, args.print_every,
                        optimizers[args.optimizer])
    for epoch, (encoder_state, decoder_state, training_loss) in enumerate(checkpoints):
        state = {'epoch':epoch,
                 'training-loss': training_loss,
                 'tokenizer':tokenizer,
                 'tokenizer-name':args.tokenizer,
                 'optimizer':args.optimizer,
                 'learning-rate':args.learning_rate,
                 'encoder':encoder_state,
                 'decoder':decoder_state,
                 'num-encoder-layers':args.num_encoder_layers,
                 'max-length': args.max_length,
                 'hidden-size' : args.hidden_size,
                 'num-keywords' : args.num_keywords,
                 'context-filter' : args.context_filter,
        }
        with open(args.save_file, 'wb') as f:
            print("=> Saving checkpoint at epoch {}".
                  format(epoch))
            torch.save(state, f)
    pass
