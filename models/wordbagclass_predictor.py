#!/usr/bin/env python3

import argparse
import time
from typing import Dict, Any, List, Tuple, Iterable, cast

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.optim.lr_scheduler as scheduler
import torch.utils.data as data

from models.tactic_predictor import TacticPredictor
from models.components import SimpleEmbedding
from format import read_pair
import tokenizer
from tokenizer import context_vocab_size, tokenize_context, \
    get_tokenizer_state, set_tokenizer_state, get_topk_keywords

from util import *

class WordBagClassifyPredictor(TacticPredictor):
    def load_saved_state(self, filename : str) -> None:
        checkpoint = torch.load(filename)
        assert checkpoint['stem-embeddings']

        self.embedding = checkpoint['stem-embeddings']
        set_tokenizer_state(checkpoint['text-encoder'])
        self.linear = maybe_cuda(nn.Linear(context_vocab_size(),
                                           self.embedding.num_tokens()))
        self.linear.load_state_dict(checkpoint['linear-state'])
        self.lsoftmax = maybe_cuda(nn.LogSoftmax(dim=1))

        self.options = checkpoint['options']
        self.criterion = maybe_cuda(nn.NLLLoss())
        pass

    def getOptions(self) -> List[Tuple[str, str]]:
        return [("classifier", str(True)),
                ("num-stems", str(self.embedding.num_tokens())),
                ] + self.options

    def __init__(self, options : Dict[str, Any]) -> None:
        assert options["filename"]
        self.load_saved_state(options["filename"])

    def predictDistribution(self, in_data : Dict[str, str]) -> torch.FloatTensor:
        goal = in_data["goal"]
        in_vec = Variable(FloatTensor(getWordbagVector(tokenize_context(goal))))\
                 .view(1, -1)
        return self.lsoftmax(self.linear(in_vec))

    def predictKTactics(self, in_data : Dict[str, str], k : int) -> List[str]:
        distribution = self.predictDistribution(in_data)
        probs, indices = distribution.squeeze().topk(k)
        return [self.embedding.decode_token(idx.data[0]) + "." for idx in indices]

    def predictKTacticsWithLoss(self, in_data : Dict[str, str], k : int,
                                correct : str) -> Tuple[List[str], float]:
        distribution = self.predictDistribution(in_data)
        stem = get_stem(correct)
        if self.embedding.has_token(stem):
            output_var = maybe_cuda(Variable(torch.
                                             LongTensor([self.embedding.encode_token\
                                                         (get_stem(correct))])))
            loss = self.criterion(distribution, output_var).data[0]
        else:
            loss = 0

        probs, indices = distribution.squeeze().topk(k)
        predictions = [self.embedding.decode_token(idx.data[0]) + "."
                       for idx in indices]
        return predictions, loss

def read_scrapefile(filename : str, embedding : SimpleEmbedding) -> \
    List[Tuple[List[int], int]]:
    dataset = []
    untokenized_contexts = []
    print("Loading first pass contexts...")
    with open(filename, 'r') as scrapefile:
        pair = read_pair(scrapefile)
        while pair:
            context, _ = pair
            untokenized_contexts.append(context)
            pair = read_pair(scrapefile)
    print("Getting keywords...")
    keywords = get_topk_keywords(untokenized_contexts, 100)
    print("Building tokenizer...")
    tokenizer.contextTokenizer = tokenizer.KeywordTokenizer(keywords, 2)
    print("Loading and tokenizing file...")
    with open(filename, 'r') as scrapefile:
        pair = read_pair(scrapefile)
        while pair:
            context, tactic = pair
            if (not re.match("[\{\}\+\-\*].*", tactic)) and \
               (not re.match(".*;.*", tactic)):
                dataset.append((tokenize_context(context),
                                embedding.encode_token(get_stem(tactic))))
            pair = read_pair(scrapefile)
    print("Done.")
    return dataset

def getWordbagVector(goal : List[int]) -> List[int]:
    wordbag = [0] * context_vocab_size()
    for t in goal:
        assert t < context_vocab_size(), \
            "t: {}, context_vocab_size(): {}".format(t, context_vocab_size())
        wordbag[t] += 1
    return wordbag

Checkpoint = Tuple[Dict[Any, Any], float]

def main(args_list : List[str]) -> None:
    parser = argparse.ArgumentParser(description=
                                     "A second-tier predictor which predicts tactic "
                                     "stems based on word frequency in the goal")
    parser.add_argument("--learning-rate", dest="learning_rate", default=.5, type=float)
    parser.add_argument("--num-epochs", dest="num_epochs", default=10, type=int)
    parser.add_argument("--batch-size", dest="batch_size", default=256, type=int)
    parser.add_argument("--print-every", dest="print_every", default=10, type=int)
    parser.add_argument("--epoch-step", dest="epoch_step", default=5, type=int)
    parser.add_argument("--gamma", dest="gamma", default=0.5, type=float)
    parser.add_argument("--optimizer", default="SGD",
                        choices=list(optimizers.keys()), type=str)
    parser.add_argument("--disable-keywords", dest="disable_keywords",
                        default=False, const=True, action="store_const")
    parser.add_argument("scrape_file")
    parser.add_argument("save_file")
    args = parser.parse_args(args_list)
    if args.disable_keywords:
        tokenizer.disable_keywords()

    embedding = SimpleEmbedding()

    print("Loading dataset...")

    dataset = read_scrapefile(args.scrape_file, embedding)

    checkpoints = train(dataset, args.learning_rate,
                        args.num_epochs, args.batch_size,
                        embedding.num_tokens(), args.print_every,
                        args.gamma, args.epoch_step, args.optimizer)

    for epoch, (linear_state, loss) in enumerate(checkpoints, start=1):
        state = {'epoch':epoch,
                 'text-encoder':get_tokenizer_state(),
                 'linear-state': linear_state,
                 'stem-embeddings': embedding,
                 'options': [
                     ("# epochs", str(epoch)),
                     ("learning rate", str(args.learning_rate)),
                     ("batch size", str(args.batch_size)),
                     ("epoch step", str(args.epoch_step)),
                     ("gamma", str(args.gamma)),
                     ("dataset size", str(len(dataset))),
                     ("use keywords", str(not args.disable_keywords)),
                     ("optimizer", args.optimizer),
                     ("training loss", "{:10.2f}".format(loss)),
                 ]}
        with open(args.save_file, 'wb') as f:
            print("=> Saving checkpoint at epoch {}".
                  format(epoch))
            torch.save(state, f)

optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
}

def train(dataset, learning_rate : float, num_epochs : int,
          batch_size : int, num_stems: int, print_every : int,
          gamma : float, epoch_step : int, optimizer_type : str) -> Iterable[Checkpoint]:
    print("Initializing PyTorch...")
    linear = maybe_cuda(nn.Linear(context_vocab_size(), num_stems))
    lsoftmax = maybe_cuda(nn.LogSoftmax(1))

    inputs, outputs = zip(*dataset)
    dataloader = data.DataLoader(
        data.TensorDataset(
            torch.FloatTensor([getWordbagVector(input) for input in inputs]),
            torch.LongTensor(outputs)),
        batch_size=batch_size, num_workers=0,
        shuffle=True, pin_memory=True, drop_last=True)

    optimizer = optimizers[optimizer_type](linear.parameters(), lr=learning_rate)
    criterion = maybe_cuda(nn.NLLLoss())
    adjuster = scheduler.StepLR(optimizer, epoch_step, gamma=gamma)

    start=time.time()
    num_items = len(dataset) * num_epochs
    total_loss = 0

    print("Training...")
    for epoch in range(num_epochs):
        print("Epoch {}".format(epoch))
        adjuster.step()

        for batch_num, (input_batch, output_batch) in enumerate(dataloader):

            optimizer.zero_grad()
            input_var = maybe_cuda(Variable(input_batch))
            output_var = maybe_cuda(Variable(output_batch))

            prediction_distribution = lsoftmax(linear(input_var))

            loss = cast(torch.FloatTensor, 0) # type: torch.FloatTensor

            loss += criterion(prediction_distribution, output_var)

            loss.backward()

            optimizer.step()
            total_loss += loss.data[0] * batch_size

            if (batch_num + 1) % print_every == 0:

                items_processed = (batch_num + 1) * batch_size + epoch * len(dataset)
                progress = items_processed / num_items
                print("{} ({:7} {:5.2f}%) {:.4f}".
                      format(timeSince(start, progress),
                             items_processed, progress * 100,
                             total_loss / items_processed))
        yield (linear.state_dict(), total_loss / items_processed)
