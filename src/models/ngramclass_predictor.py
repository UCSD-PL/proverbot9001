#!/usr/bin/env python3
import pdb
import argparse
import time
import math
from typing import Dict, Any, List, Tuple, Iterable, cast, Union

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.optim.lr_scheduler as scheduler
import torch.utils.data as data
from torch.utils.data.dataset import Dataset

from models.tactic_predictor import TacticPredictor

from tokenizer import tokenizers
from data import read_text_data, filter_data, Sentence, \
    encode_ngram_classify_data, encode_ngram_classify_input, encode_seq_classify_data
from context_filter import get_context_filter
from util import *
from serapi_instance import get_stem

class NGramClassifyPredictor(TacticPredictor):
    def load_saved_state(self, filename : str) -> None:
        checkpoint = torch.load(filename)
        assert checkpoint['stem-embeddings']
        self.embedding = checkpoint['stem-embeddings']
        self.tokenizer = checkpoint['tokenizer']
        self.num_grams = checkpoint['num-grams']
        self.linear = maybe_cuda(nn.Linear(self.tokenizer.numTokens() ** self.num_grams,
                                           self.embedding.num_tokens()))
        self.linear.load_state_dict(checkpoint['linear-state'])
        self.lsoftmax = maybe_cuda(nn.LogSoftmax(dim=1))

        self.options = checkpoint['options']
        self.criterion = maybe_cuda(nn.NLLLoss())
        pass

    def getOptions(self) -> List[Tuple[str, str]]:
        return [("classifier", str(True)),
                ("num-stems", str(self.embedding.num_tokens())),
                ] + self.options + [("num-grams", self.num_grams)]

    def __init__(self, options : Dict[str, Any]) -> None:
        assert options["filename"]
        self.load_saved_state(options["filename"])

    def predictDistribution(self, in_data : Dict[str, Union[str, List[str]]]) \
        -> torch.FloatTensor:
        goal = cast(str, in_data["goal"])
        in_vec = Variable(FloatTensor(encode_ngram_classify_input(goal, self.num_grams, self.tokenizer)))\
                 .view(1, -1)
        return self.lsoftmax(self.linear(in_vec))

    def predictKTactics(self, in_data : Dict[str, Union[str, List[str]]], k : int) \
        -> List[Tuple[str, float]]:
        distribution = self.predictDistribution(in_data)
        probs_and_indices = distribution.squeeze().topk(k)
        return [(self.embedding.decode_token(idx.data[0]) + ".",
                 math.exp(certainty.data[0]))
                for certainty, idx in probs_and_indices]

    def predictKTacticsWithLoss(self, in_data : Dict[str, Union[str, List[str]]], k : int,
                                correct : str) -> Tuple[List[Tuple[str, float]], float]:
        distribution = self.predictDistribution(in_data)
        stem = get_stem(correct)
        if self.embedding.has_token(stem):
            output_var = maybe_cuda(
                Variable(torch. LongTensor([self.embedding.encode_token(stem)])))
            loss = self.criterion(distribution, output_var).item()
        else:
            loss = 0

        probs_and_indices = distribution.squeeze().topk(k)
#        import pdb
#        pdb.set_trace()
        predictions = [(self.embedding.decode_token(idx.item()) + ".",
                        math.exp(certainty.item()))
                       for certainty, idx in zip(*probs_and_indices)]
        return predictions, loss

Checkpoint = Tuple[Dict[Any, Any], float]

def main(args_list : List[str]) -> None:
    parser = argparse.ArgumentParser(description=
                                     "A second-tier predictor which predicts tactic "
                                     "stems based on word frequency in the goal")
    parser.add_argument("--learning-rate", dest="learning_rate", default=.3, type=float)
    parser.add_argument("--num-epochs", dest="num_epochs", default=20, type=int)
    parser.add_argument("--batch-size", dest="batch_size", default=256, type=int)
    parser.add_argument("--print-every", dest="print_every", default=10, type=int)
    parser.add_argument("--epoch-step", dest="epoch_step", default=5, type=int)
    parser.add_argument("--gamma", dest="gamma", default=0.5, type=float)
    parser.add_argument("--optimizer", default="SGD",
                        choices=list(optimizers.keys()), type=str)
    parser.add_argument("--context-filter", dest="context_filter",
                        type=str, default="default")
    parser.add_argument("-n", "--num-grams", dest="num_grams", default=1, type=int)
    parser.add_argument("scrape_file")
    parser.add_argument("save_file")
    args = parser.parse_args(args_list)
    print("Loading dataset...")

    raw_dataset = read_text_data(args.scrape_file)
    filtered_dataset = filter_data(raw_dataset, get_context_filter(args.context_filter))
#    samples, tokenizer, embedding = encode_seq_classify_data(filtered_dataset,
#                                                             tokenizers["no-fallback"],
#                                                             100, 2)
    samples, tokenizer, embedding = encode_ngram_classify_data(filtered_dataset,
                                                               args.num_grams,
                                                               tokenizers["no-fallback"],
#                                                             tokenizers["chars-fallback"],
                                                               100, 2)
    checkpoints = train(samples, args.num_grams, tokenizer.numTokens(), args.learning_rate,
                        args.num_epochs, args.batch_size,
                        embedding.num_tokens(), args.print_every,
                        args.gamma, args.epoch_step, args.optimizer)

    for epoch, (linear_state, loss) in enumerate(checkpoints, start=1):
        state = {'epoch':epoch,
                 'text-encoder':tokenizer,
                 'linear-state': linear_state,
                 'stem-embeddings': embedding,
                 'tokenizer': tokenizer,
                 'num-grams': args.num_grams,
                 'options': [
                     ("# epochs", str(epoch)),
                     ("learning rate", str(args.learning_rate)),
                     ("batch size", str(args.batch_size)),
                     ("epoch step", str(args.epoch_step)),
                     ("gamma", str(args.gamma)),
                     ("dataset size", str(len(samples))),
                     ("optimizer", args.optimizer),
                     ("training loss", "{:10.2f}".format(loss)),
                     ("context filter", args.context_filter),
                 ]}
        with open(args.save_file, 'wb') as f:
            print("=> Saving checkpoint at epoch {}".
                  format(epoch))
            torch.save(state, f)

optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
}

def padInputs(inputs):
    new_inputs = []
    from collections import Counter
    c = Counter([len(i) for i in inputs])
    pdb.set_trace()    
    for i in range(len(inputs)):
        c[len(inputs[i])] += 1
        if i % 1000 == 0:
            print(c.most_common(5))
    max_len = max(len(i) for i in inputs)
    for i in range(len(inputs)):
        if len(inputs[i]) < max_len:
            new_inputs.append(inputs[i] + ([0] * (max_len - len(inputs[i]))))
    return new_inputs

def unpadInputs(inputs):
    new_inputs = []
    for i in range(len(inputs)):
        new_inputs.append([j for j in inputs[i] if j != 0])
    return new_inputs


def inputFromSentence(sentence : Sentence, max_length : int) -> Sentence:
    if len(sentence) > max_length:
        sentence = sentence[:max_length]
    if len(sentence) < max_length:
        sentence.extend([0] * (max_length - len(sentence)))
    return sentence

def getSingleSparseFloatTensor(inputs):
#    if not inputs.elements:
#        return torch.sparse.FloatTensor(torch.Size([1, len(inputs)])).to_dense()
#    return torch.FloatTensor(inputs)
#    return torch.sparse.FloatTensor(torch.LongTensor(list(inputs.elements.keys())), torch.FloatTensor(list(inputs.elements.values())))
    if not inputs.elements:
        return torch.sparse.FloatTensor(torch.Size([1, len(inputs)])).to_dense()
    indecies = []
    values = []
    for j in inputs.elements:
        indecies.append((0,j))
        values.append(inputs.elements[j])
    i = torch.LongTensor(indecies)
    v = torch.FloatTensor(values)
    try:
        x = torch.sparse.FloatTensor(i.t(), v, torch.Size([1,len(inputs)])).to_dense()
    except:
        pdb.set_trace()
        
#    x = torch.sparse.FloatTensor(i, v, torch.Size([len(inputs)])).to_dense()
    return x

def getSparseFloatTensor(inputs):
#    import pdb
    pdb.set_trace()
    
    indecies = []
    values = []
    for i in range(len(inputs)):
        for j in inputs[i].elements:
            indecies.append((i,j))
            values.append(inputs[i].elements[j])
    i = torch.LongTensor(indecies)
    v = torch.FloatTensor(values)
    x = torch.sparse.FloatTensor(i.t(), v, torch.Size([len(inputs), len(inputs[0])])).to_dense()
    return x

class CustomDataset(Dataset):
    def __init__(self, inputs, outputs):
        assert len(inputs) == len(outputs)
        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, index):
#        return self.inputs[index].elements, self.outputs[index]
        return getSingleSparseFloatTensor(self.inputs[index]), self.outputs[index]
#        return (self.inputs[index], self.outputs[index])

    def __len__(self):
        return len(self.inputs)

def train(dataset, num_grams : int, num_tokens : int, learning_rate : float,
          num_epochs : int, batch_size : int, num_stems: int, print_every : int,
          gamma : float, epoch_step : int, optimizer_type : str) -> Iterable[Checkpoint]:
    print("Initializing PyTorch...")
#    print(dataset)
#    assert len(dataset[0][0]) > 10 and len(dataset[0][0]) < 1000
#    linear = maybe_cuda(nn.Linear( len(dataset[0][0]), num_stems))
    linear = maybe_cuda(nn.Linear( num_tokens ** num_grams, num_stems))
    lsoftmax = maybe_cuda(nn.LogSoftmax(1))
    inputs, outputs = zip(*dataset)
#    new_inputs = []
#    for i in range(len(inputs)):
#        new_inputs.append(inputFromSentence(inputs[i], 100))
#    inputs = new_inputs
#    pdb.set_trace()

    dataloader = data.DataLoader(
        CustomDataset(
            inputs,
            outputs),
#        data.TensorDataset(
#            torch.sparse.FloatTensor(i.t(), v, torch.Size([len(inputs), len(inputs[0])])).to_dense(),
#            getSparseFloatTensor(inputs),
#            torch.FloatTensor(inputs),
#            Variable(inputs),
#            torch.LongTensor(outputs)),
        batch_size=batch_size, num_workers=0,
        shuffle=True, pin_memory=True, drop_last=True)
    optimizer = optimizers[optimizer_type](linear.parameters(), lr=learning_rate)
    criterion = maybe_cuda(nn.NLLLoss())
    adjuster = scheduler.StepLR(optimizer, epoch_step, gamma=gamma)

    start=time.time()
#    num_items = len(dataset) * num_epochs
    num_items = len(inputs) * num_epochs
    total_loss = 0
#    pdb.set_trace()
    print("Training...")
    for epoch in range(num_epochs):
        print("Epoch {}".format(epoch))
#        print("1")
        adjuster.step()
#        print("2")
#        pdb.set_trace()
        for batch_num, (input_batch, output_batch) in enumerate(dataloader):
#        for i in range(1):
#            input_batch = dataloader.dataset.tensors[0]
#            output_batch = dataloader.dataset.tensors[1]
#            print("3")
            input_batch = input_batch.squeeze()
            optimizer.zero_grad()
#            print("4")
#            pdb.set_trace()
#            input_batch = getSparseFloatTensor(input_batch)
#            input_batch = unpadInputs(input_batch)
#            input_batch = [getNGramTokenbagVector(num_grams, context, num_tokens) \
#                           for context in input_batch]
            input_var = maybe_cuda(Variable(input_batch))
#            print("5")
            output_var = maybe_cuda(Variable(output_batch))
#            print("6")
            prediction_distribution = lsoftmax(linear(input_var))
#            print("7")
            loss = cast(torch.FloatTensor, 0) # type: torch.FloatTensor
#            print("8")
            loss += criterion(prediction_distribution, output_var)
#            print("9")

            loss.backward()
#            print("10")
            optimizer.step()
            total_loss += loss.item() * batch_size

            if (batch_num + 1) % print_every == 0:

                items_processed = (batch_num + 1) * batch_size + epoch * len(inputs) # * len(dataset)
                progress = items_processed / num_items
                print("{} ({:7} {:5.2f}%) {:.4f}".
                      format(timeSince(start, progress),
                             items_processed, progress * 100,
                             total_loss / items_processed))
        yield (linear.state_dict(), total_loss / items_processed)
