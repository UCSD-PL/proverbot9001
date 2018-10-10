#!/usr/bin/env python3

import re
import time
import argparse
import threading

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim import Optimizer
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
import torch.utils.data as torchdata
import torch.cuda
import models.args as stdargs
import data
from data import ClassifySequenceDataset, read_text_data, filter_data, \
    encode_seq_classify_data, normalizeSentenceLength
from context_filter import get_context_filter
from tokenizer import tokenizers

from util import *
from models.components import SimpleEmbedding
from models.tactic_predictor import TacticPredictor
from models.term_autoencoder import EncoderRNN
from serapi_instance import get_stem

from typing import Dict, List, Union, Any, Tuple, Iterable, Callable
from typing import cast, overload

class AutoClassPredictor(TacticPredictor):
    def load_saved_state(self, autoclass_state_filename : str) -> None:
        checkpoint = torch.load(autoclass_state_filename)

        if not 'context-filter' in checkpoint:
            print("Warning: could not find context filter in saved autoclass state, "
                  "using default...")
            checkpoint['context-filter'] = "default"

        assert checkpoint['tokenizer']
        assert checkpoint['tokenizer-name']
        assert checkpoint['stem-embedding']
        assert checkpoint['decoder']
        assert checkpoint['num-decoder-layers']
        assert checkpoint['hidden-size']
        assert checkpoint['context-filter']
        assert checkpoint['learning-rate']
        assert checkpoint['training-loss']
        assert checkpoint['epoch']

        self.options = [("tokenizer", checkpoint['tokenizer-name']),
                        ("# input keywords", checkpoint['num-keywords']),
                        ("max input length", checkpoint['max-length']),
                        ("# encoder layers", checkpoint['num-encoder-layers']),
                        ("hidden size", checkpoint['hidden-size']),
                        ("# decoder layers", checkpoint['num-decoder-layers']),
                        ("context filter", checkpoint['context-filter']),
                        ("optimizer (autoencoder)",
                         checkpoint['autoenc-optimizer']),
                        ("optimizer (classifier)",
                         checkpoint['optimizer']),
                        ("learning rate (autoencoder)",
                         checkpoint['autoenc-learning-rate']),
                        ("learning rate (classifier)",
                         checkpoint['learning-rate']),
                        ("training loss (autoencoder)",
                         "{:.4f}".format(checkpoint['autoenc-training-loss'])),
                        ("training loss (classifier)",
                         "{:.4f}".format(checkpoint['training-loss'])),
                        ("# epochs (autoencoder)",
                         checkpoint['autoenc-epoch'] + 1),
                        ("# epochs (classifier)",
                         checkpoint['epoch'] + 1)]

        self.tokenizer = checkpoint['tokenizer']
        self.embedding = checkpoint['stem-embedding']
        self.encoder = maybe_cuda(EncoderRNN(self.tokenizer.numTokens(),
                                             checkpoint['hidden-size'],
                                             checkpoint['num-encoder-layers']))
        self.encoder.load_state_dict(checkpoint['encoder'])
        print("Have {} embedding tokens".format(self.embedding.num_tokens()))
        self.decoder = maybe_cuda(
            ClassifierDNN(checkpoint['hidden-size'],
                          self.embedding.num_tokens(),
                          checkpoint['num-decoder-layers']))
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.max_length = checkpoint['max-length']
        self.context_filter = checkpoint['context-filter']

    def __init__(self, options : Dict[str, Any]) -> None:
        self.load_saved_state(options["filename"])
        self.criterion = maybe_cuda(nn.NLLLoss())
        self.lock = threading.Lock()

    def predictDistribution(self, in_data : Dict[str, Union[List[str], str]]) \
        -> torch.FloatTensor:
        return self.decoder.run(self.encoder.run(LongTensor(normalizeSentenceLength(
            self.tokenizer.toTokenList(in_data["goal"]),
            self.max_length)).view(1, -1)))

    def predictKTactics(self, in_data : Dict[str, Union[List[str], str]], k : int) \
        -> List[Tuple[str, float]]:
        self.lock.acquire()
        prediction_distribution = self.predictDistribution(in_data)
        certainties_and_idxs = prediction_distribution.view(-1).topk(k)
        results = [(self.embedding.decode_token(stem_idx.data[0]) + ".",
                    math.exp(certainty.data[0]))
                   for certainty, stem_idx in zip(*certainties_and_idxs)]
        self.lock.release()
        return results

    def predictKTacticsWithLoss(self, in_data : Dict[str, Union[List[str], str]], k : int,
                                correct : str) -> Tuple[List[Tuple[str, float]], float]:
        self.lock.acquire()
        prediction_distribution = self.predictDistribution(in_data)
        correct_stem = get_stem(correct)
        if self.embedding.has_token(correct_stem):
            output_var = maybe_cuda(Variable(
                torch.LongTensor([self.embedding.encode_token(correct_stem)])))
            loss = self.criterion(prediction_distribution, output_var).item()
        else:
            loss = 0

        certainties_and_idxs = prediction_distribution.view(-1).topk(k)
        results = [(self.embedding.decode_token(stem_idx.item()) + ".",
                    math.exp(certainty.item()))
                   for certainty, stem_idx in zip(*certainties_and_idxs)]

        self.lock.release()
        return results, loss

    def getOptions(self) -> List[Tuple[str, str]]:
        return self.options

class ClassifierDNN(nn.Module):
    def __init__(self, hidden_size : int, output_vocab_size : int,
                 num_layers : int, batch_size : int=1) -> None:
        super(ClassifierDNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.layers = [maybe_cuda(nn.Linear(hidden_size, hidden_size))
                       for _ in range(num_layers)]
        self.out_layer = maybe_cuda(nn.Linear(hidden_size, output_vocab_size))
        self.softmax = maybe_cuda(nn.LogSoftmax(dim=1))
        pass

    def forward(self, input : torch.FloatTensor) -> torch.FloatTensor:
        layer_values = input.view(self.batch_size, -1)
        for i in range(self.num_layers):
            layer_values = F.relu(layer_values)
            layer_values = self.layers[i](layer_values)
        return self.softmax(self.out_layer(layer_values))

    def run(self, sentence : torch.FloatTensor) -> torch.FloatTensor:
        result = self(sentence)
        return result

Checkpoint = Tuple[Dict[Any, Any], float]

def train(dataset : ClassifySequenceDataset,
          autoencoder : EncoderRNN, max_length : int,
          hidden_size : int, output_vocab_size : int, num_layers : int, batch_size : int,
          learning_rate : float, gamma : float, epoch_step : int, num_epochs : int,
          print_every : int, optimizer_f : Callable[..., Optimizer]) \
          -> Iterable[Checkpoint]:
    print("Initializing PyTorch...")
    in_stream = [normalizeSentenceLength(goal, max_length)
                 for goal, tactic in dataset]
    out_stream = [tactic for goal, tactic in dataset]
    dataloader = \
        torchdata.DataLoader(torchdata.TensorDataset(torch.LongTensor(in_stream),
                                                     torch.LongTensor(out_stream)),
                             batch_size=batch_size, num_workers=0,
                             shuffle=True, pin_memory=True, drop_last=True)

    classifier = maybe_cuda(ClassifierDNN(hidden_size, output_vocab_size,
                                          num_layers, batch_size))
    optimizer = optimizer_f(classifier.parameters(), lr=learning_rate)
    criterion = maybe_cuda(nn.NLLLoss())
    adjuster = scheduler.StepLR(optimizer, epoch_step, gamma)

    start=time.time()
    num_items = len(dataset) * num_epochs
    total_loss = 0

    print("Training...")
    for epoch in range(num_epochs):
        print("Epoch {}".format(epoch))
        adjuster.step()
        for batch_num, (input_batch, output_batch) in enumerate(dataloader):

            # Reset the optimizer
            optimizer.zero_grad()

            # Run the classifier on pre-encoded vectors
            encoded_input_batch = autoencoder.run(cast(torch.LongTensor, input_batch))
            prediction_distribution = classifier.run(encoded_input_batch)

            # Get the loss
            output_var = maybe_cuda(Variable(output_batch))
            loss = criterion(prediction_distribution, output_var)

            # Update the weights
            loss.backward()
            optimizer.step()

            # Report progress
            items_processed = (batch_num + 1) * batch_size + epoch * len(dataset)
            total_loss += loss.item() * batch_size
            assert isinstance(total_loss, float)

            if (batch_num + 1) % print_every == 0:

                progress = items_processed / num_items
                print("{} ({:7} {:5.2f}%) {:.4f}".
                      format(timeSince(start, progress),
                             items_processed, progress * 100,
                             total_loss / items_processed))

        yield (classifier.state_dict(), total_loss / items_processed)

def main(arg_list : List[str]) -> None:
    parser = argparse.ArgumentParser(description="Autoencoder for coq terms")
    parser.add_argument("scrape_file")
    parser.add_argument("autoencoder_weights")
    parser.add_argument("save_file")
    parser.add_argument("--num-epochs", dest="num_epochs", default=15, type=int)
    parser.add_argument("--batch-size", dest="batch_size", default=256, type=int)
    parser.add_argument("--max-tuples", dest="max_tuples", default=None, type=int)
    parser.add_argument("--print-every", dest="print_every", default=10, type=int)
    parser.add_argument("--learning-rate", dest="learning_rate",
                        default=.7, type=float)
    parser.add_argument("--gamma", default=.9, type=float)
    parser.add_argument("--epoch-step", dest="epoch_step", default=5, type=int)
    parser.add_argument("--optimizer",
                        choices=list(stdargs.optimizers.keys()), type=str,
                        default=list(stdargs.optimizers.keys())[0])
    parser.add_argument("--num-classifier-layers", dest="num_classifier_layers",
                        default=3, type=int)
    args = parser.parse_args(arg_list)
    print("Loading autoencoder state...")
    autoenc_state = torch.load(args.autoencoder_weights)
    print("Loading data...")
    raw_data = read_text_data(args.scrape_file, args.max_tuples)
    print("Read {} raw input-output pairs".format(len(raw_data)))
    print("Filtering data based on predicate...")
    if 'context-filter' in autoenc_state:
        cfilter = autoenc_state['context-filter']
    else:
        print("Warning: could not find context filter in saved autoencoder state, using default...")
        cfilter = "default"
    filtered_data = filter_data(raw_data,
                                get_context_filter(cfilter))
    print("{} input-output pairs left".format(len(filtered_data)))
    print("Encoding data...")
    start = time.time()
    tokenizer = autoenc_state['tokenizer']
    embedding = SimpleEmbedding()
    dataset = [(tokenizer.toTokenList(goal), embedding.encode_token(get_stem(tactic)))
               for hyps, goal, tactic in filtered_data]
    timeTaken = time.time() - start
    print("Encoded data in {:.2f}".format(timeTaken))

    loadedAutoencoder = maybe_cuda(EncoderRNN(tokenizer.numTokens(),
                                              autoenc_state['hidden-size'],
                                              autoenc_state['num-encoder-layers'],
                                              args.batch_size))
    loadedAutoencoder.load_state_dict(autoenc_state['encoder'])
    checkpoints = train(dataset, loadedAutoencoder, autoenc_state['max-length'],
                        autoenc_state['hidden-size'], embedding.num_tokens(),
                        args.num_classifier_layers, args.batch_size,
                        args.learning_rate, args.gamma, args.epoch_step, args.num_epochs,
                        args.print_every, stdargs.optimizers[args.optimizer])

    for epoch, (decoder_state, training_loss) in enumerate(checkpoints):
        state = {'epoch': epoch,
                 'training-loss' : training_loss,
                 'autoenc-training-loss' : autoenc_state['training-loss'],
                 'autoenc-epoch' : autoenc_state['epoch'],
                 'tokenizer' : tokenizer,
                 'tokenizer-name' : autoenc_state['tokenizer-name'],
                 'optimizer' : args.optimizer,
                 'autoenc-optimizer' : autoenc_state['optimizer'],
                 'learning-rate' : args.learning_rate,
                 'autoenc-learning-rate' : autoenc_state['learning-rate'],
                 'encoder' : autoenc_state['encoder'],
                 'decoder' : decoder_state,
                 'num-decoder-layers' : args.num_classifier_layers,
                 'num-encoder-layers' : autoenc_state['num-encoder-layers'],
                 'context-filter' : cfilter,
                 'max-length' : autoenc_state['max-length'],
                 'hidden-size' : autoenc_state['hidden-size'],
                 'num-keywords' : autoenc_state['num-keywords'],
                 'stem-embedding' : embedding,
        }
        with open(args.save_file, 'wb') as f:
            print("=> Saving checkpoint at epoch {}".
                  format(epoch))
            torch.save(state, f)
