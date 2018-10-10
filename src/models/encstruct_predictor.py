#!/usr/bin/env python3

import argparse
import time
import threading
from typing import Dict, List, Union, Any, Tuple, Iterable, cast, Callable
from itertools import takewhile

from models.encdecrnn_predictor import inputFromSentence
from tokenizer import Tokenizer, tokenizers, get_symbols, make_keyword_tokenizer_relevance
from models.tactic_predictor import TacticPredictor
from models.args import add_std_args, optimizers
from models.components import SimpleEmbedding
from context_filter import get_context_filter

from data import read_text_data, filter_data, RawDataset, Sentence
from util import *
from util import _inflate
import serapi_instance

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim import Optimizer
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
import torch.utils.data as data
import torch.cuda

TacticStructure = Tuple[int, List[int]]
StructDataset = List[Tuple[List[Sentence], Sentence, TacticStructure]]

class EncStructPredictor(TacticPredictor):
    def load_saved_state(self, filename : str) -> None:
        checkpoint = torch.load(filename)
        assert checkpoint['tokenizer']
        assert checkpoint['tokenizer-name']
        assert checkpoint['embedding']
        assert checkpoint['context-filter']
        assert checkpoint['max-length']
        assert checkpoint['optimizer']
        assert checkpoint['num-encoder-layers']
        assert checkpoint['num-decoder-layers']
        assert checkpoint['encoder']
        assert checkpoint['stem-decoder']
        assert checkpoint['arg-decoder']

        self.options = [("tokenizer", checkpoint["tokenizer-name"]),
                        ("optimizer", checkpoint["optimizer-name"]),
                        ("context filter", checkpoint["context-filter"]),
                        ("# encoder layers", checkpoint["num-encoder-layers"]),
                        ("# decoder layers", checkpoint["num-decoder-layers"])]

        self.tokenizer = checkpoint['tokenizer']
        self.embedding = checkpoint['embedding']
        self.max_length = checkpoint["max-length"]
        self.encoder = maybe_cuda(RNNEncoder(self.tokenizer.numTokens(),
                                             checkpoint['hidden-size'],
                                             checkpoint['num-encoder-layers']))
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.stem_decoder = maybe_cuda(StemClassifier(checkpoint['hidden-size'],
                                                      self.embedding.num_tokens(),
                                                      checkpoint['num-decoder-layers']))
        self.stem_decoder.load_state_dict(checkpoint['stem-decoder'])
        self.arg_decoder = maybe_cuda(ArgumentDecoder(checkpoint['hidden-size'],
                                                      checkpoint['num-decoder-layers']))
        self.arg_decoder.load_state_dict(checkpoint['arg-decoder'])
        pass
    def __init__(self, options : Dict[str, Any]) -> None:
        assert(options["filename"])
        self.load_saved_state(options["filename"])
        self.lock = threading.Lock()
        pass
    def predictDistribution(self, in_data : Dict[str, str]) -> torch.FloatTensor:
        pass
    def predictKTactics(self, in_data : Dict[str, Union[List[str], str]], k : int) \
        -> List[Tuple[str, float]]:
        self.lock.acquire()
        in_sentence = LongTensor(inputFromSentence(
            self.tokenizer.toTokenList(in_data["goal"]),
            self.max_length))\
            .view(1, -1)
        encoded_vector = self.encoder.run(in_sentence)
        prediction_structures, certainties = \
            self.decodeKTactics(encoded_vector, k, cast(List[str], in_data["hyps"]),
                                k * k, 3)
        self.lock.release()
        return [(decode_tactic_structure(self.tokenizer, self.embedding,
                                         structure, cast(List[str], in_data["hyps"])),
                 certainty)
                for structure, certainty in zip(prediction_structures, certainties)]
    def decodeKTactics(self, encoded_vector : torch.FloatTensor, k : int,
                       hyps : List[str],
                       beam_width : int, max_args : int) -> \
                       Tuple[List[TacticStructure], List[float]]:
        stem_distribution = self.stem_decoder.run(encoded_vector)
        certainties, idxs = stem_distribution.view(-1).topk(beam_width)
        scores : torch.FloatTensor = certainties
        next_idxs = [idxs]
        next_hidden = _inflate(encoded_vector, beam_width)
        back_pointers : List[torch.LongTensor] = []

        for i in range(max_args):
            next_arg_dist, next_hidden = \
                self.arg_decoder.run(next_idxs[-1], next_hidden)
            beam_scores = next_arg_dist + scores.unsqueeze(1).expand_as(next_arg_dist)
            best_scores, best_beam_idxs = beam_scores.view(-1).topk(beam_width)
            scores = best_scores
            back_pointers_row = best_beam_idxs / next_arg_dist.size(1)
            back_pointers.append(back_pointers_row)
            next_idxs.append(best_beam_idxs - back_pointers_row * next_arg_dist.size(1))
            next_hidden = next_hidden.index_select(1,
                                                   cast(torch.LongTensor,
                                                        back_pointers_row.squeeze()))

        results = []
        for i in range(beam_width):
            result = []
            predecessor = i
            for j in range(len(back_pointers) - 1, -1, -1):
                result.append(next_idxs[j + 1][predecessor])
                predecessor = back_pointers[j][predecessor]

            results.append((predecessor, result[::-1]))
        return results[:k], scores[:k]
    def predictKTacticsWithLoss(self, in_data : Dict[str, Union[str, List[str]]], k : int,
                                correct : str) -> Tuple[List[Tuple[str, float]], float]:
        return self.predictKTactics(in_data, k), 1.0
        pass
    def getOptions(self) -> List[Tuple[str, str]]:
        return self.options

class RNNEncoder(nn.Module):
    def __init__(self, input_vocab_size : int, hidden_size : int,
                 num_layers : int, batch_size : int = 1) -> None:
        super(RNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding = maybe_cuda(nn.Embedding(input_vocab_size, hidden_size))
        self.gru = maybe_cuda(nn.GRU(hidden_size, hidden_size))
        pass
    pass
class StemClassifier(nn.Module):
    def __init__(self, hidden_size : int, output_vocab_size : int,
                 num_layers : int, batch_size : int = 1) -> None:
        super(StemClassifier, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.layers = [maybe_cuda(nn.Linear(hidden_size, hidden_size))
                       for _ in range(num_layers - 1)]
        self.out_layer = maybe_cuda(nn.Linear(hidden_size, output_vocab_size))
        self.softmax = maybe_cuda(nn.LogSoftmax(dim=1))
    pass
class ArgumentDecoder(nn.Module):
    def __init__(self, hidden_size : int, num_layers : int, batch_size : int = 1)\
        -> None:
        super(ArgumentDecoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
    pass

Checkpoint = Tuple[Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], float]

def train(dataset : StructDataset, input_vocab_size : int, output_vocab_size : int,
          hidden_size : int, learning_rate : float,
          num_encoder_layers : int, num_decoder_layers : int,
          max_length : int, num_epochs : int, batch_size : int,
          print_every : int, optimizer_f : Callable[..., Optimizer]) \
          -> Iterable[Checkpoint]:
    print("Initializing PyTorch...")
    hyps_stream = [[inputFromSentence(hyp, max_length) for hyp in hyps]
                   for hyps, goal, struct in dataset]
    pass

def decode_tactic_structure(term_tokenizer : Tokenizer, stem_embedding : SimpleEmbedding,
                            struct : TacticStructure, hyps : List[str]) -> str:
    def get_var(idx : int) -> str:
        if idx == 0:
            return "UNKNOWN"
        else:
            return serapi_instance.get_first_var_in_hyp(hyps[idx-1])
    stem_idx, arg_hyp_idxs = struct
    return " ".join([stem_embedding.decode_token(stem_idx)] +
                    [get_var(hyp_idx) for hyp_idx in takewhile(lambda idx: idx > 0,
                                                               arg_hyp_idxs)])

def encode_seq_structural_data(data : RawDataset,
                               context_tokenizer_type : \
                               Callable[[List[str], int], Tokenizer],
                               num_keywords : int,
                               num_reserved_tokens: int) -> \
                               Tuple[StructDataset, Tokenizer, SimpleEmbedding]:
    hyps_and_goals = [hyp_or_goal
                      for hyp_and_goal in [hyps + [goal] for hyps, goal, tactic in data]
                      for hyp_or_goal in hyp_and_goal]
    context_tokenizer = make_keyword_tokenizer_relevance(hyps_and_goals,
                                                         context_tokenizer_type,
                                               num_keywords, num_reserved_tokens)
    embedding = SimpleEmbedding()

    encodedData = []
    for hyps, goal, tactic in data:
        stem, rest = serapi_instance.split_tactic(tactic)
        encodedData.append(([context_tokenizer.toTokenList(hyp) for hyp in hyps],
                            context_tokenizer.toTokenList(goal),
                            (embedding.encode_token(stem),
                            [hyp_index(hyps, arg) for arg in get_symbols(rest)])))

    return encodedData, context_tokenizer, embedding

# Returns *one* indexed hypothesis indices, zero for "not found"
def hyp_index(hyps : List[str], arg : str) -> int:
    for i, hyp in enumerate(hyps, start=1):
        if arg in hyp.split(":")[0]:
            return i
    return 0

def main(arg_list : List[str]) -> None:
    argparser = argparse.ArgumentParser(description=
                                        "a structural pytorch model for proverbot")
    add_std_args(argparser)
    argparser.add_argument("--num-decoder-layers", dest="num_decoder_layers",
                           default=3, type=int)
    args = argparser.parse_args(arg_list)
    filtered_data = get_text_data(args.scrape_file, args.context_filter, verbose=True)
    print("Encoding data...")
    start = time.time()

    dataset, tokenizer, embedding = encode_seq_structural_data(filtered_data,
                                                               tokenizers[args.tokenizer],
                                                               args.num_keywords, 2)
    timeTaken = time.time() - start
    print("Encoded data in {:.2f}".format(timeTaken))

    checkpoints = train(dataset, tokenizer.numTokens(), embedding.num_tokens(),
                        args.hidden_size, args.learning_rate,
                        args.num_encoder_layers, args.num_decoder_layers,
                        args.max_length, args.num_epochs, args.batch_size,
                        args.print_every, optimizers[args.optimizer])

    for epoch, (encoder_state, stem_decoder_state, arg_decoder_state, training_loss) \
        in enumerate(checkpoints):
        state = {'epoch':epoch,
                 'training-loss': training_loss,
                 'tokenizer':tokenizer,
                 'tokenizer-name':args.tokenizer,
                 'optimizer':args.optimizer,
                 'learning-rate':args.learning_rate,
                 'embedding': embedding,
                 'encoder':encoder_state,
                 'stem-decoder':stem_decoder_state,
                 'arg-decoder':arg_decoder_state,
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
